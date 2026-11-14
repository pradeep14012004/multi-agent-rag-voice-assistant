from pathlib import Path
import streamlit as st
from app.ingestion import prepare_documents
from app.vector_store import add_chunks, collection_count, reset_collection
from app.agents import MultiAgentOrchestrator
from app.voice import browser_voice
from app.rag import assess_evidence, validate_citations

st.set_page_config(page_title="Multi-Agent RAG Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Multi-Agent RAG Knowledge Assistant")
st.caption("Document ingestion · semantic retrieval · five-agent orchestration · grounded answers · browser voice")

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MultiAgentOrchestrator()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

with st.sidebar:
    st.header("Knowledge Base")
    st.metric("Indexed chunks", collection_count())
    upload_domain = st.selectbox("Domain label for uploaded files", ["Computer Science", "Healthcare", "Other"])
    uploads = st.file_uploader("Upload knowledge documents", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True)
    if st.button("Ingest uploaded files", type="primary", disabled=not uploads, use_container_width=True):
        total, failures = 0, []
        progress = st.progress(0, text="Preparing documents…")
        for index, uploaded in enumerate(uploads):
            try:
                chunks = prepare_documents(uploaded.name, uploaded.getvalue(), upload_domain)
                added = add_chunks(chunks)
                total += added
                if not chunks:
                    failures.append(f"{uploaded.name}: no extractable text found")
            except Exception as exc:
                failures.append(f"{uploaded.name}: {type(exc).__name__}: {exc}")
            progress.progress((index + 1) / len(uploads), text=f"Processed {index + 1}/{len(uploads)} files")
        st.success(f"Indexed or updated {total} unique chunks from {len(uploads)} file(s).")
        for failure in failures:
            st.error(failure)
        st.rerun()

    st.divider()
    st.subheader("Sample knowledge base")
    if st.button("Ingest included two-domain samples", use_container_width=True):
        base = Path(__file__).parent / "data" / "sample_docs"
        total, failures = 0, []
        for path in sorted(base.iterdir()):
            if path.is_file() and path.suffix.lower() in {".txt", ".csv"}:
                domain = "Healthcare" if "health" in path.name.lower() else "Computer Science"
                try:
                    total += add_chunks(prepare_documents(path.name, path.read_bytes(), domain))
                except Exception as exc:
                    failures.append(f"{path.name}: {exc}")
        st.success(f"Indexed or updated {total} unique sample chunks.")
        for failure in failures:
            st.error(failure)
        st.rerun()

    st.divider()
    if st.button("Clear knowledge base", use_container_width=True):
        reset_collection()
        st.session_state.chat = []
        st.session_state.orchestrator.memory_agent.clear()
        st.success("Knowledge base and conversation memory cleared.")
        st.rerun()
    st.caption("Supported formats: PDF, DOCX, TXT and CSV. Scanned PDFs require OCR and are reported if no selectable text exists.")

st.subheader("Ask your knowledge base")
voice_result = browser_voice(
    answer=st.session_state.chat[-1]["answer"] if st.session_state.chat else "",
    key="browser_voice_control",
)
if isinstance(voice_result, dict) and voice_result.get("transcript"):
    st.session_state.question_input = voice_result["transcript"]

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("Your question", key="question_input", placeholder="e.g. What is supervised learning?")
    col1, col2 = st.columns([3, 1])
    with col1:
        domain_filter = st.selectbox("Search domain", ["All domains", "Computer Science", "Healthcare", "Other"])
    with col2:
        top_k = st.slider("Top-K results", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Enter or dictate a question first.")
    elif collection_count() == 0:
        st.warning("Your knowledge base is empty. Upload documents or ingest the included samples first.")
    else:
        with st.spinner("Query Understanding → Clarification → Retrieval → Response Generation → Memory…"):
            result = st.session_state.orchestrator.run(question, top_k=top_k, domain=domain_filter)
        if result["clarification"]:
            st.info(result["clarification"])
        else:
            st.session_state.chat.append({
                "question": question, "answer": result["answer"],
                "hits": result["hits"], "intent": result["intent"],
                "domain": domain_filter, "top_k": top_k,
                "evidence": assess_evidence(result["hits"]),
                "citation_validation": validate_citations(result["answer"], result["hits"]),
            })
            st.rerun()

if st.session_state.chat:
    st.subheader("Conversation")
    for item in reversed(st.session_state.chat):
        with st.container(border=True):
            st.markdown(f"**You:** {item['question']}")
            st.markdown(item["answer"])

            citation = item.get(
                "citation_validation",
                validate_citations(item["answer"], item["hits"])
            )
            st.markdown("#### Citation validation")
            if citation["status"] == "valid":
                st.success(citation["label"])
            elif citation["status"] == "invalid":
                st.error(citation["label"])
            elif citation["status"] == "missing":
                st.warning(citation["label"])
            else:
                st.info(citation["label"])

            if citation["references"]:
                st.caption(
                    f"Referenced sources: {citation['valid_references']} "
                    f"| Invalid references: {citation['invalid_references']}"
                )
            st.caption(
                "Validation checks whether citation numbers match retrieved sources; "
                "it does not verify that a source semantically supports each claim."
            )

            st.caption(f"Intent: {item['intent']} · Domain: {item['domain']} · Retrieved chunks: {len(item['hits'])}")
            with st.expander(f"Retrieved evidence ({len(item['hits'])} unique chunks)"):
                if not item["hits"]:
                    st.info("No sufficiently relevant evidence was retrieved. The assistant abstained rather than guessing.")
                for i, hit in enumerate(item["hits"], 1):
                    meta = hit["metadata"]
                    st.markdown(
                        f"**[{i}] {meta.get('filename', 'document')} · page {meta.get('page', '?')} · "
                        f"{meta.get('domain', 'unknown')} · relevance {hit['score']:.3f}**"
                    )
                    st.write(hit["text"])
    if st.button("Clear conversation only"):
        st.session_state.chat = []
        st.session_state.orchestrator.memory_agent.clear()
        st.rerun()

with st.expander("Agent pipeline"):
    st.markdown("1. **Query Understanding Agent** — normalizes the question and classifies factual, procedural or comparative intent.\n2. **Clarification Agent** — asks for more detail when the question is too vague.\n3. **Retrieval Agent** — embeds the query and searches the Chroma vector store with optional domain filtering.\n4. **Response Generation Agent** — creates a grounded answer (LLM if configured; extractive evidence otherwise) and abstains when no relevant evidence is found.\n5. **Conversation Memory Agent** — retains recent turns in the current session for follow-up context.\n\n**Multi-Agent Orchestrator** coordinates these agents in sequence.")

st.caption("Voice uses the browser Web Speech API. Microphone permission and a supported browser (Chrome recommended) are required; voice availability depends on browser/platform support.")
