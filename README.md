# Multi-Agent RAG Knowledge Assistant

A Streamlit-based, two-domain RAG application with document ingestion, Chroma vector search, five cooperating agents, conversation memory, evidence-grounded responses, retrieval evaluation, and browser voice input/output.

## Run locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

If `.env` already exists, do not overwrite it; preserve any API key stored there. The application works without an API key using extractive evidence responses. Add `OPENAI_API_KEY` to `.env` only if you want LLM-synthesized answers.

## Use the application

1. Click **Ingest included two-domain samples** to load the Computer Science and Healthcare examples.
2. Or upload PDF, DOCX, TXT, or CSV files and assign a domain label.
3. Ask a question and optionally filter retrieval by domain.
4. Use **Start voice input** and allow microphone access (Chrome is recommended). Use **Speak latest answer** for browser text-to-speech.
5. Expand **Retrieved evidence** to inspect source, page, domain, and similarity score.

The Web Speech API is browser-dependent and usually requires localhost or HTTPS. Scanned PDFs without selectable text are reported; OCR is not included.

## Retrieval evaluation

After ingesting the sample documents, run:

```bash
python evaluate.py
```

The script reports Hit@1, Hit@3, Hit@5 and MRR for labelled retrieval queries, plus abstention checks for intentionally unavailable questions. It writes `evaluation/last_results.json`. The included keyword relevance matching is a baseline proxy; manually judge relevance before treating results as final research metrics.

## Agent flow

Query Understanding → Clarification → Retrieval → Response Generation → Conversation Memory, coordinated by `MultiAgentOrchestrator`.

## Data and local state

Chroma data is stored in `chroma_db/` (ignored by Git). Re-uploading the same filename/page/chunk content updates the same deterministic chunk IDs instead of creating new copies. If upgrading an older local index, use **Clear knowledge base** once and ingest your documents again to remove legacy duplicate records.

## Tests

```bash
python -m unittest discover -s tests -v
```
