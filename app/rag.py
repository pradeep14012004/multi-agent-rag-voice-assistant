from app.config import GROQ_API_KEY, GROQ_MODEL, MIN_RELEVANCE_SCORE


ABSTENTION = (
    "I couldn't find sufficiently relevant information in the "
    "indexed knowledge base. Please upload a document covering "
    "this topic or ask about information in your documents."
)

# These are display bands for retrieval evidence, not calibrated answer
# probabilities. The embedding similarity score is not a correctness probability.
STRONG_EVIDENCE_SCORE = 0.70
MODERATE_EVIDENCE_SCORE = 0.40


def assess_evidence(
    hits: list[dict],
    threshold: float | None = None,
) -> dict:
    """Summarize retrieved evidence using transparent heuristic score bands.

    The strongest cosine-similarity score is used as the evidence-strength
    indicator. Average score and supporting chunk count provide context.
    This is not a calibrated probability that the generated answer is correct.
    """
    active_threshold = MIN_RELEVANCE_SCORE if threshold is None else threshold
    relevant_hits = [
        hit for hit in hits
        if float(hit.get("score", 0.0)) >= active_threshold
    ]

    if not relevant_hits:
        return {
            "level": "insufficient",
            "label": "Insufficient evidence",
            "score": None,
            "average_score": None,
            "supporting_chunks": 0,
            "threshold": round(active_threshold, 3),
        }

    scores = [float(hit.get("score", 0.0)) for hit in relevant_hits]
    strongest_score = max(scores)
    average_score = sum(scores) / len(scores)

    if strongest_score >= STRONG_EVIDENCE_SCORE:
        level, label = "strong", "Strong evidence"
    elif strongest_score >= MODERATE_EVIDENCE_SCORE:
        level, label = "moderate", "Moderate evidence"
    else:
        level, label = "limited", "Limited evidence"

    return {
        "level": level,
        "label": label,
        "score": round(strongest_score, 3),
        "average_score": round(average_score, 3),
        "supporting_chunks": len(relevant_hits),
        "threshold": round(active_threshold, 3),
    }


def generate_answer(
    question: str,
    hits: list[dict],
    memory: str = ""
) -> str:

    if not hits:
        return ABSTENTION

    # Reject evidence below the configured relevance threshold.
    relevant_hits = [
        hit for hit in hits
        if hit.get("score", 0.0) >= MIN_RELEVANCE_SCORE
    ]

    if not relevant_hits:
        return ABSTENTION

    context_parts = []

    for i, hit in enumerate(relevant_hits, 1):
        metadata = hit.get("metadata", {})

        context_parts.append(
            f"[{i}] {metadata.get('filename', 'document')} | "
            f"page {metadata.get('page', '?')} | "
            f"domain: {metadata.get('domain', 'unknown')}\\n"
            f"{hit['text']}"
        )

    context = "\\n\\n".join(context_parts)

    if not GROQ_API_KEY:
        return extractive_answer(relevant_hits)

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence-grounded RAG assistant. "
                        "Answer using ONLY the retrieved evidence. "
                        "Conversation memory is only for understanding "
                        "follow-up references; it is NOT factual evidence. "
                        "Never use outside knowledge or invent facts. "
                        "If evidence does not answer the question, abstain. "
                        "Cite factual claims using source labels such as [1]. "
                        "Do not cite a source that does not support the claim. "
                        "Give concise, clear answers."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation memory (not factual evidence):\\n"
                        f"{memory or '(none)'}\\n\\n"
                        f"Retrieved evidence:\\n{context}\\n\\n"
                        f"Question: {question}\\n\\n"
                        "Answer using retrieved evidence only."
                    )
                }
            ],
            temperature=0.1,
            max_completion_tokens=1024
        )

        answer = response.choices[0].message.content

        if answer and answer.strip():
            return answer.strip()

        return extractive_answer(relevant_hits)

    except Exception as exc:
        print(f"Groq API error: {type(exc).__name__}: {exc}")

        return (
            "Groq generation is currently unavailable. "
            "Showing retrieved evidence instead.\\n\\n"
            + extractive_answer(relevant_hits)
        )


def extractive_answer(hits: list[dict]) -> str:

    if not hits:
        return ABSTENTION

    lines = [
        "The knowledge base contains the following relevant evidence:"
    ]

    for i, hit in enumerate(hits[:3], 1):
        metadata = hit.get("metadata", {})

        lines.append(
            f"\\n**[{i}] {metadata.get('filename', 'document')} "
            f"— page {metadata.get('page', '?')}**\\n"
            f"{hit['text'][:1100]}"
        )

    lines.append(
        "\\n*This is an extractive response. "
        "Configure the Groq API key for generated answers.*"
    )

    return "\\n".join(lines)
