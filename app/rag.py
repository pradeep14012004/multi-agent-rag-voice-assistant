
from app.config import GROQ_API_KEY, GROQ_MODEL, MIN_RELEVANCE_SCORE


ABSTENTION = (
    "I couldn't find sufficiently relevant information in the "
    "indexed knowledge base. Please upload a document covering "
    "this topic or ask about information in your documents."
)


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
            f"domain: {metadata.get('domain', 'unknown')}\n"
            f"{hit['text']}"
        )

    context = "\n\n".join(context_parts)

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
                        f"Conversation memory (not factual evidence):\n"
                        f"{memory or '(none)'}\n\n"
                        f"Retrieved evidence:\n{context}\n\n"
                        f"Question: {question}\n\n"
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
            "Showing retrieved evidence instead.\n\n"
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
            f"\n**[{i}] {metadata.get('filename', 'document')} "
            f"— page {metadata.get('page', '?')}**\n"
            f"{hit['text'][:1100]}"
        )

    lines.append(
        "\n*This is an extractive response. "
        "Configure the Groq API key for generated answers.*"
    )

    return "\n".join(lines)
