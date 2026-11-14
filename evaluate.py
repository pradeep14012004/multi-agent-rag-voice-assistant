"""Evaluate RAG retrieval, ranking and abstention using labelled queries."""

import json
from collections import defaultdict
from pathlib import Path

from app.vector_store import search, collection_count

ROOT = Path(__file__).resolve().parent
K_VALUES = (1, 3)


def is_relevant(hit, keywords):
    """Keyword-based relevance proxy; manual review is still required."""
    text = (
        hit.get("text", "") + " " +
        hit.get("metadata", {}).get("filename", "")
    ).casefold()
    return any(keyword.casefold() in text for keyword in keywords)


def evaluate():
    if collection_count() == 0:
        print("Knowledge base is empty. Ingest the sample documents first.")
        return

    cases = json.loads(
        (ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8")
    )

    retrieval_cases = [
        case for case in cases if case.get("expected_relevant", True)
    ]
    abstention_cases = [
        case for case in cases if not case.get("expected_relevant", True)
    ]

    hit_counts = {k: 0 for k in K_VALUES}
    precision_totals = {k: 0.0 for k in K_VALUES}
    reciprocal_ranks = []
    rows = []
    type_metrics = defaultdict(lambda: {"total": 0, "hit_at_3": 0})

    print("\nRAG RETRIEVAL EVALUATION")
    print("=" * 90)

    for case in retrieval_cases:
        query = case["query"]
        domain = case.get("domain", "All domains")
        query_type = case.get("query_type", "unspecified")
        keywords = case.get("expected_keywords", [])

        hits = search(
            query,
            top_k=3,
            domain=domain,
            min_score=0.40,
        )

        relevant_flags = [is_relevant(hit, keywords) for hit in hits]
        ranks = [
            rank for rank, relevant in enumerate(relevant_flags, 1)
            if relevant
        ]
        first_rank = min(ranks) if ranks else None

        reciprocal_ranks.append(1 / first_rank if first_rank else 0)

        for k in K_VALUES:
            top_flags = relevant_flags[:k]
            relevant_count = sum(top_flags)

            if first_rank is not None and first_rank <= k:
                hit_counts[k] += 1

            # Standard Precision@K uses K as the denominator.
            precision_totals[k] += relevant_count / k

        group = type_metrics[query_type]
        group["total"] += 1
        group["hit_at_3"] += int(first_rank is not None and first_rank <= 3)

        rows.append({
            "query": query,
            "query_type": query_type,
            "domain": domain,
            "first_relevant_rank": first_rank,
            "retrieved_count": len(hits),
            "relevant_chunks_at_5": sum(relevant_flags),
            "retrieved": [
                {
                    "rank": rank,
                    "filename": hit.get("metadata", {}).get("filename", ""),
                    "score": round(hit.get("score", 0), 4),
                    "keyword_relevant": relevant_flags[rank - 1],
                    "text": hit.get("text", ""),
                }
                for rank, hit in enumerate(hits, 1)
            ],
        })

        print(f"\n[{query_type.upper()}] {query}")
        print(f"  Domain: {domain}")
        print(f"  First relevant rank: {first_rank or 'Not found'}")
        print(f"  Retrieved chunks: {len(hits)}")

    total = len(retrieval_cases)

    print("\n" + "=" * 90)
    print("OVERALL RETRIEVAL METRICS")

    metrics = {}
    for k in K_VALUES:
        hit_rate = hit_counts[k] / total if total else 0
        precision = precision_totals[k] / total if total else 0

        metrics[f"hit_at_{k}"] = round(hit_rate, 4)
        metrics[f"precision_at_{k}"] = round(precision, 4)

        print(f"Hit@{k}:       {hit_counts[k]}/{total} = {hit_rate:.2%}")
        print(f"Precision@{k}:              {precision:.2%}")

    mrr = sum(reciprocal_ranks) / total if total else 0
    metrics["mrr"] = round(mrr, 4)
    print(f"MRR: {mrr:.4f}")

    print("\nQUERY-TYPE BREAKDOWN")
    for query_type, values in sorted(type_metrics.items()):
        count = values["total"]
        rate = values["hit_at_3"] / count if count else 0
        print(f"{query_type}: Hit@3 = {values['hit_at_3']}/{count} ({rate:.2%})")

    print("\nABSTENTION EVALUATION")
    abstention_rows = []

    for case in abstention_cases:
        hits = search(
            case["query"],
            top_k=3,
            domain=case.get("domain", "All domains"),
        )

        abstained = len(hits) == 0
        abstention_rows.append({
            "query": case["query"],
            "abstained": abstained,
            "retrieved_count": len(hits),
        })

        print(
            f"{'PASS' if abstained else 'REVIEW'}: "
            f"{case['query']} — {len(hits)} chunks"
        )

    abstention_total = len(abstention_rows)
    correct_abstentions = sum(row["abstained"] for row in abstention_rows)
    abstention_rate = (
        correct_abstentions / abstention_total if abstention_total else None
    )

    print(
        f"\nCorrect abstentions: {correct_abstentions}/{abstention_total}"
        if abstention_total else "\nNo abstention test cases."
    )

    report = {
        "evaluation_method": "Keyword-based relevance proxy",
        "retrieval_query_count": total,
        "retrieval_metrics": metrics,
        "query_type_metrics": {
            query_type: {
                "total": values["total"],
                "hit_at_3": round(
                    values["hit_at_3"] / values["total"], 4
                ),
            }
            for query_type, values in type_metrics.items()
        },
        "abstention": {
            "total": abstention_total,
            "correct": correct_abstentions,
            "accuracy": (
                round(abstention_rate, 4)
                if abstention_rate is not None else None
            ),
            "results": abstention_rows,
        },
        "retrieval_results": rows,
        "limitations": [
            "Keyword matching is only a relevance proxy.",
            "Manual relevance assessment is required for final reporting.",
            "Abstention evaluation checks retrieval output, not the full generated answer.",
        ],
    }

    output = ROOT / "evaluation" / "last_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDetailed report saved to: {output}")
    print("Review retrieved chunks manually before reporting final results.")


if __name__ == "__main__":
    evaluate()
