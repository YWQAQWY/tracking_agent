from src.eval.metrics.base import mean, ratio
from src.search.url_normalizer import URLNormalizer


def ranking_metrics(retrieved: list[str], relevant: list[str] | None, k: int) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    # Ranking is over unique identities, never duplicate chunks of one URL.
    ranking = list(dict.fromkeys(retrieved))[:k]
    gold = set(relevant or [])
    if not gold:
        return {"recall": None, "precision": None, "mrr": None}
    hits = len(set(ranking) & gold)
    return {
        "recall": hits / len(gold),
        "precision": hits / len(ranking) if ranking else 0.0,
        "mrr": next((1 / rank for rank, item in enumerate(ranking, 1) if item in gold), 0.0),
    }


def retrieval_metrics(rows, k: int) -> dict:
    urls, chunks, funnels = [], [], []
    normalizer = URLNormalizer()
    for row in rows:
        raw = row.raw
        if "final_evidence" in raw or "retrieved_urls" in raw:
            retrieved = raw.get("retrieved_urls", [item["url"] for item in raw.get("final_evidence", [])])
            gold = row.labels.get("relevant_urls")
            urls.append(ranking_metrics(
                [normalizer.normalize(item) for item in retrieved],
                [normalizer.normalize(item) for item in gold] if gold else None, k,
            ))
            chunks.append(ranking_metrics(raw.get("retrieved_chunk_ids", []),
                                          row.labels.get("relevant_chunk_ids"), k))
        if "retrieval_traces" in raw:
            traces = raw["retrieval_traces"]
            funnels.append({key: sum(trace[key] for trace in traces) for key in (
                "raw_search_results", "unique_urls", "documents", "chunks", "embedding_candidates"
            )})
    return {
        "k": k,
        "url_labelled_count": sum(item["recall"] is not None for item in urls),
        "chunk_labelled_count": sum(item["recall"] is not None for item in chunks),
        **{f"{level}_{name}_at_{k}": mean(item[name] for item in values)
           for level, values in (("retrieval", urls), ("chunk", chunks))
           for name in ("recall", "precision", "mrr")},
        **{f"avg_{key}": mean(item[key] for item in funnels) for key in (
            "raw_search_results", "unique_urls", "documents", "chunks", "embedding_candidates"
        )},
        "avg_final_evidence": mean(len(row.raw["final_evidence"]) for row in rows
                                   if "final_evidence" in row.raw),
    }
