from src.eval.metrics.base import mean, ratio


def query_metrics(queries: list[str], max_queries: int) -> dict:
    normalized = [" ".join(query.casefold().split()) for query in queries]
    return {
        "query_count_valid_rate": float(1 <= len(queries) <= max_queries),
        "empty_query_rate": ratio(sum(not query.strip() for query in queries), len(queries)),
        "exact_duplicate_query_rate": ratio(len(queries) - len(set(queries)), len(queries)),
        "normalized_duplicate_query_rate": ratio(len(queries) - len(set(normalized)), len(queries)),
    }


def planner_metrics(rows) -> dict:
    attempts = [row for row in rows if row.task == "planner"]
    outputs = [row for row in attempts if row.status == "succeeded" and "queries" in row.raw]
    values = [query_metrics(row.raw["queries"], row.raw["max_queries"]) for row in outputs]
    return {
        "planner_case_count": len(attempts), "planner_output_count": len(outputs),
        "planner_success_rate": ratio(len(outputs), len(attempts)),
        **{key: mean(item[key] for item in values)
           for key in query_metrics([], 3)},
    }
