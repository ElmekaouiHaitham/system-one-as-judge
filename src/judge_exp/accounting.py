"""Token-based price estimates; no claim to reconstruct provider invoices."""
import math


def count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def usage_cost(provider, usage, rates, self_hosted=False):
    usage = usage if isinstance(usage, dict) else {}
    if provider == "clef":
        inputs = count(usage.get("input_tokens", usage.get("prompt_tokens")))
        outputs, cached = 0, 0
    else:
        inputs = count(usage.get("promptTokenCount"))
        cached = count(usage.get("cachedContentTokenCount", 0))
        candidate = count(usage.get("candidatesTokenCount"))
        thoughts = count(usage.get("thoughtsTokenCount", 0))
        tools = count(usage.get("toolUsePromptTokenCount", 0))
        total = count(usage.get("totalTokenCount"))
        outputs = candidate + thoughts if candidate is not None and thoughts is not None else None
        # A missing output count is not automatically zero. Total usage can recover it.
        if outputs is None and total is not None and inputs is not None and tools == 0:
            outputs = total - inputs if total >= inputs else None
        if tools != 0:
            outputs = None  # This text-only experiment does not price tool calls.
    valid = inputs is not None and outputs is not None and cached is not None and cached <= inputs
    full = adjusted = None
    if valid and not self_hosted:
        input_rate = rates.get(f"{provider}_input")
        output_rate = rates.get(f"{provider}_output", 0 if provider == "clef" else None)
        if all(isinstance(r, (int, float)) and math.isfinite(r) and r >= 0 for r in [input_rate, output_rate]):
            full = (inputs * input_rate + outputs * output_rate) / 1e6
            cache_rate = rates.get(f"{provider}_cached_input", input_rate if not cached else None)
            if isinstance(cache_rate, (int, float)) and math.isfinite(cache_rate) and cache_rate >= 0:
                adjusted = ((inputs - cached) * input_rate + cached * cache_rate + outputs * output_rate) / 1e6
    return {"input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached,
            "list_cost_usd": full, "cache_adjusted_cost_usd": adjusted}


def measurement(provider, body, telemetry, rates, self_hosted=False):
    attempts = []
    for attempt in telemetry["attempts"]:
        attempts.append({**attempt, **usage_cost(provider, attempt["usage"], rates, self_hosted)})
    result = body.get("result", body)
    result = result if isinstance(result, dict) else {}
    usage = result.get("usage", {}) if provider == "clef" else body.get("usageMetadata", {})
    final = usage_cost(provider, usage, rates, self_hosted)
    for key in ("list_cost_usd", "cache_adjusted_cost_usd"):
        costs = [a[key] for a in attempts]
        final[key] = sum(costs) if costs and all(c is not None for c in costs) else None
        final[f"known_{key}"] = sum(c for c in costs if c is not None)
    return {**telemetry, **final, "attempts": attempts, "usage": usage, "raw": body,
            "pricing_basis": "self_hosted_unpriced" if self_hosted else "standard_text_token_list_rates"}


def summarize_resources(clef, gemini, escalation=None, gemini_only=False):
    """Sum the stages actually required per item; unknown unused stages are ignored."""
    import numpy as np
    n = len(clef)
    escalation = np.zeros(n, dtype=bool) if escalation is None else np.asarray(escalation)
    costs, latencies, retried, known_costs = [], [], [], []
    for i in range(n):
        stages = [gemini[i]] if gemini_only else [clef[i]] + ([gemini[i]] if escalation[i] else [])
        retries = [s.get("retry_count") for s in stages]
        retried.append(any(r > 0 for r in retries) if all(r is not None for r in retries) else None)
        known_costs.append(sum(s.get("known_list_cost_usd", s.get("list_cost_usd")) or 0 for s in stages))
        for key, values in [("list_cost_usd", costs), ("latency_s", latencies)]:
            parts = [s.get(key) for s in stages]
            values.append(sum(parts) if all(v is not None and math.isfinite(v) for v in parts) else None)
    complete_cost = all(c is not None for c in costs)
    complete_time = all(t is not None for t in latencies)
    return {
        "list_cost_usd": sum(costs) if complete_cost else None,
        "known_list_cost_subtotal_usd": sum(known_costs),
        "cost_per_1000_pairs_usd": sum(costs) / n * 1000 if complete_cost else None,
        "resource_priced_n": sum(c is not None for c in costs),
        "retried_pair_rate": sum(retried) / n if all(r is not None for r in retried) else None,
        "mean_latency_s": float(np.mean(latencies)) if complete_time else None,
        "median_latency_s": float(np.median(latencies)) if complete_time else None,
        "p95_latency_s": float(np.quantile(latencies, .95)) if complete_time else None,
    }
