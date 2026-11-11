import json

import pytest

from judge_exp.accounting import measurement, summarize_resources, usage_cost
from judge_exp.common import ROOT
from judge_exp.providers import Judges, ProviderError

RATES = {"clef_input": .24, "gemini_input": .25, "gemini_output": 1.5, "gemini_cached_input": .025}


def test_token_costs_include_thinking_and_cache_without_double_counting():
    usage = {"promptTokenCount": 1000, "cachedContentTokenCount": 200,
             "candidatesTokenCount": 10, "thoughtsTokenCount": 20}
    r = usage_cost("gemini", usage, RATES)
    assert r["output_tokens"] == 30
    assert r["list_cost_usd"] == pytest.approx(.000295)
    assert r["cache_adjusted_cost_usd"] == pytest.approx(.000250)
    assert usage_cost("clef", {"input_tokens": 1000}, RATES)["list_cost_usd"] == pytest.approx(.00024)


def test_missing_usage_is_unknown_and_self_hosting_is_unpriced():
    assert usage_cost("gemini", {}, RATES)["list_cost_usd"] is None
    assert usage_cost("gemini", {"promptTokenCount": 100}, RATES)["list_cost_usd"] is None
    assert usage_cost("gemini", {"promptTokenCount": 100, "totalTokenCount": 120}, RATES)["output_tokens"] == 20
    assert usage_cost("clef", {"input_tokens": 1000}, RATES, True)["list_cost_usd"] is None
    assert usage_cost("gemini", {"promptTokenCount": 10, "candidatesTokenCount": 2,
                                "cachedContentTokenCount": 20}, RATES)["list_cost_usd"] is None


def test_cascade_counts_only_required_calls_and_reports_latency_percentiles():
    c = [{"list_cost_usd": .01, "latency_s": 1, "retry_count": 0}, {"list_cost_usd": .02, "latency_s": 2, "retry_count": 0}]
    g = [{}, {"list_cost_usd": .04, "latency_s": 4, "retry_count": 1}]
    r = summarize_resources(c, g, [False, True])
    assert r["list_cost_usd"] == pytest.approx(.07)
    assert r["cost_per_1000_pairs_usd"] == pytest.approx(35)
    assert r["mean_latency_s"] == r["median_latency_s"] == 3.5
    assert r["p95_latency_s"] == pytest.approx(5.75)
    assert r["retried_pair_rate"] == .5
    assert summarize_resources(c, g, [True, True])["list_cost_usd"] is None
    assert summarize_resources(c, g, [True, True])["known_list_cost_subtotal_usd"] == pytest.approx(.07)


class Response:
    def __init__(self, status, body):
        self.status_code, self.body, self.headers = status, body, {}
        self.ok = status < 400

    def json(self):
        return self.body


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def test_latency_includes_retries_and_backoff_and_cost_counts_all_attempts(monkeypatch):
    import judge_exp.providers as providers
    clock = [0.0]
    monkeypatch.setattr(providers.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    judges = Judges(config(), "policy")
    responses = iter([
        (.4, Response(500, {"usageMetadata": {"promptTokenCount": 50, "totalTokenCount": 50}})),
        (.6, Response(200, {"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": 5}})),
    ])

    def post(*args, **kwargs):
        duration, response = next(responses)
        clock[0] += duration
        return response

    monkeypatch.setattr(judges.session, "post", post)
    _, record = judges.post("https://example.test", {}, {}, "gemini")
    assert record["latency_s"] == pytest.approx(3)
    assert record["request_latency_s"] == pytest.approx(1)
    assert record["retry_wait_s"] == pytest.approx(2)
    assert record["retry_count"] == 1
    assert record["list_cost_usd"] == pytest.approx((150*.25 + 15*1.5)/1e6)


def test_unpriced_retry_does_not_silently_become_free():
    body = {"usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10}}
    m = measurement("gemini", body, {"attempts": [
        {"usage": {}}, {"usage": body["usageMetadata"]}]}, RATES)
    assert m["list_cost_usd"] is None
    assert m["known_list_cost_usd"] > 0


def test_blocked_verdict_keeps_timing_usage_and_raw_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    judges = Judges(config(), "policy")
    body = {"promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {"promptTokenCount": 100, "totalTokenCount": 100}}
    monkeypatch.setattr(judges.session, "post", lambda *a, **kw: Response(200, body))
    with pytest.raises(ProviderError) as exc:
        judges.judge("gemini", {"prompt": "test", "response": "test"})
    assert exc.value.record["latency_s"] >= 0
    assert exc.value.record["raw"] == body
    assert exc.value.record["terminal_error"] is True
    assert exc.value.record["list_cost_usd"] == pytest.approx(.000025)


def test_timeout_retains_elapsed_time_with_unknown_cost(monkeypatch):
    import requests
    judges = Judges(config(), "policy")

    def timeout(*args, **kwargs):
        raise requests.Timeout("private URL must not appear in logs")

    monkeypatch.setattr(judges.session, "post", timeout)
    with pytest.raises(ProviderError) as exc:
        judges.post("https://example.test", {}, {}, "gemini")
    assert str(exc.value) == "Timeout"
    assert exc.value.record["latency_s"] >= 0
    assert exc.value.record["list_cost_usd"] is None


def test_live_wall_time_and_request_path_time_are_distinct(tmp_path, monkeypatch):
    import sys
    import judge_exp.cli as cli
    from judge_exp.common import write_json, write_jsonl, read_jsonl
    clock = [0.0]
    monkeypatch.setattr(cli.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(cli, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(cli, "bind_run", lambda *args: (tmp_path, {}))

    class FakeJudges:
        def __init__(self, *args):
            pass

        def judge(self, provider, row):
            clock[0] += 1 if provider == "clef" else 4  # Gemini has 2s pacing + 2s request
            return {"label": 1, "confidence": .6, "latency_s": 1 if provider == "clef" else 2,
                    "list_cost_usd": .1 if provider == "clef" else .2}

    monkeypatch.setattr(cli, "Judges", FakeJudges)
    write_json(tmp_path / "routing_policy.json", {"policies": {"fixed": {"negative_threshold": .9, "positive_threshold": .9}}})
    inputs, output = tmp_path / "inputs.jsonl", tmp_path / "out.jsonl"
    write_jsonl(inputs, [{"prompt": "test", "response": "test"}])
    monkeypatch.setattr(sys, "argv", ["judge-experiment", "cascade", "--input", str(inputs), "--output", str(output)])
    cli.main()
    result = read_jsonl(output)[0]
    assert result["wall_latency_s"] == 5
    assert result["request_path_latency_s"] == 3
    assert result["list_cost_usd"] == pytest.approx(.3)


def test_gemini_sampling_controls_can_be_omitted_without_changing_baseline(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    for value in (None, 0, .7):
        c = config()
        c["gemini_temperature"] = value
        judges = Judges(c, "policy")
        captured = {}

        def post(url, headers, payload, provider):
            captured.update(payload)
            return {"candidates": [{"finishReason": "STOP", "content": {
                "parts": [{"text": '{"label":"negative"}'}]}}]}, {"latency_s": 1}

        monkeypatch.setattr(judges, "post", post)
        judges.judge("gemini", {"prompt": "test", "response": "test"})
        generation = captured["generationConfig"]
        if value is None:
            assert "temperature" not in generation
        else:
            assert generation["temperature"] == value
