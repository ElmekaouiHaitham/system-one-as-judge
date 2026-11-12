import json

import pytest

from judge_exp.accounting import measurement, usage_cost
from judge_exp.common import ROOT, write_jsonl
from judge_exp.providers import Judges, ProviderError, parse_bedrock, reported_latency


def config(name="glm-5"):
    return json.loads((ROOT / "configs" / f"bedrock-{name}.json").read_text())


def body(label="positive"):
    return {"output": {"message": {"content": [
        {"reasoningContent": {"reasoningText": {"text": "not a verdict"}}},
        {"text": json.dumps({"label": label})}]}}, "stopReason": "end_turn",
        "usage": {"inputTokens": 100, "outputTokens": 30}, "metrics": {"latencyMs": 1234}}


class Response:
    def __init__(self, status, data):
        self.status_code, self.data, self.ok, self.headers = status, data, status < 400, {}

    def json(self):
        return self.data


def test_bedrock_verdict_ignores_reasoning_and_retains_failures():
    assert parse_bedrock(body())[0] == 1
    assert parse_bedrock(body("negative"))[0] == 0
    for reason in ("max_tokens", "guardrail_intervened", "content_filtered", "tool_use"):
        with pytest.raises(ValueError):
            parse_bedrock({**body(), "stopReason": reason})
    with pytest.raises(ValueError):
        parse_bedrock(body("unknown"))
    assert reported_latency(body()) == 1.234
    for value in (None, True, -1, float("nan"), "10"):
        assert reported_latency({"metrics": {"latencyMs": value}}) is None


@pytest.mark.parametrize("name,model", [("glm-5", "zai.glm-5"), ("kimi-k3", "global.moonshotai.kimi-k3")])
def test_request_contract_and_separate_client_and_provider_latency(monkeypatch, name, model):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "offline-token")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    judges = Judges(config(name), "same category policy")
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response(200, body())

    monkeypatch.setattr(judges.session, "post", post)
    row = {"prompt": "context", "response": "answer", "label": 1}
    r = judges.judge("gemini", row)
    assert captured["url"] == f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model}/converse"
    assert captured["headers"] == {"Authorization": "Bearer offline-token"}
    payload = captured["json"]
    assert payload["system"] == [{"text": "same category policy"}]
    assert json.loads(payload["messages"][0]["content"][0]["text"]) == {"prompt": "context", "response": "answer"}
    schema = json.loads(payload["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"])
    assert schema["properties"]["label"]["enum"] == ["positive", "negative"]
    assert "additionalModelRequestFields" not in payload  # K3 effort isn't validated on Converse.
    assert r["label"] == 1 and r["model"] == model
    assert r["provider_latency_s"] == r["provider_attempt_latency_sum_s"] == 1.234
    assert r["latency_s"] < 1.234  # Measured client time, not overwritten by fake provider time.
    assert r["list_cost_usd"] == pytest.approx((100 * config(name)["prices_usd_per_million"]["bedrock_input"] +
                                               30 * config(name)["prices_usd_per_million"]["bedrock_output"]) / 1e6)


def test_bedrock_cache_counts_and_reasoning_not_double_counted():
    rates = config("kimi-k3")["prices_usd_per_million"]
    usage = {"inputTokens": 100, "cacheReadInputTokens": 200, "cacheWriteInputTokens": 50,
             "outputTokens": 30, "reasoningTokens": 20}
    result = usage_cost("bedrock", usage, rates)
    assert result["input_tokens"] == 350 and result["output_tokens"] == 30
    assert result["list_cost_usd"] == pytest.approx((350 * 3 + 30 * 15) / 1e6)
    assert result["cache_adjusted_cost_usd"] == pytest.approx((100 * 3 + 200 * .3 + 50 * 3.75 + 30 * 15) / 1e6)
    assert usage_cost("bedrock", {}, rates)["list_cost_usd"] is None
    assert usage_cost("bedrock", {"inputTokens": 10}, rates)["list_cost_usd"] is None


def test_bedrock_retry_keeps_client_time_and_unknown_service_time(monkeypatch):
    import judge_exp.providers as providers
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "offline-token")
    clock = [0.0]
    monkeypatch.setattr(providers.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(providers.time, "sleep", lambda n: clock.__setitem__(0, clock[0] + n))
    judges = Judges(config(), "policy")
    responses = iter([(.2, Response(429, {"message": "Throttled"})), (1.5, Response(200, body()))])

    def post(*a, **kw):
        duration, response = next(responses)
        clock[0] += duration
        return response

    monkeypatch.setattr(judges.session, "post", post)
    r = judges.judge("fallback", {"prompt": "a", "response": "b"})
    assert r["latency_s"] == pytest.approx(3.7)
    assert r["request_latency_s"] == pytest.approx(1.7)
    assert r["retry_wait_s"] == 2 and r["retry_count"] == 1
    assert r["provider_latency_s"] == 1.234
    assert r["provider_attempt_latency_sum_s"] is None
    assert r["list_cost_usd"] is None and r["known_list_cost_usd"] > 0


def test_bedrock_block_retains_usage_and_raw_without_changing_verdict(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "offline-token")
    judges = Judges(config(), "policy")
    raw = {**body(), "stopReason": "content_filtered"}
    monkeypatch.setattr(judges.session, "post", lambda *a, **kw: Response(200, raw))
    with pytest.raises(ProviderError) as error:
        judges.judge("gemini", {"prompt": "a", "response": "b"})
    assert error.value.record["raw"] == raw
    assert error.value.record["terminal_error"] is True
    assert error.value.record["list_cost_usd"] > 0


def test_reuse_checks_inputs_and_keeps_provenance(tmp_path, monkeypatch):
    import judge_exp.experiment as experiment
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    source, target = tmp_path / "source", tmp_path / "target"
    c = config()
    for split in ("dev", "test"):
        write_jsonl(tmp_path / "data" / "prepared" / f"{split}.jsonl", [{"id": split}])
    experiment.bind_run(source, c, "policy")
    experiment.bind_run(target, c, "policy")
    for split in ("dev", "test"):
        write_jsonl(source / f"{split}_clef.jsonl", [{"id": split, "status": "ok", "label": 1}])
    experiment.reuse_clef(target, source, c, "policy")
    assert (target / "reused_clef_provenance.json").exists()
    assert (target / "test_clef.jsonl").read_bytes() == (source / "test_clef.jsonl").read_bytes()
    with pytest.raises(ValueError, match="policy"):
        experiment.reuse_clef(target, source, c, "changed policy")
    with pytest.raises(ValueError, match="timeout_seconds"):
        experiment.reuse_clef(target, source, {**c, "timeout_seconds": 1}, "policy")


def test_service_time_unknown_after_resumed_timeout(tmp_path):
    from judge_exp.experiment import load_predictions
    write_jsonl(tmp_path / "dev_gemini.jsonl", [
        {"id": "a", "status": "error", "provider_attempt_latency_sum_s": None},
        {"id": "a", "status": "ok", "provider_latency_s": 1,
         "provider_attempt_latency_sum_s": 1},
    ])
    r = load_predictions(tmp_path, "dev", "gemini")["a"]
    assert r["provider_latency_s"] == 1
    assert r["provider_attempt_latency_sum_s"] is None


def test_report_names_model_and_separates_reused_collection_cost(tmp_path):
    from judge_exp.common import write_json
    from judge_exp.report import analyze
    samples = {}
    for split in ("dev", "test"):
        rows, clef, fallback = [], [], []
        for i in range(8):
            label = i % 2
            rows.append({"id": f"{split}-{i}", "label": label, "source_row": i,
                         "control_type": "positive" if label else "safe_control"})
            common = {"id": f"{split}-{i}", "status": "ok", "label": label,
                      "latency_s": 2, "request_latency_s": 2, "retry_count": 0, "list_cost_usd": .01}
            clef.append({**common, "probability": .9 if label else .1, "confidence": .9})
            fallback.append({**common, "provider_latency_s": 1.5, "provider_attempt_latency_sum_s": 1.5})
        samples[split] = rows
        write_jsonl(tmp_path / f"{split}_clef.jsonl", clef)
        write_jsonl(tmp_path / f"{split}_gemini.jsonl", fallback)
    write_json(tmp_path / "reused_clef_provenance.json", {"source_run": "offline fixture"})
    analyze(tmp_path, samples, config())
    report = tmp_path / "report"
    metrics = json.loads((report / "metrics.json").read_text())
    assert metrics[1]["display_name"] == "GLM-5"
    assert metrics[0]["list_cost_usd"] == pytest.approx(.08)
    ledger = json.loads((report / "collection_accounting.json").read_text())
    assert ledger["known_list_cost_subtotal_usd"] == pytest.approx(.16)
    assert ledger["records"][0]["collection_calls"] == 0
    assert ledger["records"][0]["reused_source_records"] == 8
    import pandas as pd
    times = pd.read_csv(report / "latency_diagnostics.csv")
    assert set(times.judge) == {"Clef", "GLM-5"}
    assert times.loc[times.judge == "Clef", "provider_latency_final_call_s"].isna().all()
    assert (times.loc[times.judge == "GLM-5", "provider_latency_final_call_s"] == 1.5).all()


def test_retry_free_comparison_uses_same_examples_for_both_judges():
    from judge_exp.report import paired_first_attempt_latency
    def record(seconds, retry=0):
        return {"status": "ok", "latency_s": seconds, "retry_count": retry,
                "attempts": [{"http_status": 200}], "provider_latency_s": seconds - .1}
    clef = [record(1), record(2), record(3)]
    fallback = [record(20, retry=1), record(5), {"status": "error"}]
    r = paired_first_attempt_latency(clef, fallback, "Kimi K3")
    assert r["paired_first_attempt_n"] == 1 and r["excluded_pairs"] == 2
    assert r["systems"][0]["median_client_s"] == 2
    assert r["systems"][1]["median_client_s"] == 5
    assert r["systems"][1]["median_provider_reported_s"] == 4.9
