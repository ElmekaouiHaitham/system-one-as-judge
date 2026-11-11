import json

import pytest

from judge_exp.common import write_jsonl
from judge_exp.experiment import bind_run, collect, load_predictions


def test_manifest_rejects_changed_rubric_or_data(tmp_path, monkeypatch):
    import judge_exp.experiment as experiment
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    for split in ("dev", "test"):
        write_jsonl(tmp_path / "data" / "prepared" / f"{split}.jsonl", [{"id": split}])
    run = tmp_path / "run"
    bind_run(run, {"seed": 42}, "rubric")
    bind_run(run, {"seed": 42}, "rubric")
    with pytest.raises(ValueError, match="changed"):
        bind_run(run, {"seed": 42}, "new rubric")
    write_jsonl(tmp_path / "data" / "prepared" / "test.jsonl", [{"id": "different"}])
    with pytest.raises(ValueError, match="changed"):
        bind_run(run, {"seed": 42}, "rubric")


def test_collection_resumes_successes_and_retries_errors(tmp_path, monkeypatch):
    import judge_exp.experiment as experiment
    calls = []

    class FakeJudges:
        def __init__(self, *args):
            pass

        def judge(self, provider, row):
            calls.append(row["id"])
            return {"label": 1}

    monkeypatch.setattr(experiment, "Judges", FakeJudges)
    write_jsonl(tmp_path / "dev_clef.jsonl", [
        {"id": "a", "status": "ok", "label": 0},
        {"id": "b", "status": "error"},
    ])
    assert collect(tmp_path, {"dev": [{"id": "a"}, {"id": "b"}]}, {}, "rubric", ["dev"], ["clef"])
    assert calls == ["b"]
    assert load_predictions(tmp_path, "dev", "clef")["b"]["status"] == "ok"


def test_collection_saves_accounting_on_failure(tmp_path, monkeypatch):
    import judge_exp.experiment as experiment
    from judge_exp.providers import ProviderError

    class FailedJudge:
        def __init__(self, *args):
            pass

        def judge(self, *args):
            raise ProviderError("blocked", {"latency_s": 2, "list_cost_usd": .001, "usage": {"input_tokens": 100}})

    monkeypatch.setattr(experiment, "Judges", FailedJudge)
    assert not collect(tmp_path, {"dev": [{"id": "a"}]}, {}, "policy", ["dev"], ["clef"])
    r = load_predictions(tmp_path, "dev", "clef")["a"]
    assert r["status"] == "error" and r["latency_s"] == 2 and r["list_cost_usd"] == .001


def test_content_blocks_are_retained_without_retries_or_stopping_collection(tmp_path, monkeypatch):
    import judge_exp.experiment as experiment
    from judge_exp.providers import ProviderError
    calls = []

    class FakeJudges:
        def __init__(self, *args):
            pass

        def judge(self, provider, row):
            calls.append(row["id"])
            if row["id"] == "b":
                raise ProviderError("blocked", {"list_cost_usd": .001}, terminal=True)
            return {"label": 0}

    monkeypatch.setattr(experiment, "Judges", FakeJudges)
    write_jsonl(tmp_path / "test_gemini.jsonl", [{"id": "a", "status": "error",
        "raw": {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}}])
    samples = {"test": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
    assert collect(tmp_path, samples, {}, "policy", ["test"], ["gemini"])
    assert calls == ["b", "c"]
    assert collect(tmp_path, samples, {}, "policy", ["test"], ["gemini"])
    assert calls == ["b", "c"]  # no repeat for a different verdict, including blocked calls
    results = load_predictions(tmp_path, "test", "gemini")
    assert results["a"]["status"] == results["b"]["status"] == "error"
    assert results["c"]["status"] == "ok"


def test_resumed_transport_attempts_keep_their_time_cost_and_unknown_usage(tmp_path):
    write_jsonl(tmp_path / "test_gemini.jsonl", [
        {"id": "a", "status": "error", "latency_s": 90, "request_latency_s": 90,
         "retry_wait_s": 0, "retry_count": 0, "list_cost_usd": None,
         "known_list_cost_usd": 0, "attempts": [{"http_status": None}]},
        {"id": "a", "status": "ok", "label": 1, "latency_s": 1.2, "request_latency_s": 1.2,
         "retry_wait_s": 0, "retry_count": 0, "list_cost_usd": .001,
         "known_list_cost_usd": .001, "attempts": [{"http_status": 200}]},
    ])
    result = load_predictions(tmp_path, "test", "gemini")["a"]
    assert result["status"] == "ok"
    assert result["latency_s"] == pytest.approx(91.2)
    assert result["list_cost_usd"] is None
    assert result["known_list_cost_usd"] == .001
    assert result["retry_count"] == result["collector_restarts"] == 1
    assert len(result["attempts"]) == 2
    assert len((tmp_path / "test_gemini.jsonl").read_text().splitlines()) == 2
