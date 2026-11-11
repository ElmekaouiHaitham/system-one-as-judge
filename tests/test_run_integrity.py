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
