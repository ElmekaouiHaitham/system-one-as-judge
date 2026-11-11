import json

import pytest

from judge_exp import workflow


def test_full_freezes_dev_policy_before_any_test_collection(tmp_path, monkeypatch):
    events = []

    def collect(run, samples, config, policy, splits, providers, **kwargs):
        events.append(splits[0])
        assert kwargs["requests_per_minute"] == 2
        if splits == ["test"]:
            assert (run / "routing_policy_before_test.json").exists()
        return True

    monkeypatch.setattr(workflow, "collect", collect)
    monkeypatch.setattr(workflow, "freeze_policies", lambda *args: {"selection_split": "dev", "policies": {}})
    monkeypatch.setattr(workflow, "analyze", lambda *args: events.append("analyze"))
    workflow.full_experiment(tmp_path, {}, {}, "policy")
    assert events == ["dev", "test", "analyze"]
    assert json.loads((tmp_path / "workflow_status.json").read_text())["status"] == "complete"


def test_access_error_does_not_trigger_repeated_calls(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "collect", lambda *a, **kw: calls.append(1) or False)
    monkeypatch.setattr(workflow, "load_predictions", lambda *a: {
        "a": {"status": "error", "attempts": [{"http_status": 403}]}})
    with pytest.raises(RuntimeError):
        workflow.full_experiment(tmp_path, {}, {}, "policy")
    assert calls == [1]


def test_transient_restart_budget_is_bounded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "collect", lambda *a, **kw: calls.append(1) or False)
    monkeypatch.setattr(workflow, "load_predictions", lambda *a: {
        "a": {"status": "error", "attempts": [{"http_status": 429}]}})
    with pytest.raises(RuntimeError):
        workflow.full_experiment(tmp_path, {}, {}, "policy", max_restarts=2)
    assert len(calls) == 3
