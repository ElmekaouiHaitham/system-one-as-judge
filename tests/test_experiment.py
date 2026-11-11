import numpy as np
import pytest

from judge_exp.common import CATEGORY
from judge_exp.data import prompt_key, sample
from judge_exp.providers import parse_clef, parse_gemini, probability
from judge_exp.routing import route, select_thresholds
from judge_exp.report import metrics, score_metrics


def test_probability_validation():
    for invalid in [True, -0.1, 1.1, float("nan"), "0.9"]:
        with pytest.raises(ValueError):
            probability(invalid)


def test_provider_contracts_and_blocks():
    assert parse_clef({"success": True, "result": {"answers": {"category_positive": .82},
                       "usage": {"input_tokens": 400}}})[0] == .82
    assert parse_clef({"answers": {"category_positive": .2}})[0] == .2
    assert parse_gemini({"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": '{"label":"positive"}'}]}}]})[0] == 1
    for body in [{}, {"candidates": [{"finishReason": "SAFETY"}]},
                 {"candidates": [{"finishReason": "MAX_TOKENS"}]}]:
        with pytest.raises(ValueError):
            parse_gemini(body)


def test_routing_boundary_and_failure():
    assert route([0, 1, -1], [.8, .8, 0], .8, .9).tolist() == [False, True, True]
    assert route([0, 1], [1, 1], 1.000001, 1.000001).all()


def test_threshold_selection_and_adaptive_benefit():
    y = np.array([0, 0, 1, 1])
    c = np.array([0, 0, 1, 1])
    # One confidently wrong positive verdict; negative predictions are correct.
    y[3] = 0
    p = [.6, .7, .6, .9]
    fixed = select_thresholds(y, c, y, p, [.5, .8, 1.000001], 0)
    adaptive = select_thresholds(y, c, y, p, [.5, .8, 1.000001], 0, True)
    assert fixed["dev_accuracy"] == 1
    assert adaptive["dev_accuracy"] == 1
    assert adaptive["dev_escalation_rate"] < fixed["dev_escalation_rate"]


def test_sampling_no_prompt_overlap_and_correct_controls():
    rows = []
    for i in range(40):
        label, safe = i < 20, 20 <= i < 30
        rows.append({"prompt": f"Prompt {i}", "response": f"Response {i}",
                     "category": {CATEGORY: label}, "is_safe": safe})
    selected = sample(rows + rows, 12, 42, "test", {prompt_key("Prompt 0")})
    assert len(selected) == len({prompt_key(r["prompt"]) for r in selected}) == 12
    assert "Prompt 0" not in [r["prompt"] for r in selected]
    assert sum(r["label"] for r in selected) == 6
    assert sum(r["control_type"] == "safe_control" for r in selected) == 6
    assert all(r["label"] == 1 or r["is_safe"] for r in selected)
    assert selected == sample(rows + rows, 12, 42, "test", {prompt_key("Prompt 0")})


def test_excluded_response_does_not_block_safe_response_for_same_prompt():
    rows = [
        {"prompt": "shared", "response": "excluded", "category": {CATEGORY: False}, "is_safe": False},
        {"prompt": "shared", "response": "safe", "category": {CATEGORY: False}, "is_safe": True},
        {"prompt": "positive", "response": "positive", "category": {CATEGORY: True}, "is_safe": False},
    ]
    for seed in range(10):
        selected = sample(rows, 2, seed, "test")
        assert {r["response"] for r in selected} == {"safe", "positive"}


def test_failures_are_not_silently_dropped():
    m = metrics(np.array([0, 1, 0, 1]), np.array([0, 1, -1, -1]))
    assert m["accuracy"] == .5
    assert m["failure_rate"] == .5
    assert m["recall"] == .5


def test_auroc_ranking_ties_and_failures():
    def result(p):
        return {"status": "ok", "probability": p}
    m, _ = score_metrics([0, 1, 0, 1], [result(.1), result(.9), result(.2), result(.8)])
    assert m["classification_auroc"] == 1
    assert m["error_detection_auroc"] is None  # no errors to rank
    m, _ = score_metrics([0, 1], [result(.5), result(.5)])
    assert m["classification_auroc"] == .5
    m, _ = score_metrics([0, 0, 1, 1], [result(.1), result(.6), result(.4), result(.9)])
    assert m["error_detection_auroc"] == 1  # errors have higher uncertainty
    m, _ = score_metrics([0, 1, 1], [result(.1), {"status": "error"}, result(float("nan"))])
    assert m["classification_auroc"] is None  # only one valid target class
    assert m["auroc_scored_n"] == 1
    assert m["auroc_excluded_n"] == 2
