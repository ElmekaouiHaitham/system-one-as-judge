from judge_exp.common import write_jsonl
from judge_exp.comparison import compare_models


def test_standalone_comparison_includes_both_router_groups(tmp_path):
    samples = {}
    for split in ("dev", "test"):
        rows = [{"id": f"{split}-{i}", "label": i} for i in (0, 1)]
        samples[split] = rows
        for provider in ("clef", "gemini"):
            records = []
            for row in rows:
                correct = split == "test" if provider == "clef" else (split == "dev" or row["label"] == 0)
                records.append({"id": row["id"], "status": "ok",
                    "label": row["label"] if correct else 1-row["label"],
                    "latency_s": 1, "retry_count": 0, "list_cost_usd": .001})
            write_jsonl(tmp_path / f"{split}_{provider}.jsonl", records)
    result = compare_models(tmp_path, samples, {"seed": 42, "fallback_display_name": "Kimi K3"})
    assert [r["n"] for r in result] == [4, 4]
    assert [r["correct"] for r in result] == [2, 3]
    assert [r["accuracy"] for r in result] == [.5, .75]
