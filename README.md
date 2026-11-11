# System One as Judge: Clef → Gemini

[![Python checks](https://github.com/ElmekaouiHaitham/system-one-as-judge/actions/workflows/checks.yml/badge.svg)](https://github.com/ElmekaouiHaitham/system-one-as-judge/actions/workflows/checks.yml)

Compare Cloudflare Clef (27B), Gemini 3.1 Flash-Lite, and confidence cascades on
BeaverTails' `discrimination,stereotype,injustice` category. Existing human labels
are the reference. No new annotation is required. This is an illustrative
reproducible experiment, with conclusions limited to its sample and label policy.

**Question:** How much evaluation cost can confidence routing save, and which
errors remain when uncertain Clef verdicts are escalated to Gemini?

| System | Decision |
|---|---|
| Clef alone | Binary decision from Clef's category probability |
| Gemini alone | Independent verdict using the same category policy |
| Fixed cascade | Escalate below one development-selected confidence threshold |
| Class-dependent cascade | Use separate thresholds for Clef-positive and Clef-negative predictions |

**Status:** experiment implementation, prepared sampling procedure, and offline
checks are available. Live model results are not included. Any synthetic demo
outputs are only for checking the plotting pipeline.

## Quick start

You need Python 3.11+, Git, a Cloudflare account with Workers AI access, and a
Gemini Developer API key. No local GPU is required. The standard experiment
attempts **800 model calls**: two judges on 100 development and 300 test pairs.
The three-example smoke test is cached and reused by the full run.

Clone the repository:

```sh
git clone https://github.com/ElmekaouiHaitham/system-one-as-judge.git
cd system-one-as-judge
python -m venv .venv
```

Activate the environment and create your private configuration:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

**macOS / Linux**

```sh
source .venv/bin/activate
cp .env.example .env
```

Install the project (all following `python` commands use the activated environment):

```sh
python -m pip install -e ".[dev]"
```

On Windows, if activation is unavailable, use `.\.venv\Scripts\python.exe` in
place of `python` in each command; changing your execution policy is unnecessary.

Edit `.env` with your Cloudflare account ID, a Workers AI API token (Workers AI
Read and Edit permissions), and a Google AI Studio Gemini API key. Do not share this file.
The code reads keys from environment variables or `.env`. No credentials are
included in run manifests. Never commit `data/`, `runs/`, or `.env`.

| Variable | Purpose |
|---|---|
| `CLOUDFLARE_ACCOUNT_ID` | Account ID shown in your Cloudflare dashboard |
| `CLOUDFLARE_API_TOKEN` | API token permitted to call Workers AI |
| `GEMINI_API_KEY` | Key from [Google AI Studio](https://aistudio.google.com/api-keys) |
| `CLEF_ENDPOINT` | Optional full self-hosted SystemOne URL; leave blank for Cloudflare |
| `CLEF_API_KEY` | Optional self-hosted server token |

`.env.example` contains blank values and is committed as the public template.
Your `.env` stays local and is ignored by Git.

`requirements-lock.txt` records the versions used for local verification. To use
that snapshot instead of resolving the version ranges, install with
`python -m pip install -r requirements-lock.txt` inside your own virtual environment.

## Recommended hosting and prices

Checked November 11, 2026:

| Model | Hosted option | USD per million tokens, input / output |
|---|---|---|
| Clef, 27B | Cloudflare Workers AI | $0.24 / $0 |
| Gemini 3.1 Flash-Lite | Gemini Developer API | $0.25 / $1.50 |

Use Cloudflare first: no GPU provisioning or idle GPU time. Workers AI advertises
10,000 free Neurons per day; check the dashboard for your model availability and
remaining allowance. Gemini lists a free tier, subject to account/model quotas.
The default 10 requests/minute is configurable and does not guarantee it fits
your account's daily quota. Collection resumes across days.

For scale only: at 500 input tokens per example, 400 examples cost about $0.048
for Clef and $0.05 Gemini input, plus Gemini output/thinking. Actual token counts
include the policy and request schema. These are list-price illustrations, not
a bill or a promise of free access. Prices in `config.json` are editable.

Modal is unnecessary for this sample. If later self-hosting, Clef requires its
typed decision interface, not ordinary chat generation. The official release
documents the custom `joint_schema_model` loader and SGLang's `/v1/systemone`
endpoint. Set `CLEF_ENDPOINT` to that full endpoint, `CLEF_API_KEY` if needed,
and set `clef_request_model` in a new config to the server's model name (typically
`Cloudflare/clef`). Self-hosted GPU costs are NOT estimated by the hosted token
prices; change the accounting before comparing deployments. A Modal deployment
is deliberately not provisioned by this project.

## Run the experiment

```sh
# Download pinned 30k_train and 30k_test; save exact sampled rows and a manifest.
python -m judge_exp.cli prepare

# Check API access with 3 examples per model on development data first.
python -m judge_exp.cli collect --split dev --limit 3

# Resume and collect both judges on all development + test examples.
python -m judge_exp.cli collect

# Tune on development only, then generate test metrics and graphs.
python -m judge_exp.cli analyze
```

Run `prepare` once per checkout. It refuses to overwrite an existing prepared
sample. A fresh clone regenerates the same sample using the pinned dataset
revision and seed; `data/prepared/manifest.json` records the sample hashes.
Collection and analysis can then be repeated as described below.

The first failed request stops collection with a nonzero exit code so an invalid
key or exhausted quota does not burn through a sample. Fix it and rerun the same
command: successful calls are reused, failed calls are retried. Each response is
saved immediately. Avoid running two collectors into the same run directory.
Do not change your config/policy halfway through a run: the manifest rejects
such changes. Use `--run runs/another_name` BEFORE the subcommand for a new run.
Prompt edits, prices, and model settings all form part of the run identity.

Analysis requires all examples to have been attempted, and development verdicts
must all be successful before tuning. Attempted test failures are retained as
failures, counted as incorrect for accuracy and as misses for positive recall.
They are visible in confusion matrices and failure rates. Gemini safety blocks
are failures, never silently converted to category labels. Resolve failures
and rerun analysis when possible. Unknown token usage yields unknown cost.

## Sample and controls

Default: 100 development examples from `30k_train`, 300 test examples from
`30k_test`. Each split has 50% category positives and 50% globally safe negative
controls. We use `category[...]` as the target. Negative controls must also have
`is_safe = true`.

One response per normalized prompt is selected. All test-pool prompts are
excluded from development. Selection is seeded; dataset revision, row indexes,
content IDs, and sample hashes are saved. Sample sizes must be positive even numbers.
These controls do not guarantee a particular number of refusals or subtle
identity-related negatives; there are no new semantic annotations.

This is a balanced diagnostic sample: precision and routing savings do not
represent deployment prevalence. Public data may have appeared in model
training; this cannot establish generalization to fresh red-team outputs.
The rubric in `prompts/policy.txt` is an operational adaptation of the dataset
definition, not a claim to reproduce its complete original annotation process.
The model-facing prompt uses a general safety-evaluation context and does not
name BeaverTails or mention a benchmark. Dataset attribution remains here.
Gemini receives this policy as its system instruction; JSON output is constrained
separately through the API's `responseMimeType` and `responseSchema`, with a
required `label` field restricted to `positive` or `negative`.

## Cascade selection

Clef predicts positive at `p >= 0.5`; confidence is `max(p, 1-p)`. This is a model
score, not guaranteed calibrated confidence. Escalate if confidence is BELOW the
routing threshold. Gemini receives the original pair and same rubric, without
Clef's verdict, confidence, or the human label.

Two routers are fitted on development only:

1. **Fixed:** one confidence threshold for both predicted classes.
2. **Class-dependent:** separate thresholds for Clef-positive and Clef-negative
   predictions, allowing different treatment of overflagging and missed harm.

Both minimize escalation subject to development accuracy being within 2 percentage
points of Gemini alone (`max_dev_accuracy_drop`). Ties favor accuracy, then lower
thresholds. The grid includes a value above 1 to represent always escalating.
This is a development heuristic, not a statistical safety guarantee. 100 examples
make threshold selection noisy. Compare both frozen policies on the held-out test.

The class-dependent router is a simple input-dependent extension, not an online
learning algorithm or an established novelty claim. Avoid adding complex learned
routers until the small baseline experiment is working. Never tune using test
labels or choose the article's preferred threshold from the test sweep.

## Outputs

Under `runs/main/report/`:

- `metrics.csv/json`: accuracy, precision/recall/F1, false-positive and miss rates,
  failure rate, bootstrap accuracy intervals and paired accuracy differences
  versus Gemini, routing rate, estimated list cost, cost per 1,000 pairs, mean,
  median, and 95th-percentile latency, the fraction of pairs requiring a retry,
  and the count of fully priced pairs. The main latency chart uses median and p95.
- `predictions.csv`: IDs and per-system verdicts for tracing disagreements.
- `control_slices.csv`: results for positives and safe controls.
- `cascade_diagnostics.json`: fallback corrections, regressions, shared errors.
- `confusion_matrices.png`, `cost_latency.png`, `coverage_tradeoff.png`: article
  figures. Also a diagnostic threshold-sweep CSV.
- `clef_roc.png`: ROC curves for category classification and error detection.

Clef also reports **classification AUROC** from its positive-category probability,
and **error-detection AUROC** from `1 - confidence` against whether its verdict
was wrong. The second metric tests the premise of confidence routing. Both use
valid successful Clef scores only; scored/excluded counts are reported explicitly.
AUROC is undefined if only one outcome class remains (for example, no errors).
Failures remain counted in the other metrics as described above. Gemini returns
only a label, so Gemini and cascade AUROCs are left unavailable; no comparable
continuous score is invented from their binary decisions. AUROC measures ranking,
not probability calibration. See [the metric documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

`collection_accounting.json` separately sums all development/test collection
records, including failed verdicts and repeated calls after resuming. It reports
a known cost subtotal and unpriced calls; a complete total is unavailable if any
call has unknown usage. It is still an estimate at list prices, not your bill.

`routing_policy.json` records the thresholds chosen on development.
The 2,000-resample bootstrap is over sampled test items, conditional on the
sampling design; it does not account for annotation uncertainty or model drift.

## How latency and cost are measured

**Latency:** a monotonic `time.perf_counter()` timer starts immediately before the
HTTP request wrapper and stops after the final response has been read and decoded.
`latency_s` includes network travel, provider queueing/inference, response transfer,
all retry requests, and retry backoff. It excludes deliberate quota pacing before
the wrapper, input preparation, file writes, dataset downloads, and plotting.
This is client-observed API latency, not pure model inference time.

Each record also retains every HTTP attempt's duration/status/usage,
`request_latency_s` (sum of attempt durations), `retry_wait_s`, and `retry_count`.
Transport errors, HTTP errors, blocked verdicts, and invalid JSON/verdicts preserve
available timing and usage. A timed-out request may have incurred a charge even
when no token counts were received; its cost stays unknown.

**Cost:** use provider-reported token counts, including the input policy/schema,
and configurable rates per million tokens:

```text
Clef cost   = input_tokens × clef_input_rate / 1,000,000
Gemini cost = (prompt_tokens × gemini_input_rate
              + (candidate_tokens + thinking_tokens) × gemini_output_rate)
              / 1,000,000
```

Gemini's `promptTokenCount` already includes cached tokens. The comparison column
`list_cost_usd` deliberately prices all prompt tokens at the standard uncached
rate for a consistent baseline. Saved request records also contain
`cache_adjusted_cost_usd`: subtract cached tokens from standard input and price
them once at `gemini_cached_input`. This uses observed implicit cache hits; cache
hits could differ when actually deploying selective routing. Missing usage stays
unknown instead of being silently priced as zero. Returned thinking tokens count
as output. Token usage and raw responses remain available for auditing. See
[Gemini usage metadata](https://ai.google.dev/api/generate-content#UsageMetadata).

Every retry's reported usage contributes to the logical call's cost. If any
attempt lacks usage, only the known subtotal is available and the complete cost
is unknown. A self-hosted Clef endpoint is explicitly unpriced: Cloudflare's token
rates do not estimate Modal's GPU-time bill.

**Cascade simulation:** for each test pair, add Clef's measured latency/cost plus
Gemini's ONLY if that pair is escalated. Sum per-pair costs; report mean, median,
and p95 of per-pair latency sums. Never add the two models' p95 values. Unknown
usage on a Gemini call that would not be used does not affect the cascade total.

Both judges are called on all development and test items for the benchmark.
Its collection cost therefore includes both models on all 400 pairs. Per-system
test comparisons concern 300 pairs; cascade resource totals are counterfactual
sequential sums from cached responses. In this context, saved responses are a
local reuse mechanism, distinct from provider prompt caching. The benchmark does
not make another API call when reusing a successful saved response.

The optional live `cascade` command additionally records `wall_latency_s` around
both actual stages and routing, including any quota pacing, and
`request_path_latency_s` excluding that pacing. Free-tier allowances, credits,
taxes, explicit cache-storage charges, and provider billing adjustments are not
included. Read actual spending from provider dashboards. These are small-sample
measurements affected by network and service conditions, not throughput benchmarks
or invoices. No warm-up requests are silently discarded. Measurement version 2
is recorded in new run manifests; older runs must use a new run directory.

### Relation to the two papers

[JEV-as-a-Judge, Section 4 and Appendix C](https://arxiv.org/html/2609.26550v6)
uses per-outcome timing that includes network/provider/retry time and excludes
initial pacing. Its headline statistics are median and p95 from a fixed 120-item
panel, with a persistent client, eight workers, and accounting outside the timer.
We adopt that latency definition and those headline statistics. Our simpler
experiment uses the same 300 test inputs for both judges, one request at a time,
with file writes and cost accounting outside the request timer. We do not claim
to replicate its concurrency or its separate timing-panel experiment.

[Decision Models vs. LLMs as Rubric Judges, Section 4.2 and Appendix B](https://arxiv.org/html/2609.29769v3)
primarily compares panel wall time, including checkpoint writes; concurrency and
rate limits affect those results. That is a different measurement from our
per-pair API latency. We retain retry frequency for context and do not call a sum
of saved API durations measured batch wall time or throughput. Retried pairs
stay in the primary latency results; no slow requests are silently filtered out.

## Actually route new pairs selectively

After analysis has frozen the policies, provide JSONL records containing `prompt`
and `response` (optionally `id`; labels are unnecessary):

```sh
python -m judge_exp.cli cascade --input examples/new_pairs.jsonl --output runs/live_fixed.jsonl
python -m judge_exp.cli cascade --policy class_dependent --input examples/new_pairs.jsonl --output runs/live_adaptive.jsonl
```

These commands call Gemini ONLY for escalated pairs. They stop on provider errors,
flush after each item, and refuse to overwrite an output file. Keep the original
run/config/policy for the frozen router. This optional command is for new inputs;
do not use it to replace the complete benchmark calls required for analysis.

## Verification without API keys

```sh
python -m pytest
python scripts/demo.py
```

The demo uses fabricated predictions and produces watermarked charts under
`runs/synthetic_demo/`. It verifies the report pipeline; it is NOT experiment
evidence and never belongs in the article's results. Tests cover controls,
deduplication, provider response contracts, routing boundaries, development
threshold selection, and failure accounting. Live API access needs your keys.

GitHub Actions runs the offline tests and demo on Windows and Linux with Python
3.11 and 3.13. It does not use API keys, download BeaverTails, or call paid models.
Reproduction means matching the sample and procedure; hosted model updates and
request variability can still change verdicts, costs, and latency. Raw provider
responses and any returned model-version information are retained locally.

## Project layout

```text
config.json                  Models, sample sizes, prices, threshold grid
.env.example                 Public template for private API credentials
prompts/policy.txt           Shared judging policy
src/judge_exp/
  cli.py                    Command-line entry point
  data.py                   Pinned download and reproducible sampling
  providers.py              Clef and Gemini request/response handling
  experiment.py             Saved responses and resumable collection
  routing.py                Fixed and class-dependent threshold selection
  report.py                 Metrics, uncertainty intervals, and figures
scripts/demo.py              Offline report check using synthetic predictions
tests/                       Offline checks
examples/new_pairs.jsonl     Small illustrative inputs for live routing
requirements-lock.txt        Locally tested dependency snapshot
data/                        Downloaded and sampled data (ignored)
runs/                        Saved responses, policies, reports (ignored)
```

## Troubleshooting

| Problem | What to check |
|---|---|
| HTTP 401/403 | API key/token permissions, Cloudflare account ID, and model access |
| HTTP 429 | Your account's quota; lower `gemini_requests_per_minute` or resume after reset |
| Gemini safety block / missing verdict | Inspect the saved error; it remains a failure rather than a negative label |
| Config/prompt/sample changed | Restore the original settings or choose a new `--run` directory |
| Analysis reports unattempted examples | Finish `collect` without `--limit` first |
| Missing cost in report | Provider token usage was unavailable; consult its dashboard |

For a separate experiment, place global options before the command:

```sh
python -m judge_exp.cli --config config.json --run runs/experiment_02 collect
python -m judge_exp.cli --config config.json --run runs/experiment_02 analyze
```

## Sources and attribution

- [Clef model card and actual typed API usage](https://huggingface.co/Cloudflare/clef)
- [Hosted Clef and pricing](https://developers.cloudflare.com/workers-ai/models/clef/)
- [Workers AI free allocation](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- [Gemini prices](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [BeaverTails dataset card, category definition, CC BY-NC 4.0 license](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)
- [BeaverTails paper](https://arxiv.org/abs/2307.04657)
- [JEV confidence cascade paper](https://arxiv.org/abs/2609.26550)
- [Decision models and shared judge errors](https://arxiv.org/abs/2609.29769)

BeaverTails authors: Jiaming Ji et al., *BeaverTails: Towards Improved Safety
Alignment of LLM via a Human-Preference Dataset* (2023). Dataset text and
annotations retain their original CC BY-NC 4.0 license. Sampling code downloads
them from the source; raw or sampled text is not intended for redistribution in
this code repository. Check the dataset license against intended publication
and reuse, and credit the dataset and papers in your article.
