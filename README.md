# System One as Judge: Clef → LLM

[![Python checks](https://github.com/ElmekaouiHaitham/system-one-as-judge/actions/workflows/checks.yml/badge.svg)](https://github.com/ElmekaouiHaitham/system-one-as-judge/actions/workflows/checks.yml)

Compare Cloudflare Clef (27B), an LLM judge, and confidence cascades on
BeaverTails' `discrimination,stereotype,injustice` category. Existing human labels
are the reference. No new annotation is required. This is an illustrative
reproducible experiment, with conclusions limited to its sample and label policy.

The original baseline uses Gemini 3.1 Flash-Lite. Separate Amazon Bedrock configs
support **GLM-5** and **Kimi K3** with the same judging policy and samples. These
adapters have offline contract tests; live access and model behavior require a
successful pilot with your own account.

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

Transport errors, HTTP errors, and exhausted retries stop collection with a nonzero
exit code so an invalid key or exhausted quota does not burn through a sample.
Fix the issue and rerun the same command: successful calls are reused and those
transient failures can be retried. A completed provider response with a content
block, invalid verdict, or truncation is instead retained as a terminal failed
verdict and collection continues; it is never rerun for a better answer. Previously
saved Gemini prompt blocks are also recognized on resume. Each response is
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

### Optional Gemini 3.8 Flash development pilot

The original Flash-Lite configuration and results can be retained as a baseline.
`configs/gemini-3.8-flash.json` defines a separate Flash experiment: medium
thinking (the model default), provider-default sampling, a 4,096-token output
ceiling including reasoning, and conservative pacing of five calls per minute.
The same rubric and exact sampled pairs are used. Current introductory list
prices apply through December 31, 2026; verify prices for later runs.
Gemini 3.8 does not support the old `minimal` thinking level, and its migration
guide advises omitting deprecated sampling controls. See the
[official migration guide](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.8).

```sh
python -m judge_exp.cli --config configs/gemini-3.8-flash.json --run runs/gemini-3.8-flash collect --split dev --provider gemini --limit 20
```

This calls only the first 20 already-shuffled development pairs and saves them in
a separate run. It is a development pilot, not a held-out test result. Full
`analyze` still requires all development/test predictions for both judges. Use
the same command without `--limit 20` to finish Gemini development collection;
retain the model settings and use a new run for any changes. Clef predictions
from the earlier experiment can be reused only after checking the rubric, exact
sample hashes and Clef configuration match; record their provenance. Reused Clef
timings were measured earlier, not simultaneously with the new Flash run.

### Default sampling

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
  and the count of fully priced pairs. The main latency chart uses median and p95
  on a logarithmic seconds axis so both short responses and long retry tails are visible.
- `predictions.csv`: IDs and per-system verdicts for tracing disagreements.
- `control_slices.csv`: results for positives and safe controls.
- `cascade_diagnostics.json`: fallback corrections, regressions, shared errors.
- `confusion_matrices.png`, `cost_latency.png`, `coverage_tradeoff.png`: article
  figures. Also a diagnostic threshold-sweep CSV.
- `clef_roc.png`: ROC curves for category classification and error detection.

If required calls have missing usage, the cost chart hatches that system's bar
and shows its known list-cost subtotal rather than inventing a complete total.
`list_cost_usd` remains unavailable in the table; the subtotal is reported
separately. Do not interpret incomplete-cost bars as exact totals or use them to
claim an exact savings percentage.

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
When a transient failure is retried after restarting collection, analysis combines
the failed and subsequent call durations, attempts, and known token costs for that
example. Time between collector sessions is excluded. The original log records
remain unchanged; the separate collection ledger reads every physical call once.

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

## Amazon Bedrock: GLM-5 and Kimi K3

Use **US East (N. Virginia), `us-east-1`**, for both example configs. In the
[Bedrock console](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1),
open **API keys → Long-term API keys → Generate**, and choose a short expiration
such as seven days. These exploratory keys avoid creating general-purpose AWS
access keys. Make sure the key's identity can invoke the models; a valid key does
not by itself guarantee account/model access or Free plan eligibility.

Save the key privately in the existing `.env`:

```dotenv
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_key
```

The configs pin Standard on-demand inference and these November 11, 2026 list prices:

| Config | Model/profile | USD per million input / output tokens |
|---|---|---|
| `configs/bedrock-glm-5.json` | `zai.glm-5`, US East | $1.00 / $3.20 |
| `configs/bedrock-kimi-k3.json` | `global.moonshotai.kimi-k3` | $3.00 / $15.00 |

Different regions and inference profiles can have different prices. Credits are
not deducted from estimated list costs, and trial-credit coverage isn't assumed.

If your key was generated in **Stockholm (`eu-north-1`)**, use
`configs/bedrock-kimi-k3-stockholm.json` with a separate
`runs/bedrock-kimi-k3-stockholm` directory. It uses the same Global Kimi K3
inference profile and its Global list rates, with the request sent from Stockholm.
Short-term keys work only in the Region where they were generated and last up
to 12 hours. No credential or decoded token is stored in the config or manifest.
The collector can resume with a fresh key generated in the same Region.

Keep one run directory per fallback model. If you already completed Clef in
`runs/main`, import it with compatibility checks and a provenance record:

```sh
python -m judge_exp.cli --config configs/bedrock-glm-5.json --run runs/bedrock-glm-5 reuse-clef --source runs/main
python -m judge_exp.cli --config configs/bedrock-kimi-k3.json --run runs/bedrock-kimi-k3 reuse-clef --source runs/main
```

Otherwise, collect Clef once using `--provider clef` in one of these runs, then
reuse that run as the source for the other. Importing makes **no API calls**.

Start with development pilots:

```sh
python -m judge_exp.cli --config configs/bedrock-glm-5.json --run runs/bedrock-glm-5 collect --provider fallback --split dev --limit 20
python -m judge_exp.cli --config configs/bedrock-kimi-k3.json --run runs/bedrock-kimi-k3 collect --provider fallback --split dev --limit 20
```

To finish the complete **100-development / 300-test** experiment sequentially,
including threshold freezing and report generation, use:

```sh
python -m judge_exp.cli --config configs/bedrock-kimi-k3-stockholm.json --run runs/bedrock-kimi-k3-stockholm full --requests-per-minute 2
```

This command resumes saved successes; it doesn't submit the dataset as a bulk
job. There is one in-flight request, initially at most two Bedrock attempts per
minute. A persistent local scheduler spaces retries as well as new examples.
HTTP 429 causes a cooldown of at least 60 seconds, increasing to 120 seconds for
consecutive throttles. After success, the steady configured spacing resumes.
`Retry-After` is
honored, including HTTP-date and fractional-second values. The rate can never
exceed the config's ceiling. The scheduler state survives process restarts.
There are bounded automatic resumes for transient transport/service failures;
access/configuration errors stop immediately. Successful or terminally blocked
verdicts are never rerun to obtain a different answer.

Collection-session rate settings are recorded in `transport_sessions.jsonl`,
and new Bedrock outcomes identify the limiter strategy and effective interval.
The model/rubric/sample manifest stays immutable: changing transport pacing is
recorded separately from changing judging settings. Earlier outcomes collected
under the original five-per-minute ceiling remain in the same cache with their
original timings and retry history. Never describe the full run as having used
one fixed request rate throughout.

An initial `persistent_adaptive_v1` session also slowed future examples after a
429. Once the account's reported K3 quota was checked (10M TPM), transport was
changed to `persistent_paced_cooldown_v2`: retain a steady ceiling and apply
cooldowns to throttled attempts. Shared-capacity errors can occur below quotas;
the quota alone doesn't establish the cause of a specific error. Session and
per-outcome strategy fields identify which transport policy was used.

The workflow freezes `routing_policy_before_test.json` using only development
data, collects the held-out test rows, then generates the report. Simulated
cascades reuse these independent cached calls. Planned queue pacing occurs
outside the outcome timer; cooldowns required by retries stay in retry latency.
`pre_request_pacing_s` records planned waiting on completed Bedrock calls.
If an automatic collector restart retries the same unresolved example, its
persisted cooldown is included in that outcome's aggregated retry time. Manual
gaps between sessions remain excluded. Immutable per-call records are retained.
The CLI also accepts `collect --requests-per-minute 2` for a single split.

After checking reliability, finish development with the same command without
`--limit`. Fix the model/settings using development data, then collect test with
`--split test` and run `analyze` using the matching `--config` and `--run`.
To preserve compatibility with historical runs, the fallback cache files and
some machine-readable keys still use the internal name `gemini`; Bedrock records
also identify `backend`, `model`, and `region`, and report display labels name the
actual model. Neither a Gemini key nor new Gemini calls are used in these runs.

Both adapters use Bedrock's **Converse** endpoint with schema-constrained JSON,
the original policy as the system instruction, and only prompt/response as user
data. Human labels and Clef predictions are never included. Reasoning stays at
the provider default; **Kimi K3 defaults to max**. No unsupported effort override
is sent on Converse. The 8,192-token output cap includes reasoning; truncations
remain failed outcomes. A failed output isn't retried for a different verdict.
An unsupported-schema/configuration response stops collection for inspection;
it never silently changes the judging protocol.

### Two distinct latency measurements

- `latency_s`: measured client outcome time, including network, provider time,
  retries and backoff, excluding deliberate pre-request quota pacing. This is
  the common basis used for Clef, each fallback, and simulated cascades.
- `provider_latency_s`: Bedrock's returned `metrics.latencyMs / 1000` for the
  final HTTP response, when available. This is an additional provider diagnostic,
  not a measure of pure model computation and not the total after retries.
- `provider_attempt_latency_sum_s`: sum of reported service times only if every
  attempt reports a value; otherwise unknown. Restart aggregation preserves this
  rule. Individual attempt times remain in the raw cache.

Reports include `latency_diagnostics.csv` with both measurements. Do not compare
Bedrock-reported time directly with Clef client time or add those two bases in
one cascade. Cached Clef timings are from an earlier collection period: reuse is
appropriate for quality and estimated routing resources, but controlled speed
claims need fresh timings collected under comparable conditions. New JSON
schemas can require a first-call compilation; retain that call and identify it
when reporting cold versus warm performance.

`paired_first_attempt_latency.json` / `.csv` and its figure provide an additional
conditional comparison: keep only the same examples where **both** judges
returned valid verdicts on their first HTTP attempt, and publish the retained
and excluded counts. This prevents comparing different retry-free subsets.
This diagnostic doesn't replace the full operational latency table and doesn't
measure pure model computation. Optional AWS-reported statistics remain separate.

Bedrock's `outputTokens` already includes reasoning, so it is charged once.
Cached reads/writes are included once in total input tokens. The uncached list
estimate and cache-adjusted estimate remain separate; missing usage or unknown
cache-write pricing stays unknown. Reused Clef calls count in simulated system
resources but add no new calls/cost to the run's physical collection ledger.

Sources: [API keys](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html),
[Converse latency](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseMetrics.html),
[structured outputs](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html),
[GLM-5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5.html),
[Kimi K3 and reasoning defaults](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html),
[prices by region](https://aws.amazon.com/bedrock/pricing/).

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
