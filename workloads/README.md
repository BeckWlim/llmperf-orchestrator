# Background pressure and periodic L4 recall

This directory contains two independent campaigns for `dfs-beta` /
`spark-x2.5-1.7b-L4`. Both use 65,536 input tokens, a 52,428-token shared
prefix, a fresh suffix, 16 output tokens, and one request per Worker.
Inputs use the default `input_token_count: estimate`: the pinned general tokenizer
calibrates a corpus ratio once per request actor, reused across requests. These
token numbers are targets; inference-reported usage is authoritative. Set
`input_token_count: tokenizer` explicitly for per-input tokenization.

| Campaign | Shape | Request budget | Maximum outstanding requests |
| --- | --- | ---: | ---: |
| `basic-pressure.yaml` | 1,536 distinct-seed Runners; 156 Workers each; 544-second completion-relative cooldown | 239,616 | 24 |
| `l4-recall.yaml` | Prime 256 prefixes, then 36 recall waves; 32 serial lanes of eight members per wave | 9,472 | 32 |

The background uses the generic submission ceiling `runtime.concurrent_workers: 24`.
Recall uses 32 serial lanes. Together they offer at most 56 Workers to the 64-slot
pool, without reserving capacity from other submissions. Runners release active
Runner admission during cooldown and compete for it again when due; the existing
1,024 active Runner limit does not need to increase to cover every seed.

## Evidence and sizing assumptions

Inspected on 2026-10-21 in namespace `spark-llm`, cluster
`spark_slm_v0_3_13_post1_dfs_beta`:

- L3 capacity: 1,000 GiB; configured eviction trigger: 80%, with a 20% eviction ratio.
- L4 physical capacity: 4,916 GiB (1,229 buckets x 4 GiB); high/low watermarks:
  80% / 60%, or 3,932.8 / 2,949.6 GiB. The high watermark is roughly 4 TiB.
- Between 10:57:30 and 10:58:18 CST, both tier-hit byte counters stayed flat,
  while L3/L4 eviction increased by approximately 358/432 GiB and allocation
  failures increased by 200,472. These are cache-object operations, not failed
  inference requests or physical disk bandwidth measurements.
- One prefill pod's retained traces, 09:57:43–11:04:56 CST, contained 3,936,730
  memory-read objects and 378 DFS-read objects. This is a partial pod/time sample.
  Yesterday's inference traces were not retained in that log, so a numerical
  yesterday/today DFS-hit comparison is unavailable.
- The old campaign's `repeat: 20` enlarged every pressure interval by 20x.
  Original memory/DFS-pressure phases offered about 0.4375/1.09375 TiB of
  full-attention KV; scaled phases offered 8.75/21.875 TiB. Successful inference
  does not establish successful KV persistence.

The deployed model has seven full-attention layers, two KV heads, head dimension
256, BF16, and TP=2. Across both ranks, full-attention KV is approximately
`7 * 2 * 2 * 256 * 2 = 14,336 bytes/token`. Its other 21 layers use sliding-window
attention. The estimates below **exclude their storage/checkpoints**, alignment,
extra replicas, and differences between the workload tokenizer and provider
tokenizer. They are a starting budget, not a measured stored footprint.

| Estimated full-attention footprint | GiB |
| --- | ---: |
| One prefix | 0.700 |
| One fresh suffix | 0.175 |
| Background pool: 1,536 prefixes | 1,075.18 |
| Probe cohort: 256 prefixes | 179.20 |
| L4 low-watermark space after reserving both prefix sets | 1,695.22 |

## Background behavior

The task compiler enumerates the generated 1,536-seed pool exactly once. Each
Runner keeps its own unique prefix seed for all 156 Workers. Suffix seeds change
between Workers. This replaces 240,000 one-request Runners with 1,536 repeated
Runners and 239,616 Workers, reducing the request budget by 0.16%.

`worker.every: 544` is a Runner-local cooldown measured from the previous Worker
completion. No dependency graph is generated for these repetitions. During cooldown,
the Runner releases its active admission slot and consumes no Worker slot. It retains
only its progression state and competes for admission again when due. The submission
ceiling limits physical Worker concurrency to 24 across all its Runners.

The gap-only rate ceiling is 1,536 / 544 = 24 / 8.5 = 2.824 requests/s, matching
the old configuration's gap-only ceiling. This is an approximation: the first wave
is not staggered, the seed sequence is now exhaustive rather than random draws with
replacement, and provider/admission latency changes revisit timing. At mean service
and admission latency L, a rough rate match to the old shape would use a cooldown
of `544 + 63L` seconds. Measure actual traffic and cache hits before tuning it.

Each Runner has 23 h 25 m 20 s of waiting plus execution/admission time, bounded by
its 48-hour timeout. An individual Worker error still allows its Runner's remaining
independent repetitions; there are no retries. A failed Runner makes the enclosing
task instance fail under existing task semantics, which may cancel siblings not yet
started. This is a pressure workload, not a strict schedule or guaranteed cache-hit rate.

## L4 recall behavior

```text
Prime the 256-prefix cohort (32 serial lanes)
  -> wait 10 min -> recall all 256 once
  -> wait 20 min -> recall all 256 once
  -> wait 40 min -> recall all 256 once
  -> repeat 33 times: wait 40 min -> recall all 256 once
```

Each wave waits for the previous whole wave to succeed. Its lanes each contain
eight serial, one-request Runners. Prefixes stay identical across waves; suffix
seeds change with every node path. The cohort is generated once from parent
seed `202610211102` by `for_each.values`; aliases reuse its compact definition. Independent
random picks at prime and recall would not preserve paired prefix identity.
Change the generated pool's parent seed or count to select another cohort.
Its seeds are checked to be disjoint from the background pool.
The periodic part uses `repeat.every`; `for_each.parallelism: 32` expands into
32 static serial lanes and preserves the whole-wave barrier. Every wave enumerates
all 256 seeds exactly once. Each child Runner sends one request, so the lane count
also bounds request concurrency. This is compiler expansion, with no runtime loop.

The delays total 23 h 10 min, plus execution/admission time. They start at wave
completion, not at scheduled wall-clock times. Repeated reads can refresh cache
residency: this measures access-conditioned recall, not untouched-prefix TTL.
A failed node terminates this task instance; no ambiguous cache probe is retried.

The 10/20/40-minute waves are calibration points. The 40-minute steady interval
is provisional until the new background's admitted write rate and tier reads
are measured. For example, **assuming** 0.4 GiB/s of new stored KV, one 800-GiB
L3 turnover takes about 33 minutes, and the reserved L4 churn headroom lasts
about 71 minutes. At the old write-heavy rate, 40 minutes can be much too late.
Do not reuse that old rate as a prediction for this background.

A recall wave offers up to 179.2 GiB of reusable full-attention prefix data;
36 waves offer approximately 6.30 TiB. Actual L4 reads can be lower if prefixes
remain in upper tiers, were never stored successfully, or have already been evicted.

## Evaluate and regenerate

From the repository or an updated installed environment:

```bash
.venv/bin/python -m llmperf_tools.cache_sizing \
  --l3-gib 1000 --l4-gib 4916 --kv-bytes-per-token 14336
```

The JSON output reports footprints, concurrency, request counts, wait budgets,
and warnings. `--write-gib-per-second` accepts an observed admitted write rate
under the proposed background; `--mean-request-seconds` estimates duration.
Without those observations, the tool leaves timing/rate conclusions uncalibrated.
For a hypothetical 1 TiB / 4 TiB system with a 90% L3 trigger, use
`--l3-gib 1024 --l4-gib 4096 --l3-high 0.9`.

Regenerate both YAMLs with adjusted pool/cohort sizes, gaps or concurrency:

```bash
.venv/bin/python -m llmperf_tools.cache_sizing \
  --l3-gib 1000 --l4-gib 4916 --kv-bytes-per-token 14336 \
  --pool-size 1536 --cohort-size 256 \
  --background-concurrent-workers 24 --probe-lanes 32 --total-slots 64 \
  --runner-template workloads/basic-pressure.yaml --output-dir workloads
```

Change `--background-pool-seed` and `--probe-pool-seed` for independent prefix
families in a later experiment. Output mode overwrites only the two named YAMLs;
it does not deploy a service, change capacities or submit a campaign. The tool
checks grammar expansion, concurrency bounds, node limits, planned spans,
seed separation and shared concurrency. Cache fit warnings remain advisory.

## Validate, start and evaluate

Both CLI and Backend must include the `for_each` / `run.prefix_seed` / generated-pool
grammar. Updating YAML alone is insufficient. The new submission runtime, compact Worker records, and file-backed log index
require a fresh database for this deployment. The sizing helper is separate from Scheduler/Worker runtime.

Static validation for the inspected 32-CPU runtime (16 actors x 1.5 CPUs x four
requests, with eight CPUs reserved by the validation model for preparation):

```bash
.venv/bin/python .codex/skills/operate-llmperf/scripts/validate_workload.py \
  workloads/basic-pressure.yaml --scheduler-slots 1024 --ray-num-cpus 32 \
  --ray-actor-num-cpus 1.5 --ray-actor-max-concurrency 4 --preparation-workers 8 \
  --max-runner-concurrency 64 --max-effective-concurrency 64
```

Repeat for `l4-recall.yaml`. Totals exceed packaged advisory budgets; the inspected
Backend has campaign-total admission disabled. The background expands to 1,536 durable task nodes. The pair offers 249,088
Workers and 16,328,216,576 configured input/output tokens.

After the Backend update, validate artifacts, prove a short smoke, start the
background, then start the probe once background progress is healthy:

```bash
llmperfctl campaign validate -f workloads/basic-pressure.yaml
llmperfctl campaign validate -f workloads/l4-recall.yaml
llmperfctl campaign start -f workloads/basic-pressure.yaml
llmperfctl campaign start -f workloads/l4-recall.yaml
```

Record both Campaign IDs and the actual prime/recall timestamps. Correlate
probe node paths (`prime`, `calibration_*`, `sustained_recall`) and request IDs
with inference traces. Master hit bytes describe replica-list results; do not
report them as completed physical read bandwidth. Prefer inference-side
`mem_reads`, `dfs_reads`, transferred bytes, read latency and errors for tier
classification, together with request throughput and TTFT p50/p95/p99.

If L3 dominates, increase the revisit gap or admitted background churn. If the
prefix is absent from both tiers, shorten the gap or reduce churn after checking
that priming stored successfully. If allocation failures rise, reduce background
concurrency or increase its pacing gap. Adjust only one dimension at a time and
record background throughput during recall waves; slot contention can otherwise
make a probe concurrency comparison misleading.

Previous workload files are archived remotely under
`~/llmperf-test/workload-archive/before-dual-campaigns-20261021-111251/`.
