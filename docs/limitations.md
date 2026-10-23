# Execution and measurement boundaries

## Family seeds and trials

For run-only workflows, `instances.prefix_seed_scope: family` derives a prefix seed
from the task seed and all sorted matrix coordinates, excluding the trial index.
The suffix batch seed derives from that family seed, trial index, and run node path.
Calibrated generation additionally derives a suffix seed for each request index.
Named inputs namespace prefix seeds by input name.

An explicit `run.prefix_seed` overrides prefix derivation and passes its integer
unchanged to generation, including named inputs. It may be a literal, matrix
reference, or deterministic `{pick: [...]}` selection. Picks are with replacement.
`{pick: {seed: PARENT, count: SIZE}}` generates the finite pool deterministically.
The parent seed controls pool membership independently of the task selection seed.
Selection occurs once per Runner; equal seeds can recur within and across rounds. Input configuration
must still match to reproduce text. The task seed and expanded node path select the
pool entry but do not modify the chosen seed. Suffix derivation stays separate.

```text
task seed + matrix coordinates -> family prefix seed -> stable prefix
family prefix seed + trial + node path -> suffix seed -> request suffixes
```

Trials only generate and submit work. They do not shuffle families, rotate quotas,
insert fairness delays, or control Worker admission. Runner decides when Worker
dependencies and due times permit submission. The shared Worker dispatcher and
request pool own admission, queueing, and slot allocation. Default matrix-major
compilation is unchanged. Competition can interleave families but does not promise
random or uniform recall.

The default `prefix_seed_scope: trial` preserves the previous seed algorithm.
Existing Campaigns retain their persisted configuration. Family mode requires a
Backend supporting this field and rejects workflows containing atomic invokes.
Identical prefixes require unchanged corpus revision, adapter, tokenizer, prefix
budget, generation path, and named input. Literal text is unchanged. Different
suffix seeds do not guarantee unique text in a repetitive or small corpus.

## Runner activation, creation, and Worker execution

The default active Runner ceiling is 1024 (`LLMPERF_RUNNERS_MAX`, maximum 16384).
It is a control-plane bound, not a request slot count or a fairness quota.
Backend aggregation uses eight threads by default (`LLMPERF_PREPARATION_THREADS`).
Activation creates lightweight Worker recipes, without allocating prompt batches.
Each recipe enters shared admission and the fair request queue before its Ray actor
prepares input, sends one Provider request, and normalizes the result. Actors never
access PostgreSQL. One preparation lock per request actor bounds CPU preparation;
the lock is released during Provider I/O. The actor's request slot remains occupied
through both stages. No separate preparation actors are started on this path.
`LLMPERF_PREPARATION_WORKERS` retains its existing CPU-headroom effect when sizing
request actors, so this update does not silently increase the shared slot count.

The request pool remains independently bounded: eight request actors with eight
slots each provide 64 simultaneous requests. Explicit Runner `concurrent_workers`
bounds outstanding Workers, including admission waits. A value of 1 allows one
outstanding Worker from that Runner. Omitting it delegates capacity to shared
Worker admission.

Default competition is work-conserving: ready Runners can use idle capacity without
reserving a per-Runner quota. Blocked admission attempts wait in semaphore order,
with at most one pending attempt per Runner. The request pool then rotates among
Runner queues, dispatching one Worker per nonempty queue per turn. A large backlog
from one Runner therefore does not outrank other queued Runners.

This fairness applies to ready, waiting work. It does not preempt requests already
running, reserve slots for Runners still preparing, or guarantee equal completion
rates when request lengths differ. It also does not guarantee uniform seed-family
sampling. Pressure workloads may omit `concurrent_workers`; fairness must not
depend on configuring that optional per-Runner ceiling.
The basic-pressure workload explicitly sets it to 8 as a defensive bound on
outstanding Workers per Runner. This does not reserve eight slots or move fairness
into Runner logic.

The existing dispatcher bounds submitted Workers awaiting completion/commit at
twice the physical request capacity (128 when capacity is 64). At capacity it awaits
a semaphore; each Runner retains at most one asynchronous submission attempt while
continuing to listen to its mailbox. Result commit or cancellation cleanup releases
admission capacity. There is no added retry loop or submission-limit setting.
The admitted queue holds recipes, not prebuilt prompt batches. Actor prefix caches
retain at most 128 calibration plans and an estimated 64 MiB of prefix characters
(one oversized plan may exceed that budget). Completed results and conversation
history still consume memory and durable storage; this is not a total memory cap.
1024 active Runners does not imply 1024 simultaneous preparations or running requests.

Cancellation prevents later submissions. An already executing synchronous actor
call is not forcibly interrupted; physical capacity remains occupied until it
finishes. A deadline is checked before preparation and again before Provider I/O,
including time spent waiting for the actor's preparation lock.

Calibrated concatenate workloads preserve their indexed family/trial/request seed
mapping. Other generated inputs use deterministic `worker-index-v1` per-request
seeds to avoid materializing an entire batch; their concrete text may differ from
older batch-generation versions. Family prefixes and repeat-count reuse remain
deterministic across actor placement. Explicit legacy cache-probe schedule settings
retain their requested order; ordinary trials add no random scheduling.

Validated runtime activity renews the ownership lease. Runner claims are serialized
before acquiring database connections, and reconciliation fetches nodes in one
query instead of one query per instance. Expired or superseded ownership is still
rejected; it is never silently revived or used to replay Provider requests.

## Status semantics

| Field | Meaning |
|---|---|
| `workers.planned` | Declared requests of materialized Runners; excludes future plan occurrences and unmaterialized nodes |
| `workers.created` | Lightweight Worker descriptions created; not a count of prepared prompts |
| `workers.submitted` | Cumulative durable single-send reservations, including terminal Workers |
| `workers.not_submitted` | Planned minus submitted; includes uncreated, dependency-delayed, and abandoned work |
| `workers.admission_waiting` | Submission attempts waiting for dispatcher capacity |
| `workers.queued` | Submitted requests waiting in the shared slot queue |
| `workers.running` | Workers holding slots, including actor preparation and Provider I/O |
| `workers.pending_results` | Reservations without committed results; not a running count |
| `workers.succeeded/failed/cancelled/indeterminate` | Durable Worker outcomes |
| `workers.creation_phases` | Runner counts waiting, creating descriptions, submitting, or aggregating |
| `request_slots` | Global shared busy/free/total slots, never a Campaign allocation |

Pool telemetry is sampled separately from PostgreSQL counts and may briefly disagree.
Its age is exposed. Unavailable or stale execution telemetry is null (`-`), not zero.
Legacy or interrupted Runners without creation evidence have unknown creation counts.
Reservation counts cover the durable Worker dispatcher; older execution paths may
lack them. Campaign lifecycle and outcome remain separate from Worker progress.

## Deadlines and cache evidence

Runner timeout bounds admission, actor preparation, and execution. Competition for 64 slots
can exhaust it before all planned Workers execute. Inspect actual counts and
`timed_out`; Runner success alone does not prove the planned count completed.
The basic workload uses a 32-by-32 seed matrix with no additional trials: 1024
families and 16 requests per Runner (16,384 requests total). Its 600-second Runner
deadline includes admission and actor preparation and can still truncate work.

An 80% local prefix budget is an offered workload property. Calibrated token counts
are estimates; Provider tokenization and block boundaries may differ. Prefix hashes
prove local content reuse, not DFS cache hits. Tier-specific online counters or logs
must distinguish device, memory, and DFS hits. This feature changes neither cache
routing nor retention.

## Matrix phases and long-running experiments

The framework provides matrix expansion, ordered completion barriers, bounded cycle
counts and deterministic seeds. It has no Mooncake tier, eviction policy, capacity
target or hit-ratio controller. Experiment roles belong in workload YAML.

`runner_groups` expands integer matrices into independent Runners.
`campaign.rounds.phases` orders arbitrary named phases within each cycle. All Runners
in a phase must become terminal before the next phase is eligible. This is a completion
barrier, not an all-success condition: failed, cancelled or indeterminate work can
allow later phases to proceed. Exclude incomplete cycles from cache analysis.
Readiness is reconstructed from durable states; restart does not replay interrupted
Provider calls or promise uninterrupted pressure.

With `rounds.prefix_seed_scope: family`, group name and sorted matrix coordinates
identify a family across phases and cycles. Keep input name, dataset revision,
tokenizer, prefix length and generation settings unchanged too. The suffix seed
depends on the family, cycle and phase; request indices change repeated suffixes.
Dimension names are arbitrary coordinates, not scheduling priorities. The same group
name is allowed in different phases, but duplicates within one phase are rejected.
Legacy ungrouped Runners use template position as identity.

`rounds.seed: date` (also the omitted-seed default) resolves UTC date/time plus a random
nonce into one numeric seed at API validation. Every expanded Runner persists it
in `metadata.runner_round.seed`; execution never consults the date again. Crossing
midnight preserves prefixes. Separate submissions, even on the same day, get new seeds.
Preview and artifact validation are separate requests and do not reserve a submission
seed; set an explicit integer to reproduce their exact inputs. Accepted Runner
metadata and exports retain the numeric seed.

All cycles are expanded and persisted upfront without preparing prompts. Cycles times
Runners per cycle must not exceed 10,000; there is no unbounded mode or lazy cycle
materialization. `rounds.count` limits work, not elapsed time. Budget throughput,
Runner deadlines, database results and export volume. Active Runner limits do not
bypass phase barriers or Worker slots. `worker.repeat` repeats within a Runner,
not across global phases or cycles.

The optional recall profiles use this configuration-only pattern:

```text
target recall → independent no-prefix pressure → target probe → next cycle
```

`shared_prefix_tokens: 0` removes intentional prefix reuse, not incidental corpus
overlap. Pressure seeds change full prompts each cycle. Provider completion does not
prove asynchronous DFS writes finished or memory eviction reached a target. Input
volume alone cannot guarantee a memory or DFS hit.

These profiles are capacity-calibration candidates, not measured hit guarantees:

| Workload | Target families | Pressure requests per cycle | Cycles | Total requests |
|---|---:|---:|---:|---:|
| `memory-recall.yaml` | 128 | 256 | 16 | 8,192 |
| `dfs-recall.yaml` | 128 | 640 | 8 | 7,168 |
| `tiered-recall.yaml` | 128 | 256 then 640 | 7 | 8,960 |

Each request offers 128K input tokens; targets retain approximately 80% prefixes.
The combined profile probes after each pressure phase. Adjust pressure matrices using
live occupancy, eviction and client-read evidence. Excess pressure can evict targets
from DFS too. Exclude the first cycle and pressure requests; compare target probes
by cycle/phase and distinguish byte/token hit fractions from key counts. Engine
`storage_hit` alone cannot distinguish Mooncake memory from DFS: use tier-specific
client evidence. Other workloads, routing, leases and topology changes affect ratios.
