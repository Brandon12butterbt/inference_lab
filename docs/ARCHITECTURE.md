# InferenceLab — Architecture v1.0 (revision r2)

**Status:** Revision r2, approved by the owner. Implementation proceeds milestone by milestone (M1 authorized incrementally by work package); this document alone does not authorize any work package.  
**Scope:** One-machine, local, single-user benchmarking app.  
**Precedence (benchmarking requirements):** `BENCHMARK_PROTOCOL.md` > `SPEC.md` > `ARCHITECTURE.md` > `TASKS.md`.

## 1. Architecture in words

`Streamlit UI (app.py)` → `application service (service.py)` → (`benchmark worker subprocess`, `SQLite repository`).  
The worker subprocess runs the **benchmark engine**, which invokes model loading, the deterministic fixture, parity validation, timing, memory profiling, metric analysis and hardware capture. The engine must be **independent of Streamlit and of SQLite** so it can be tested headlessly and later reused by a CLI or API if wanted.

**Staged execution model:**
- **M1–M2 (in-process):** the engine is a plain library function, `run_benchmark(config) -> ExperimentResult`, called directly by tests. There is no subprocess, no database and no UI. Each of M1 and M2 is independently testable. Results from these runs carry `process_isolation = "shared_process"`.
- **M3 onward (subprocess per configuration):** `service.py` launches **one short-lived local subprocess per configuration** (`python -m inference_lab.worker`, using `sys.executable`, i.e. the project's `.venv`), waits for it, and persists the outcome. The worker calls the same `run_benchmark` function used in M1–M2 and runs in a fresh process, which gives clean host-RSS and CUDA allocator state, and isolates CUDA OOM or crashes from the UI. Results carry `process_isolation = "subprocess"`.

Constraints that remain: no web server beyond Streamlit's local server; no FastAPI, Docker, Redis, cloud, job queue, scheduler, long-lived worker pool or database server; no new dependencies (the orchestration uses the standard library only). **One experiment runs at a time, sequentially.**

## 2. Intended repository layout

```text
inference_lab/                     # project root, contains pyproject.toml
├── CLAUDE.md                      # project-level Claude Code instructions
├── app.py                         # Streamlit entry point (M4)
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── docs/
│   ├── SPEC.md
│   ├── ARCHITECTURE.md
│   ├── BENCHMARK_PROTOCOL.md
│   └── TASKS.md
├── src/
│   └── inference_lab/             # Python import package; underscore is intentional
│       ├── __init__.py
│       ├── errors.py              # M1  error taxonomy
│       ├── config.py              # M1
│       ├── dataset.py             # M1
│       ├── fixtures/              # M1  committed fixture source data + manifest
│       ├── models.py              # M1
│       ├── parity.py              # M1
│       ├── metrics.py             # M2
│       ├── timing.py              # M2
│       ├── memory.py              # M2
│       ├── hardware.py            # M2
│       ├── benchmark.py           # M2
│       ├── storage.py             # M3
│       ├── worker.py              # M3  subprocess entry point
│       └── service.py             # M3
├── tests/
│   ├── conftest.py                # markers: gpu, slow, integration (config needs owner approval)
│   ├── test_config.py             # M1
│   ├── test_dataset.py            # M1
│   ├── test_parity.py             # M1
│   ├── test_metrics.py            # M2
│   ├── test_benchmark.py          # M2 headless integration (CPU; GPU conditional)
│   ├── test_gpu_smoke.py          # M1/M2 conditional GPU tests
│   ├── test_storage.py            # M3
│   ├── test_service.py            # M3 (includes subprocess integration)
│   └── test_worker.py             # M3
└── data/
    └── .gitkeep                   # databases, run files and generated data ignored
```

The project/directory name `inference_lab`, distribution name `inference-lab` and Python import package `inference_lab` are related but distinct. Do not reintroduce an erroneous `src/inferencelab/` folder. The fixture source data and manifest live in the package (`src/inference_lab/fixtures/`), **not** under `data/`, because `data/` is gitignored and the fixture must be version-controlled.

## 3. Module responsibilities

| Module | Responsibility | Avoid |
|---|---|---|
| `errors.py` | Exception types and the error-code taxonomy (§6) shared by engine, worker and service | UI code; message formatting for display |
| `config.py` | Typed benchmark configuration, allowed values, validation, protocol constants (iteration counts, thresholds, tolerances) | UI code or direct GPU calls |
| `dataset.py` | Generate the deterministic 512-text fixture from committed source data; verify against the manifest; tokenize/pad/truncate to 128; batch positions and indexing | Timed tokenization; private datasets; external downloads |
| `models.py` | Fetch/load the approved checkpoint at the pinned revision, explicit FP32, eval mode, requested/resolved attention backend, device placement | Automatic version upgrades, model training |
| `parity.py` | Untimed CPU/CUDA output comparison under the documented tolerance; structured `ParityResult` | Timing; persistence |
| `timing.py` | CPU timer, CUDA synchronization, per-batch timing samples | Reporting asynchronous dispatch time as execution |
| `memory.py` | Separate configuration-level memory-profile pass: host RSS fields and CUDA allocator baselines/peaks | Calling total system RAM 'model memory'; background sampling threads |
| `metrics.py` | Pure functions: P50/P95, per-trial medians, summed batch seconds, throughput, spread and stability flag, units | Device or database access |
| `hardware.py` | Hardware/software/environment snapshot (thread settings, precision flags, power plan, driver) | User secrets / personal identifiers; changing system settings |
| `benchmark.py` | `run_benchmark(config)`: apply controlled settings, parity check, warm-up, trials, validation, memory pass; returns an `ExperimentResult` | Streamlit imports, SQL, subprocess management |
| `worker.py` (M3) | Subprocess entry point: read config JSON, call `run_benchmark`, write result/error JSON, exit code | SQLite access; Streamlit imports; long-running state |
| `storage.py` (M3) | SQLite schema, persistence, read/query history, transaction handling, lifecycle updates | GPU timing or model execution |
| `service.py` (M3) | Validate config; own the lifecycle; launch the worker; persist results (sole DB writer); stale-run reconciliation; **single owner of cross-run comparison** | UI rendering |
| `app.py` (M4) | Streamlit controls, status, charts, rendering of what the service returns | Benchmarks, SQL, statistics or comparison logic embedded in widgets |

A module file is added only when its corresponding milestone is implemented. Empty placeholders are not required.

## 4. Interfaces / data contracts

All contracts are plain, JSON-serializable data structures (`to_dict` / `from_dict`) so they can cross the worker process boundary unchanged.

### BenchmarkConfig
- `model_id`; `model_revision` (pinned 40-char commit hash)
- `device`: `cpu` or `cuda:0`
- `batch_size`: one of 1, 4, 8, 16, 32
- `sequence_length`: 128
- `precision`: `fp32`
- `matmul_precision`: `highest` (set explicitly)
- `attn_implementation`: requested backend (default `sdpa`, subject to M1 verification)
- `warmup_iterations`: 10
- `measured_iterations_per_trial`: 50
- `trials`: 3
- `memory_profile_forwards`: 10
- `fixture_id`, `fixture_seed` (42)
- `cpu_threads`: hardware-aware default — physical core count from `psutil`, else logical count, else 1 — resolved to a concrete integer when the config is created without an explicit value and saved with each experiment. Explicit values must be positive integers, are never capped by local hardware, and reload unchanged from stored configs (12 on the reference machine)

For real runs, validation rejects iteration counts other than the protocol values. Only unit tests with injected fake forward callables use smaller counts, and their results are never persisted as experiments.

### FixtureMetadata (recorded in every result)
- `fixture_id`, `seed`, generator version, `text_sha256`, `token_ids_sha256`, `attention_mask_sha256`, tokenizer/model revision, real-token-length min/median/max, truncated count, `distinct_inputs_per_trial`.

### ParityResult
- `status` (`passed` / `failed` / `not_applicable`), `atol`, `rtol`, `max_abs_diff`, `max_rel_diff`, `labels_match`, `batch_position`, `reference_device`, `compared_device`.

### TrialResult
- `trial_index`
- `latency_samples_ms`: the chronological **50 batch-latency samples** in milliseconds
- `trial_median_ms`
- `sum_batch_execution_seconds`: sum of the 50 samples ÷ 1000 (explicitly **not** loop wall-clock time)
- `total_processed_inputs`: `batch_size × 50`
- `throughput_inputs_per_sec`: `total_processed_inputs / sum_batch_execution_seconds`

### MemoryProfile (one per experiment; configuration-level)
- Host: `host_rss_baseline_bytes`, `host_rss_after_profile_bytes`, `host_rss_delta_bytes`, `host_peak_working_set_bytes` (or `N/A`), `process_isolation`.
- CUDA (CUDA runs only): `cuda_allocated_baseline_bytes`, `cuda_reserved_baseline_bytes`, `cuda_peak_allocated_bytes`, `cuda_peak_reserved_bytes`, plus above-baseline values.
- `method` / `scope` description text; definitions in `BENCHMARK_PROTOCOL.md` §5. Memory is **not** attached to individual trials.

### AggregateSummary
- `p50_batch_latency_ms`, `p95_batch_latency_ms`, `min_batch_latency_ms`, `max_batch_latency_ms`
- `aggregate_throughput_inputs_per_sec`, `total_processed_inputs`, `sum_batch_execution_seconds`
- `trial_median_spread_pct`, `stability_flag` (`stable` / `unstable`)

### ExperimentResult
- Immutable `experiment_id` (UUID), `created_at_utc`, `started_at_utc`, `finished_at_utc`, lifecycle `status`, `error_code` / `error_message` (failures only), normalized `BenchmarkConfig`, model metadata (including resolved revision and resolved attention backend), `FixtureMetadata`, hardware/software/environment metadata, `ParityResult`, trials, `MemoryProfile`, `AggregateSummary` (completed runs only), `methodology_id` (`inferencelab-protocol-v0.2`).
- Do not call per-batch latency "request latency"; no arrival, queue, network or tokenization costs are timed in v1.

Keep units explicit (`ms`, `s`, `bytes`, `inputs/sec`). Do not store only rounded strings; persist numeric measurements and format at the UI boundary.

## 5. SQLite boundaries and experiment lifecycle

### Lifecycle states
| State | Meaning |
|---|---|
| `running` | Row created when the run starts; no results are reportable yet |
| `completed` | All validations passed and all samples, aggregates, parity and memory profile were stored in one transaction |
| `failed` | A handled failure occurred; `error_code` and `error_message` are set; no aggregate metrics are stored as results |
| `interrupted` | The run was abandoned (application closed or crashed) while `running`; detected at the next application startup |

Transitions: `running → completed`, `running → failed`, `running → interrupted`. Terminal states never change. Only `completed` experiments appear in comparisons and charts as measurements; failed and interrupted runs are listed with their state and error.

### Storage rules
- Database location: ignored local `data/inferencelab.sqlite3` (or comparable name) within project boundaries.
- Suggested relational records: `experiments` (config, metadata, status, error, timestamps, methodology id, stability flag, aggregates); `trials` (per-trial totals, `trial_median_ms`, `sum_batch_execution_seconds`, `throughput_inputs_per_sec`); `latency_samples` (raw batch durations); `parity_results` (0..1 per experiment); `memory_profiles` (**0..1 per experiment**, keyed by `experiment_id`, not by trial).
- Use parameterized queries and transactions; enable `PRAGMA foreign_keys = ON` on every connection; use `PRAGMA user_version` as the schema version. Create tables if absent; document incompatible changes rather than silently resetting data.
- **Single writer:** only the parent process (`service.py`) writes to SQLite. The worker never touches the database.
- The service singleton is created once per application process. At initialization it marks any leftover `running` rows as `interrupted`. Only one Streamlit app instance per database is supported.
- Atomicity: a completed result is written in a single transaction; on any failure the `running` row is updated to `failed` with details. Incomplete results are never labeled successful.

## 6. Error and availability behavior

Errors carry a stable `error_code` and a human-readable message. Initial taxonomy (`errors.py`):

| `error_code` | Situation |
|---|---|
| `CONFIG_INVALID` | Unsupported value, protocol-count mismatch, matmul precision not `highest` |
| `CUDA_UNAVAILABLE` | CUDA requested but not available or the device cannot run a real computation |
| `MODEL_LOAD_FAILED` | Download/initialization failure or revision not found |
| `ATTENTION_BACKEND_MISMATCH` | Resolved attention backend differs from the requested one |
| `FIXTURE_INTEGRITY_FAILED` | Fixture hashes do not match the manifest |
| `PARITY_FAILED` | CPU/CUDA output validation outside tolerance, non-finite logits, or label mismatch |
| `INVALID_MEASUREMENT` | Wrong sample count, NaN, non-positive duration, empty samples |
| `CUDA_OOM` | GPU out of memory; the message suggests a smaller batch size, but the batch size is never changed automatically |
| `CUDA_RUNTIME_ERROR` | Any other CUDA failure |
| `WORKER_CRASHED` | Worker exited abnormally or produced no valid result (stderr tail captured) |
| `STORAGE_ERROR` | Database I/O failure; already persisted results are never discarded |
| `INTERRUPTED` | Used for experiments reconciled as `interrupted` |

Behavior:
- CUDA unavailable → CPU still works; CUDA controls are unavailable or show the actionable error. Never fall back silently and mislabel as GPU.
- Model download/initialization failure → explanatory message, no fake benchmark results.
- Database I/O failure → show the error; never discard already persisted results quietly.
- Background GPU workloads may alter results; show a note that results reflect current hardware conditions.

## 7. Testing boundaries

- **Unit:** `metrics.py`, `config.py`, `parity.py`, `storage.py`, batch sequencing and fixture generation with small deterministic data; no model download required (the text hash test needs no model).
- **Integration (slow, require cached model assets):** CPU model forward pass; **M2 headless benchmark integration test** — `run_benchmark` on CPU with the full protocol parameters at batch size 1 (10 warm-ups, 3 × 50 measured), asserting 150 valid samples, metrics consistency and JSON round-trip, with no database or UI; **M3 subprocess integration test** — the same configuration through `service.py` and the real worker, asserting a persisted `completed` experiment.
- **GPU (conditional):** marked `gpu`; skips cleanly when CUDA is unavailable; validates a real CUDA forward computation, the CPU/CUDA parity check, and synchronized timing. On the reference machine these must be executed and passing, not skipped, to count as GPU acceptance.
- **Failure paths (M3):** worker killed or non-zero exit → `failed` / `WORKER_CRASHED`; stale `running` row → `interrupted`; second concurrent run refused. These use simulated workers and temporary databases (unit-test fixtures only).
- **Dashboard:** UI smoke check and history rendering later; no hard-coded results.
- **Quality:** `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`.

## 8. Subprocess orchestration (introduced in M3)

- **Launch:** `subprocess` from the standard library, command `[sys.executable, "-m", "inference_lab.worker", "--config", <path>, "--result", <path>]`. Files live under ignored `data/runs/<experiment_id>/`.
- **Contract:** worker reads the config JSON, calls `run_benchmark`, writes a result JSON (success) or a structured error JSON (handled failure, with `error_code`), and exits `0` / non-zero accordingly. No result file after exit means `WORKER_CRASHED`; stderr tail is kept as the error detail.
- **Ownership:** the parent creates the `running` row, launches the worker, validates the returned JSON, then persists the final state. The worker is stateless and exits after one configuration; it is never reused.
- **Sequential execution:** the service holds an in-process lock so only one run is active; a second start request is refused with a clear message. Multi-configuration requests run one worker after another in the fixed order defined in `BENCHMARK_PROTOCOL.md` §3.
- **Streamlit rerun safety (assumption to be verified in M4):** `service.run_experiment()` is designed as a single synchronous, blocking call that contains no Streamlit calls and persists the final state before returning. The working assumption is that a UI rerun therefore cannot leave an experiment half-recorded, and the UI disables the run action while a run is active. This is an **assumption, not an unconditional guarantee**; it depends on Streamlit's script-execution behavior and must be verified by testing in M4. The synchronous service and subprocess architecture is retained as specified. No background controller, worker thread or other additional threading architecture is introduced unless M4 testing demonstrates it is necessary, and any such change requires owner approval and an amendment to this document.
- **Out of MVP:** timeout policy, cancellation, per-iteration progress, retry logic, parallel workers.

## 9. Comparison ownership (F4)

`service.py` is the **single owner** of cross-run comparison logic: `compare_experiments(experiment_ids) -> ComparisonResult`. It uses `storage.py` for records and `metrics.py` for any pure arithmetic, includes only `completed` experiments (listing excluded ones with reasons), and emits comparability warnings when runs differ in `methodology_id`, fixture hashes, model revision, precision/matmul/attention settings, `cpu_threads`, key package versions, `process_isolation` (affects memory comparability) or when a run is `unstable`. `app.py` renders the returned structure and contains no comparison logic.

## 10. Technical trade-offs

Streamlit speeds up local UI delivery at the cost of less granular frontend control. SQLite keeps maintenance negligible for a single user. A subprocess per configuration costs model-load time per run (outside the timed region) and a small amount of orchestration code, but gives clean memory state, crash isolation and a UI that cannot be blocked by CUDA failures; it is far simpler than a queue. Introducing it in M3, after the engine is proven in-process, keeps M1 and M2 independently testable. Keep dependencies and interfaces clear so future GPU optimization, queued jobs or APIs can be added without rewriting engine logic.
