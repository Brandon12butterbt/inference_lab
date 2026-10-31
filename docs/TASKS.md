# InferenceLab — Implementation Roadmap and Milestones (revision r2)

**Status:** Revised draft (documentation revision r2, 2026-10-31) — **pending final owner approval. Do not start application code until the owner explicitly authorizes the first milestone.**  
**Precedence (benchmarking requirements):** `BENCHMARK_PROTOCOL.md` > `SPEC.md` > `ARCHITECTURE.md` > `TASKS.md`. Open decisions are tracked in `SPEC.md` §8.

## Workflow rules

- One milestone at a time. Before each implementation, summarize the files to change, acceptance criteria and test plan, and request approval.
- Keep pull requests/commits small and reviewable; no unrelated refactoring or dependency upgrades.
- Use **Sonnet 5.5, Medium** for straightforward coding, documentation, tests and UI; **Opus 5.5, High** only for tricky CUDA timing, performance instrumentation, or difficult debugging with concrete justification. These are suggestions, not requirements.
- Report commands actually executed and whether they passed; do not claim unrun tests passed. A skipped test is reported as skipped, never as passed.
- Verify one working baseline before adding optimizations.
- Indicative effort: roughly one week per milestone (M1 ≈ Week 1 … M5 ≈ Week 5). These are planning estimates, not commitments.

### Conditional GPU testing rule (applies to every milestone)

GPU tests are marked `gpu` and skip cleanly when CUDA is unavailable. On the reference machine (RTX 5070 Ti) a milestone's GPU acceptance criteria are met only when the GPU tests were **executed and passed**. If they were skipped, the milestone report must state that GPU acceptance is **not verified**.

## M0 — Documentation revision and baseline verification (Step 7D, NO APP CODE)

**Tasks:**
- Revise `CLAUDE.md`, `docs/SPEC.md`, `docs/ARCHITECTURE.md`, `docs/BENCHMARK_PROTOCOL.md` and `docs/TASKS.md` for internal consistency (revision r2 — done, awaiting review).
- Confirm the existing project, lockfile and working CUDA runtime remain intact; no dependency upgrades; `pyproject.toml` and `uv.lock` unchanged.
- Check all files are in the intended paths and no secret/model assets are staged.

**Acceptance:** all five docs reviewed and **finally approved by the owner**; status headers updated from "pending final owner approval" only on the owner's instruction; no Python application source changes; `git status` understandable.  
**Suggested Claude:** Sonnet 5.5 Medium, Plan/Manual mode.

## M1 — Core contract, fixture, model loading and parity (≈ Week 1)

**Approval gates inside M1 (ask before doing):** first model download and pinning of the revision hash; edits to `pyproject.toml` for pytest markers/`testpaths` and Ruff settings (no dependency changes); the fixture source data and generator, for owner review before manifest hashes are committed.

**Tasks:**
- `errors.py` with the error-code taxonomy; `config.py` typed configuration validation (fixed protocol counts, allowed batch sizes, `cpu_threads`, `matmul_precision`, `attn_implementation`).
- `dataset.py` + `fixtures/`: committed source data, deterministic 512-text generator (seed 42), canonical serialization, manifest with `text_sha256`, `token_ids_sha256`, `attention_mask_sha256`, integrity verification, batch position/indexing.
- `models.py`: load the pinned DistilBERT revision with explicit FP32, `.eval()`, requested attention backend with the resolved backend recorded; verify the exact Transformers 5.22.1 API behavior in the locked environment.
- `parity.py`: untimed CPU/CUDA output comparison, tolerance `atol = rtol = 1e-3` plus label agreement, structured `ParityResult`.
- Import smoke check for the locked Pandas 2.2.3 / NumPy 2.5.4 pairing.
- Unit tests, a CPU smoke test, and conditional (`gpu`) CUDA tests.

**Acceptance:**
- Fixture: exactly 512 distinct texts; `text_sha256` matches the committed manifest without a model; token-ID and attention-mask hashes match with the pinned tokenizer; input shape `(512, 128)`.
- Batch sequencing tests pass (positions, wraparound, identical order).
- CPU forward on a prepared batch returns finite logits of shape `[batch, 2]`; the resolved attention backend equals the requested one or the owner has chosen an alternative.
- Parity function is unit tested with synthetic pass/fail cases. On the reference machine the real CUDA-vs-CPU parity test is executed, passes, and its measured `max_abs_diff` / `max_rel_diff` are reported. If the real difference exceeds the tolerance, stop and report; do not loosen it.
- No benchmark results are invented; `uv run pytest`, `uv run ruff check .` and `uv run ruff format --check .` pass; GPU acceptance follows the conditional GPU rule.  
**Suggested Claude:** Sonnet 5.5 Medium for scaffold; Opus High only if an unexpected model/CUDA compatibility issue needs deep analysis.

## M2 — Measurement engine, metric math, profiling and headless benchmark (≈ Week 2)

**Tasks:**
- `metrics.py` pure functions: P50/P95 (NumPy linear), per-trial medians, summed batch seconds, per-trial and aggregate throughput, spread percentage and stability flag (threshold 10 %).
- `timing.py`: monotonic CPU timing and CUDA-synchronized timing (sync before and after each timed forward); GC handling per protocol.
- `memory.py`: configuration-level memory profile (host RSS fields; CUDA baselines and peaks) in a separate pass.
- `hardware.py`: environment snapshot including thread settings, precision flags, attention backend, power plan, driver version (read-only; `N/A` when unavailable).
- `benchmark.py`: `run_benchmark(config)` implementing protocol steps 1–11 in-process (no subprocess, no database), including controlled `cpu_threads`, matmul precision, parity step for CUDA, sample validation and failure codes; JSON-serializable result.
- **Headless benchmark integration test** (`tests/test_benchmark.py`): CPU run with the full protocol parameters at batch size 1; CUDA equivalent as a conditional `gpu` test.
- Test showing synchronized CUDA timings are not dispatch-only.

**Acceptance:**
- Headless CPU run completes without UI or database and yields exactly 150 finite, strictly positive samples (3 × 50), consistent per-trial/aggregate metrics and a valid JSON round-trip; `process_isolation = "shared_process"` is recorded.
- Deterministic unit tests for all metric math, including boundary cases at the stability threshold.
- Conditional GPU run follows the protocol, records a passed parity result, and the synchronization test passes on the reference machine (executed, not skipped).
- Failure paths produce explicit error codes (CUDA unavailable, invalid measurement, parity failure) and never a successful aggregate.
- Units and sample counts are correct; iteration counts are not alterable for real runs.  
**Suggested Claude:** **Opus 5.5 High** for measurement design/code review, then Sonnet Medium for bounded tests and fixes.

## M3 — Local persistence, subprocess worker and service orchestration (≈ Week 3)

**Tasks:**
- `storage.py`: SQLite schema (`experiments`, `trials`, `latency_samples`, `parity_results`, `memory_profiles`), `PRAGMA foreign_keys`, `PRAGMA user_version`, transactional writes, lifecycle states (`running`, `completed`, `failed`, `interrupted`).
- `worker.py`: subprocess entry point (config JSON in, result/error JSON out, exit code), standard library only; no database access.
- `service.py`: validate config; create the `running` row; launch one worker per configuration sequentially with an in-process lock; persist results as sole DB writer; map failures to error codes; reconcile stale `running` rows to `interrupted` at startup; **own `compare_experiments`** and its comparability warnings.
- Tests with temporary SQLite databases and simulated **unit-test fixtures only** for storage/service logic; one real **subprocess integration test** (CPU, full protocol, batch size 1) and a conditional GPU equivalent.

**Acceptance:**
- Store and reload a run without data loss (raw samples, per-trial medians, parity, memory profile, metadata).
- A subprocess run records `process_isolation = "subprocess"` and persists as `completed`.
- Simulated worker crash or non-zero exit is stored as `failed` / `WORKER_CRASHED`; a leftover `running` row becomes `interrupted` at the next startup; a concurrent start is refused.
- Failed and interrupted runs are distinguishable from completed runs and excluded from comparisons with a stated reason.
- `compare_experiments` returns comparability warnings for differing fixture/methodology/settings and unstable runs.
- No UI or SQL code touches the GPU; the UI never connects directly to SQL.  
**Suggested Claude:** Sonnet 5.5 Medium.

## M4 — Local dashboard MVP (≈ Week 4)

**Tasks:**
- Create `app.py` Streamlit controls for device and batch size (single or multiple) and a run action that calls the service.
- Add result history, comparison table, distinct latency/throughput charts and memory information, rendering only what `service.py` returns.
- Label scopes/units (MiB for memory), show missing CUDA, parity/stability indicators, comparability warnings, and failed/interrupted-run messages clearly; avoid placeholder values presented as measurements.
- Verify rerun safety: the run action is disabled during a run, and interacting with widgets mid-run does not corrupt or abort the recorded experiment.
- Evaluate (with approval) any Streamlit setting needed for torch file-watcher warnings.

**Acceptance:** the user can start a benchmark locally and compare two real saved configurations; no backend, SQL, statistics or comparison logic is embedded in Streamlit widgets; rerun-safety check documented.  
**Suggested Claude:** Sonnet 5.5 Medium.

## M5 — End-to-end validation and portfolio quality (≈ Week 5)

**Tasks:**
- Run tests, Ruff and real CPU/GPU benchmarks via the app; investigate variability; review the stability threshold and default `cpu_threads` against real data and propose any change as an owner-approved protocol revision.
- Validate results serialization, exact metadata, stable fixture hashes, deterministic config and memory labeling.
- Fill README with install/usage/test instructions, research methods, screenshot/demo plan, known limitations.
- Record actual performance results and conditions; avoid inventing benchmarks or speedups.

**Acceptance:** MVP definition of done in `SPEC.md` §7 passed (GPU tests executed and passing on the reference machine), reviewed Git diff and documentation, clean repository, shareable local demo/screenshots.  
**Suggested Claude:** Sonnet Medium for docs/tests, Opus High for a focused independent review of timing methodology if needed.

## Post-MVP candidate features (NOT approved for immediate implementation)

ONNX Runtime or TensorRT baseline comparisons, quantization, TF32/fp16 variants, dynamic batching and queue scheduling, experiment confidence intervals, Pareto frontier and cost modeling, concurrent request load, GPU profiling, CPU thread-count sweeps and affinity pinning, worker timeout/cancellation and progress streaming. Propose separately after MVP evidence.

## At the end of every milestone

- List exact files changed and rationale.
- List exact test commands with pass/fail/skip results (or state not run); apply the conditional GPU rule.
- Call out uncertainty, unverified timing assumptions and any performance claim requiring actual measurements.
- Provide one suggested commit message; **do not push or commit without owner approval**.
