# InferenceLab — Product Specification (MVP v1.0, revision r2)

**Status:** Revised draft (documentation revision r2, 2026-10-31) — **pending final owner approval. No application code is authorized.**  
**Prepared:** 2026-10-31  
**Related:** `ARCHITECTURE.md`, `BENCHMARK_PROTOCOL.md`, `TASKS.md`, repository-root `CLAUDE.md`  
**Precedence (benchmarking requirements):** `BENCHMARK_PROTOCOL.md` > `SPEC.md` > `ARCHITECTURE.md` > `TASKS.md`. If documents still conflict, stop and ask the owner.

## 1. Purpose and target user

InferenceLab is a **locally running, single-user AI inference benchmarking application**. It lets an ML engineer or student run reproducible benchmarks for a small pretrained model on CPU or GPU, inspect latency, throughput and memory results, and compare batch sizes. It is a portfolio-quality ML systems engineering project, **not** a hosted model-serving system.

### Primary success criterion

A user can start InferenceLab locally, choose CPU or CUDA GPU, select a supported batch size, run a repeatable benchmark, save the results, and compare actual measurements in a dashboard. No fabricated performance values may appear as measured results.

## 2. Target environment and stack

- **OS / hardware:** Windows 11; AMD Ryzen 9 9900X3D (12 cores); 32 GB RAM; NVIDIA GeForce RTX 5070 Ti (16 GB VRAM).
- **Python:** 3.12, using the project's uv-managed `.venv` (not whatever `python` resolves to globally).
- **Libraries:** PyTorch (installed CUDA 12.8 build), Hugging Face Transformers, Streamlit, Pandas, NumPy, psutil, SQLite standard library; `pytest` and `ruff` for development. The benchmark subprocess uses only the standard library (`subprocess`, `json`).
- **Authority for versions:** `uv.lock` is authoritative. At revision r2 it resolves PyTorch 2.11.0+cu128, Transformers **5.22.1**, Streamlit 1.66.1, Pandas **2.2.3**, psutil 7.2.2, NumPy 2.5.4, pytest 9.2.0 and Ruff 0.17.3. Verify against `uv.lock` rather than assuming these remain unchanged; every run records the versions actually used.
- **Windows compatibility note:** A later Pandas release was blocked by Windows Smart App Control on this machine. Do not upgrade Pandas 2.2.3, disable Smart App Control, or change the working PyTorch index/driver without explicit approval.
- **Cost:** Free local operation after installing dependencies and downloading the model; no hosted services, API keys or paid inference APIs are required.

## 3. Initial model and data

- **Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (sentiment classification), pinned to an exact commit hash chosen during M1 with owner approval. The resolved revision is recorded in every run.
- **Precision:** FP32 on CPU and GPU for the baseline, with float32 matmul precision explicitly set to `highest` (TF32 disabled) and the attention backend explicitly requested and recorded. Comparisons are labeled accordingly. See `BENCHMARK_PROTOCOL.md` §2.3.
- **Dataset (`synthetic-sentiment-v1`):** a deterministic, version-controlled fixture of **512 distinct text inputs**, reproducibly generated from a small committed source file of vocabulary/template components using the fixed seed `42`. No private data and no external download. Inputs are tokenized once outside timing, then padded/truncated to **128 tokens** with attention masks.
- **Fixture identity:** a committed manifest records `fixture_id`, seed, generator version, and SHA-256 hashes of the canonical text, the token IDs and the attention masks. The manifest is verified before every run; a mismatch fails the run. Full definition: `BENCHMARK_PROTOCOL.md` §2.1.
- **Important:** The fixture is a controlled *performance fixture*, not a representative accuracy benchmark. Accuracy scoring is out of scope for MVP.
- **Batch sequencing:** for each batch size, a deterministic ordered sequence of prepared batches is drawn from the fixture (batch position `k` takes indices `(k·B + j) mod 512`). Both devices use the same inputs in the same order. Warm-up uses positions 0–9; every trial replays positions 10–59. Cycling is disclosed. See `BENCHMARK_PROTOCOL.md` §2.2.

## 4. The six approved MVP features

### F1. Load one pretrained model
- Load the approved DistilBERT tokenizer/model via Transformers at the pinned revision; first use may download public model assets (with owner permission during development).
- Load explicitly as FP32, set to evaluation mode, and use `torch.inference_mode()` during benchmark forward passes. Request the attention backend explicitly and record the resolved value.
- Support choosing CPU or a CUDA device; if CUDA is requested but unavailable, fail with the actionable `CUDA_UNAVAILABLE` error — never fall back silently.
- **Acceptance:** a successful CPU forward pass returning finite logits of shape `[batch, 2]`; a successful GPU forward pass when CUDA is available; model identifier, pinned revision, dtype, requested and resolved attention backend visible in run metadata.

### F2. Run inference on consistent test inputs
- Use the deterministic 512-input fixture and 128-token padded inputs; pre-tokenize outside the measured region.
- Ensure batch tensors and attention masks are on the selected device **before timing**.
- **Acceptance:**
  - Fixture manifest hashes verified (text, token IDs, attention masks); 512 distinct texts; input shape `(512, 128)`.
  - Identical input indices for CPU/GPU trials with a matching batch size and identical batches replayed in every trial.
  - Output has the expected batch dimension and finite values; no randomization during benchmark loops.
  - **Untimed CPU/GPU parity check** for CUDA configurations: logits compared on the same batch under the documented tolerance (initially `atol = rtol = 1e-3`, identical argmax labels); result recorded; the run fails with `PARITY_FAILED` when validation does not pass.

### F3. Measure latency, throughput and memory
- Follow `docs/BENCHMARK_PROTOCOL.md` exactly (10 warm-ups, 50 measured forwards per trial, 3 trials).
- Measure per-batch execution time (not user-facing end-to-end latency); report P50 and P95 batch latency, per-trial medians, and throughput in inputs/second computed from **summed batch execution durations**.
- Report host-process RSS and, for CUDA runs, PyTorch allocator peak allocated/reserved bytes with clear definitions; memory is one **configuration-level** profile collected in a separate pass. Report `N/A` if a metric cannot be obtained; never substitute estimates.
- Flag instability when the spread between per-trial medians exceeds the documented initial threshold (10 %); never change iteration counts automatically.
- **Acceptance:** every completed configuration has exactly 150 samples (3 × 50), all finite and strictly positive; unit-tested metric calculations (median/P50/P95, throughput, per-trial medians, spread/flag); CUDA timings synchronize correctly and a test demonstrates timings are not dispatch-only; metric labels and units are unambiguous.

### F4. Compare CPU/GPU and batch sizes
- Supported batch sizes: **1, 4, 8, 16, 32**.
- Allow selecting one or multiple supported batch sizes and CPU or CUDA as available. A run is associated with exactly one configuration (device + batch size + other fixed parameters). Multiple selections execute as sequential single-configuration runs.
- **Single comparison owner:** `service.py` produces cross-run comparisons. It includes only `completed` experiments and attaches comparability warnings when runs differ in methodology, fixture, model revision, precision/attention/thread settings, key package versions, process isolation (for memory), or when a run is flagged unstable. The UI only renders what the service returns.
- **Acceptance:** dashboard can compare saved configurations and show batch latency, throughput and memory with appropriate units; charts distinguish processing latency from throughput; comparability warnings are visible.

### F5. Save experiment results locally
- Persist configuration, hardware/software metadata, parity result, per-trial raw samples and statistics, aggregate metrics, the memory profile, run timestamps, lifecycle status and errors to **SQLite** under ignored `data/`.
- Experiments have explicit lifecycle states: `running`, `completed`, `failed`, `interrupted` (`ARCHITECTURE.md` §5).
- Each saved result is traceable to its model/fixture identity and test conditions.
- **Acceptance:** runs survive app restart; retrieved results match stored values exactly; failed and interrupted runs are never mistaken for successful benchmarks; stale `running` rows from a previous application process are marked `interrupted` at startup.

### F6. Display results in an interactive dashboard
- Local Streamlit UI with experiment controls, run status/error feedback, tabular results and at least separate latency and throughput charts.
- Display CPU/GPU and batch-size comparisons and metadata for each run (including parity result and stability flag); allow opening historical run details.
- **Acceptance:** user can initiate a benchmark and compare at least two saved configurations without using command-line application code; all displayed numbers originate in real saved measurements; interacting with widgets during a run does not corrupt or abort the recorded experiment.

## 5. Nonfunctional requirements

- **Accuracy / honesty:** Never fabricate data, benchmarks, or synthetic success outcomes in actual results. Test fixtures must be clearly labeled as fixtures.
- **Reproducibility:** Store model revision, fixture manifest and hashes, seed, dtype, matmul precision, attention backend (requested and resolved), device, batch size, token length, repetitions, CPU thread settings, power plan, timing approach, methodology identifier, package versions, PyTorch CUDA build, driver version and GPU identity when available.
- **Engineering quality:** Layered architecture, type hints where useful, a small explicit error taxonomy, tests for math/storage, GPU smoke test; Ruff formatting and linting.
- **Reliability:** Handle missing model downloads, CUDA unavailability, out-of-memory errors, benchmark-process crashes and database failures with understandable messages and error codes; keep existing successful results intact.
- **Security/privacy:** No secrets in source control. Do not collect personally identifying input data. No telemetry or cloud service required. Respect existing `.gitignore`; do not commit `.venv`, model weights, databases or local data dumps.
- **Scope:** One benchmark at a time, executed sequentially. From M3 each configuration runs in one short-lived local subprocess. No distributed workers, training, multi-user auth, remote hosting, job queue, network APIs, or advanced cancellation in MVP.

## 6. Explicit out-of-scope features

LLM serving, distributed inference, TensorRT, ONNX Runtime, quantization, PyTorch compilation, training, dataset accuracy leaderboards, model comparisons, cloud infrastructure, Pareto-frontier analysis, queued request latency, advanced cancellation, per-iteration progress streaming, CPU thread-count sweeps and affinity pinning, TF32/fp16 comparisons and confidence intervals are later extensions—**not MVP requirements**.

## 7. MVP demonstration and definition of done

1. On the verified Windows system, run the Streamlit app locally.
2. Run DistilBERT FP32 benchmarks at batch sizes 1 and 8 on CPU, and on CUDA if available.
3. View real batch-latency, throughput and memory results with clearly identified measurement scopes, parity result and stability flag.
4. Save and reload the results from SQLite.
5. Compare at least two saved configurations visually.
6. Run automated unit tests, a headless CPU benchmark integration test, a subprocess integration test, GPU tests and Ruff checks successfully. GPU tests are conditional (marked, skipped when CUDA is unavailable), but on the reference machine they must be **executed and passing, not skipped**, to count as GPU acceptance. Document any device-dependent tests or limits.
7. README explains setup, usage, methodology, tested machine, known limitations and a short reproducible example without claiming any unmeasured speedup.

## 8. Unresolved decisions (tracked)

Resolved in revision r2: fixture generation/storage method (§3, Protocol §2.1); CPU RSS and CUDA memory semantics (Protocol §5); subprocess-per-configuration orchestration from M3 (`ARCHITECTURE.md` §1, §8); parity tolerance (initial values); CPU thread policy; stability threshold (initial value).

Still open — each must be resolved explicitly, with owner approval where noted:

1. **Model revision hash** — choose and pin the exact commit during M1 (requires permission for the first model download).
2. **Attention backend** — verify `attn_implementation="sdpa"` is supported by DistilBERT in the locked Transformers 5.22.1; if not, the owner chooses the baseline (e.g. `eager`). Also validate the explicit FP32 loading API against 5.22.1 before writing loader code; do not change installed versions as a first troubleshooting step.
3. **Parity tolerance** — measure the real CPU/CUDA difference in M1 and confirm `atol = rtol = 1e-3`; any loosening needs owner approval.
4. **Stability threshold (10 %)** and **default `cpu_threads = 12`** — review against real data after M2/M5.
5. **Fixture source content** — vocabulary/template data and generator are authored in M1 and reviewed by the owner before the manifest hashes are committed.
6. **Tool configuration** — pytest markers (`gpu`, `slow`, `integration`), `testpaths` and Ruff settings require editing `pyproject.toml`; this needs explicit owner approval in M1 (no dependency changes).
7. **Worker timeout and orphaned workers** — MVP has no timeout or cancellation. If the application is killed, the worker may keep running and the experiment is later marked `interrupted`. Decide in M3 whether a simple wall-clock timeout is wanted.
8. **Sweep grouping** — no sweep/session identifier is included; sequential multi-config runs use a fixed order (Protocol §3). Revisit if comparisons need grouping.
9. **Streamlit with torch** — file-watcher warnings may require a Streamlit server setting (M4, needs approval).
10. **Environment checks** — confirm `uv` is on the PATH of the terminal used for development, and add an import smoke check for the Pandas 2.2.3 / NumPy 2.5.4 pairing in M1 (not verified at planning time).
11. **Fixed constants** (`memory_profile_forwards = 10`, parity batch position 10) are initial choices that may be revisited with the owner.

Keep any necessary amendments small, recorded in the docs, and owner-approved.

## 9. References

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [DistilBERT sentiment model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- [uv project documentation](https://docs.astral.sh/uv/guides/projects/)
