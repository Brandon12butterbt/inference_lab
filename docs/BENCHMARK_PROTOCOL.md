# InferenceLab — Benchmark Protocol v0.2

**Status:** Revision r2, approved by the owner. Implementation is authorized incrementally by work package (see `TASKS.md` and `CLAUDE.md`).  
**Methodology identifier:** `inferencelab-protocol-v0.2` (stored with every experiment).  
**Purpose:** Define controlled steady-state **model-execution** benchmarking on CPU vs NVIDIA CUDA GPU.  
**Precedence:** For benchmarking requirements this document outranks `SPEC.md`, which outranks `ARCHITECTURE.md`, which outranks `TASKS.md`. Remaining conflicts must be raised with the owner, not resolved silently (see `CLAUDE.md`).  
**Revision note:** No measurements exist under v0.1. v0.2 keeps the core methodology (10 warm-ups, 50 measured forwards per trial, 3 trials) and makes batch sequencing, threading, precision, attention backend, parity validation, memory semantics and stability assessment explicit.

## 1. Questions and limits

1. How do CPU and GPU **batch execution times** and throughput differ under matched model/input/precision conditions?
2. How does batch size (1, 4, 8, 16, 32) affect batch latency, throughput and measured memory?

This is **not** a hosted service or end-to-end inference benchmark. It excludes request queues, network, input tokenization, data transfers to device and user-visible wait time. A smaller batch-processing latency does not necessarily mean lower P95 service response latency.

## 2. Locked baseline workload

- **Model:** `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, pinned to an exact 40-character commit hash (`model_revision`). The hash is chosen during M1 with owner approval (it requires the first model download) and then written into the configuration defaults; the resolved revision is recorded with every run.
- **Task:** English sentiment classification; model in `.eval()`; forward passes under `torch.inference_mode()`.
- **Precision:** FP32 weights and activations on both devices. No fp16/bf16, quantization or compilation in the MVP. See §2.3 for the controlled matmul and attention settings.
- **Input:** the versioned synthetic fixture in §2.1 — 512 examples, tokenized once, padded/truncated to exactly 128 tokens with attention masks. No dynamic or random model inputs in timed iterations.
- **Devices:** CPU (Ryzen 9 9900X3D) vs CUDA GPU `cuda:0` (RTX 5070 Ti) with equivalent inputs and configuration.
- **Batch sizes:** 1, 4, 8, 16, 32.
- **Warm-up:** 10 full forwards per configuration, not counted.
- **Timed samples:** 50 forwards per trial × 3 trials = **150 batch timings per configuration**.
- **Model setup:** download, load, initialization, transfer of the model to the target device, tokenization, input batch assembly and transfer of inputs to the device all occur **outside** the timed section.

Iteration counts are fixed. If 150 timings prove insufficient for stable results, the instability flag (§6) reports it; any change to the counts requires an owner-approved protocol revision and a new methodology identifier. The application never changes counts automatically or silently.

### 2.1 Fixture: `synthetic-sentiment-v1`

The fixture is a deterministic, version-controlled performance fixture. It is not an accuracy benchmark and contains no private or personal data.

- **Source data:** a small committed source file of vocabulary and sentence-template components (subjects, aspects, sentiment phrases, connectors) inside the package (`src/inference_lab/fixtures/`). No external dataset or download is used. The source file is authored in M1 and reviewed by the owner.
- **Generation:** a documented, self-contained generator builds exactly **512 distinct** texts from the source data using the fixed **seed `42`**. The generator must not depend on unspecified behavior of a library RNG across versions; the committed hashes below detect any drift. Texts vary from one to several sentences so that real token lengths differ.
- **Ordering:** list order is the fixture index order (0–511) and is part of the fixture identity.
- **Canonical text form:** the 512 strings, Unicode-normalized (NFC), serialized as a compact JSON array (`ensure_ascii=False`, separators `(",", ":")`, no trailing newline), encoded as UTF-8.
- **Tokenization:** `tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")` using the tokenizer from the same pinned model revision.
- **Hashes (SHA-256), committed in a fixture manifest in the package:**
  - `text_sha256` — canonical text bytes.
  - `token_ids_sha256` — `input_ids` as int64, little-endian, C-order, shape `(512, 128)`.
  - `attention_mask_sha256` — `attention_mask` encoded the same way.
- **Manifest** (`fixture_id`, `seed`, generator version, the three hashes, tokenizer/model revision, sequence length, real-token-length min/median/max, truncated count) is verified before every run. A mismatch is a failure (`FIXTURE_INTEGRITY_FAILED`), never a warning. Changing the source data, generator or seed requires a new `fixture_id` (e.g. `synthetic-sentiment-v2`); results from different fixtures are never silently comparable.
- **Limitation to disclose:** because every input is padded to 128 tokens, compute cost depends on shape, not on text content. Real-token-length statistics are recorded so this is visible.

### 2.2 Deterministic batch sequencing and warm-up

Let `B` be the batch size and `k` a zero-based **batch position**. Batch `k` contains fixture indices `(k·B + j) mod 512` for `j = 0 … B−1`, in that order. The same positions apply to every device.

- **Warm-up:** batch positions `0 … 9` (10 forwards), same device, same code path, same batch size.
- **Measured batches:** batch positions `10 … 59` (50 batches).
- **Every trial replays the identical 50 measured batches in the identical order.** Trial-to-trial differences therefore reflect system variability, not input differences.
- **Memory-profile pass:** batch positions `10 … 19` (see §5).
- **Parity check:** batch position `10` (see §3, step 5).
- All 60 batches (positions 0–59) are assembled and placed on the target device before warm-up; nothing is tokenized, sliced, copied or transferred during timing.
- **Cycling disclosure:** distinct fixture inputs per trial are `min(512, 50·B)`. The fixture wraps for `B ≥ 11`. Record `distinct_inputs_per_trial`.
- No random shuffling. No pause or cache clearing between trials.

### 2.3 Controlled execution settings

These are set explicitly by the benchmark (not left to defaults) and recorded with every run.

- **CPU intra-op threads:** `cpu_threads`, whose **default is hardware-aware**: when a configuration is created without an explicit value, the default resolves to the number of physical CPU cores reported by `psutil` (falling back to the logical CPU count, then to 1 if detection is unavailable). The resolved value is a concrete integer stored in the configuration and saved with every experiment; a saved configuration reloads with exactly the stored value on any machine and is never re-derived. An explicitly provided value must be a positive integer and is never capped or altered based on local hardware (hardware suitability checks and warnings belong to the benchmark engine, M2). On the reference Ryzen 9 9900X3D the default resolves to 12. It is applied with `torch.set_num_threads(cpu_threads)` before any measured work, and applied for CUDA runs too (host-side threads). Record requested and effective `torch.get_num_threads()`; they must match. Record `torch.get_num_interop_threads()` and the values of `OMP_NUM_THREADS` / `MKL_NUM_THREADS` if present. Inter-op parallelism is recorded, not controlled, in the MVP. The MVP does not pin CPU affinity; record `psutil.Process().cpu_affinity()`. Thread-count sensitivity is a future extension.
- **Float32 matmul and cuDNN TF32 control:** the FP32 baseline requires that both the matrix-multiplication TF32 setting and the cuDNN TF32 setting are explicitly controlled and verified, using APIs that are compatible with the installed PyTorch version (2.11.0+cu128 per `uv.lock`). The intended controls are `torch.set_float32_matmul_precision("highest")`, `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32`, and their effective values are read back and recorded. The exact APIs, including any newer or deprecated TF32-control interfaces in this PyTorch version, are **verified during M1** and the protocol is amended if they differ. TF32 must not be active for the baseline; results are labeled "FP32 (TF32 disabled)". If the effective settings cannot be set and confirmed, the run fails (`CONFIG_INVALID`).
- **Attention backend:** request `attn_implementation="sdpa"` when loading the model, and record both the requested value and the resolved value (`model.config._attn_implementation`). If they differ, the run fails (`ATTENTION_BACKEND_MISMATCH`) — no silent substitution. Also record PyTorch's SDPA kernel enable flags. CPU and CUDA use the same requested backend, but PyTorch may dispatch different kernels internally; this is disclosed, not hidden. Whether `sdpa` is supported by DistilBERT in the locked Transformers 5.22.1 must be verified in M1; if not, the owner decides the baseline (for example `eager`) before M1 completes.
- **Hardware and power context** (best-effort, `N/A` if unavailable, never guessed): CPU model, logical/physical core counts; RAM total; GPU name, compute capability, total VRAM, NVIDIA driver version (read-only `nvidia-smi` query if present), `torch.version.cuda`, cuDNN version; Windows active power plan (read-only `powercfg /getactivescheme`) and AC/battery status. The application never changes power, driver, clock or security settings. The preferred condition is an idle system on AC power; deviations are the user's responsibility to document.

## 3. Measurement procedure

For each configuration (model, revision, fixture, device, batch size, precision, token length):

1. Capture environment metadata (§2.3) and verify the requested device. If `cuda:0` is requested and unavailable, fail with `CUDA_UNAVAILABLE`; never fall back to CPU.
2. Build the fixture and verify it against the manifest (`FIXTURE_INTEGRITY_FAILED` on mismatch).
3. Load the pinned model on CPU in FP32, `.eval()`, with the §2.3 settings applied.
4. Assemble batch positions 0–59 for the configured batch size.
5. **Parity check (CUDA configurations only; untimed; before warm-up):**
   - Compute reference logits on CPU for batch position 10 using the already-loaded model, then move the model to `cuda:0` and compute logits for the same batch.
   - **Criteria (initial values):** all values finite; `|cuda − cpu| ≤ atol + rtol·|cpu|` elementwise with `atol = 1e-3`, `rtol = 1e-3`; and the argmax label is identical for every row.
   - Record `status` (`passed` / `failed` / `not_applicable`), `atol`, `rtol`, `max_abs_diff`, `max_rel_diff`, `labels_match`, the batch position and the devices.
   - On failure, the run fails with `PARITY_FAILED`, no timed measurements are collected, and the failure is persisted as `failed`.
   - For CPU configurations parity is `not_applicable` (CPU is the reference).
   - The tolerance is an initial value to be validated empirically in M1 (see `SPEC.md` §8). It is never loosened automatically; loosening requires owner approval and a protocol revision.
6. Execute the 10 warm-up forwards (positions 0–9); synchronize the GPU after warm-up for CUDA runs.
7. For each of 3 trials, run the 50 measured batches (positions 10–59) in order. Before each trial, run `gc.collect()` (outside timing) and disable the Python cyclic garbage collector for the duration of the trial, re-enabling it afterwards.
   - **CPU:** read a monotonic high-resolution timer (`time.perf_counter_ns`) immediately before and after each forward.
   - **CUDA:** call `torch.cuda.synchronize(device)` before the start timestamp, run the forward, call `torch.cuda.synchronize(device)`, then read the end timestamp. Default timing is synchronized host wall clock. CUDA events are a permitted alternative only with a separately documented timing scope and a distinct methodology identifier.
   - Append **per-batch elapsed milliseconds** to the trial's samples. Nothing but the forward and the synchronization required for correct completion is inside the timed region.
   - Do not call `.item()`, move outputs to the host, or otherwise read outputs during timed execution.
8. Validate the samples: exactly 50 per trial, all finite and strictly positive. Otherwise fail with `INVALID_MEASUREMENT`.
9. Compute per-trial and aggregate metrics (§4) and the stability assessment (§6).
10. Run the **separate memory-profiling pass** (§5), after all timed trials.
11. Return the complete result; the caller persists it (M3 onward) as `completed`, or as `failed` with an error code. Failed or partial configurations are never reported as successful.

### Fair comparisons

- Compare CPU and GPU for a fixed batch size, precision, sequence length, model revision and input order. For different batch sizes the number of inputs processed per trial necessarily differs because the number of iterations is fixed. **Never claim the total workloads are identical across batch sizes.**
- The fixture is cycled for sustained throughput; report this (`distinct_inputs_per_trial`). Results benefit from warmed caches and do not predict first-request behavior.
- When running several configurations in one session the service uses a fixed, documented order (CPU configurations first, then CUDA; batch sizes ascending) and executes them sequentially.
- Run on an otherwise idle system where practical. Repeats on separate days are needed for stronger claims later.

## 4. Metric definitions

Field names below are the canonical names used in storage and code.

| Metric / field | Unit | Definition |
|---|---|---|
| `latency_samples_ms` | ms/batch | Synchronized elapsed time of one forward pass at the configured batch size; 50 per trial |
| `sum_batch_execution_seconds` (per trial) | s | Sum of that trial's 50 batch latencies, divided by 1000. **Not** loop wall-clock time and not including gaps between forwards |
| `total_processed_inputs` (per trial) | inputs | `batch_size × 50` |
| `throughput_inputs_per_sec` (per trial) | inputs/s | `total_processed_inputs / sum_batch_execution_seconds` |
| `trial_median_ms` | ms/batch | Median of that trial's 50 samples |
| `p50_batch_latency_ms` | ms/batch | 50th percentile of all 150 samples |
| `p95_batch_latency_ms` | ms/batch | 95th percentile of all 150 samples |
| `min_batch_latency_ms`, `max_batch_latency_ms` | ms/batch | Extremes of all 150 samples |
| `aggregate_throughput_inputs_per_sec` | inputs/s | Total inputs across 3 trials divided by the sum of the three `sum_batch_execution_seconds` (never an average of rounded rates) |
| `trial_median_spread_pct` | % | See §6 |

Percentiles use NumPy's default linear interpolation and are unit tested. Storage keeps raw numeric values (bytes, ms, seconds); formatting happens only at the UI boundary. The UI displays memory in **MiB (2^20 bytes)** and labels the unit; bytes are stored. Throughput excludes queueing, preprocessing and transfers.

## 5. Memory profile (configuration-level)

There is **one memory profile per experiment**, not per trial. It is collected in a separate pass after all timed trials so that no sampling overhead can influence latency samples. The pass runs 10 untimed forwards over batch positions 10–19; no background sampling thread is used in the MVP.

**Host (CPU-side) process memory — recorded for every run, labeled "host process":**
- `host_rss_baseline_bytes` — `psutil.Process().memory_info().rss` after model placement and batch assembly, before warm-up.
- `host_rss_after_profile_bytes` — RSS after the profiling pass.
- `host_rss_delta_bytes` — after minus baseline, stored as measured (may be zero).
- `host_peak_working_set_bytes` — Windows `peak_wset`: the process-lifetime high-water mark (it includes imports, loading, warm-up and timing); `N/A` where unavailable.
- `process_isolation` — `subprocess` (M3 onward, a fresh process per configuration) or `shared_process` (M1/M2 headless runs and tests).

RSS is the resident set of the benchmark process: Python, PyTorch libraries, tokenizer, weights and allocator caches. It is **not** "model RAM" and not system-wide memory. Absolute RSS values are comparable between runs only when both are `subprocess`; the UI labels `shared_process` values as non-comparable.

**CUDA memory — recorded for CUDA runs only:**
- Immediately before the pass: synchronize, `torch.cuda.empty_cache()`, `torch.cuda.reset_peak_memory_stats(device)`, then record `cuda_allocated_baseline_bytes` and `cuda_reserved_baseline_bytes`. The baseline includes model weights and the prepared device batches.
- After the pass (synchronized): `cuda_peak_allocated_bytes` (`max_memory_allocated`) and `cuda_peak_reserved_bytes` (`max_memory_reserved`), plus the derived above-baseline values. Both absolute and above-baseline values are stored and labeled.
- These are PyTorch allocator statistics for this process. They are not total VRAM use, exclude the CUDA context and other processes, and must not be described as such.

`N/A` is acceptable when a value cannot be obtained; estimates are never substituted.

## 6. Stability assessment

- Per-trial medians (`trial_median_ms`) are stored for all three trials.
- `trial_median_spread_pct = (max(trial_medians) − min(trial_medians)) / median(trial_medians) × 100`.
- **Initial threshold:** `STABILITY_SPREAD_THRESHOLD_PCT = 10.0`. A spread above the threshold sets `stability_flag = "unstable"`; otherwise `"stable"`.
- The threshold is an unvalidated starting point. It is reviewed against real data after M2 and M5; changing it requires an owner-approved protocol revision.
- An unstable run is still stored with status `completed`; the flag is a visible warning in the UI and comparisons. The application does not rerun, extend, trim or discard samples automatically, and never changes iteration counts.

## 7. Experiment lifecycle and failures

Lifecycle states (`running`, `completed`, `failed`, `interrupted`) and error codes are defined in `ARCHITECTURE.md` §5–§6. Protocol-level rules:

- Only `completed` experiments contain reportable measurements.
- Any failure listed in §3 (CUDA unavailable, fixture integrity, parity, attention mismatch, invalid samples, OOM, CUDA error, worker crash) produces a `failed` experiment with an error code and message, never a partial success.
- A run abandoned by closing the application is `interrupted`, not `completed`.

## 8. Data quality safeguards

- Store all raw batch latencies, not merely rounded summaries.
- Use actual timed measurements only; UI example numbers must be marked demonstrations and excluded from stored results.
- Document sample count, time units, warm-up count and repetitions next to results.
- GPU availability must be proven with a real computation (the parity check and timed forwards); never infer benchmark completion solely from `torch.cuda.is_available()`.
- Detect NaN/non-positive durations, empty samples, CUDA failure, parity failure and interrupted runs; produce an explicit failure, never a successful aggregate.
- Hardware/environment metadata includes everything in §2.3 plus: Python and OS versions; PyTorch, Transformers, NumPy, Pandas, psutil and tokenizers versions; selected device; seed; fixture manifest values and hashes; model revision; dtype; batch size; token length; thread settings; timing mechanism; methodology identifier; timestamps.
- Limitations: no concurrency or load generator, no service-level queue metrics, no accuracy test, no power-draw or energy measurement, no CPU affinity control, GPU clock and thermal state not controlled (Windows WDDM; the GPU may also drive the display), and no generalization claim to other workloads.

## 9. Initial validation tests

- Known synthetic **test timing samples** (clearly fixtures): median/P50/P95, per-trial and aggregate throughput, per-trial medians, spread and stability flag return mathematically correct values, including boundary cases at the threshold.
- Per-trial and aggregate throughput agree with raw totals; `sum_batch_execution_seconds` equals the sum of samples.
- Batch sequencing: positions, wraparound and identical order across trials and devices.
- Fixture: 512 distinct texts; `text_sha256` matches the manifest without any model; `token_ids_sha256` and `attention_mask_sha256` match with the pinned tokenizer; shape `(512, 128)`.
- Parity: pass and fail cases with synthetic tensors (fixtures); real CUDA-vs-CPU parity test on the reference machine.
- CPU-only runs with CUDA absent; CUDA-specific tests skip safely when CUDA is unavailable. On the reference machine, GPU tests must be **executed and passing, not skipped**, for GPU acceptance.
- CUDA test verifies a real forward computation and demonstrates that synchronized timings are not dispatch-only (for example, synchronized time of a heavy operation is materially larger than unsynchronized dispatch time).
- Headless end-to-end benchmark run on CPU with the full protocol parameters (batch size 1) in M2, plus a CUDA equivalent when available.
- Database round-trip of units, sample arrays, lifecycle states and metadata.
- UI labels batch latency correctly and does not conflate it with end-to-end latency.

## 10. Future extensions (separate versioned protocols)

End-to-end latency including preprocessing/transfers, TTFT and tokens/sec for generative LLMs, concurrent load/queueing, dynamic batching, energy/VRAM telemetry, quantization, TF32/fp16 comparisons, CPU thread-count sensitivity and affinity pinning, confidence intervals, and model comparisons. Each requires a new protocol rather than silently redefining v0.2 measurements.

## 11. References

- [PyTorch CUDA asynchronous execution and timing](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch CUDA memory APIs](https://docs.pytorch.org/docs/stable/cuda.html)
