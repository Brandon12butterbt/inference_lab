"""OPT-IN real-model timing smoke test (needs the cached model; CUDA test needs a GPU).

Skipped in the ordinary ``pytest`` run. To run it (cache-only and offline, per
``conftest.py``)::

    $env:INFERENCE_LAB_RUN_MODEL_INTEGRATION = "1"
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_timing_integration.py -v -s

These are exploratory M2.1 runs of the timing component only. They are NOT
finalized protocol results: metrics, the memory profile and the full environment
metadata are not integrated yet, and nothing is stored. Assertions check structure
only, never speed.
"""

import os
import statistics

import pytest

from inference_lab import models, timing
from inference_lab.config import BenchmarkConfig
from inference_lab.manifest import verify_fixture
from inference_lab.parity import ParityStatus, run_parity_check

pytestmark = pytest.mark.skipif(
    os.environ.get("INFERENCE_LAB_RUN_MODEL_INTEGRATION") != "1",
    reason="opt-in: set INFERENCE_LAB_RUN_MODEL_INTEGRATION=1 (cache-only, needs the "
    "cached model; the CUDA test also needs a CUDA GPU)",
)


def summarize(result):
    print(
        f"EXPLORATORY M2.1 timing (not a protocol result): device={result.device} "
        f"batch_size={result.batch_size} mechanism={result.timing_mechanism} "
        f"warmups={result.warmup_forwards} "
        f"trials={len(result.latency_samples_ms)} "
        f"samples/trial={[len(t) for t in result.latency_samples_ms]}"
    )
    print("threads:", result.thread_settings.to_dict())
    print(
        "precision:",
        result.precision_settings.float32_matmul_precision,
        "cuda_matmul_tf32=",
        result.precision_settings.cuda_matmul_allow_tf32,
        "cudnn_tf32=",
        result.precision_settings.cudnn_allow_tf32,
    )
    for i, trial in enumerate(result.latency_samples_ms):
        print(
            f"trial {i}: min={min(trial):.3f} median={statistics.median(trial):.3f} "
            f"max={max(trial):.3f} ms; first 5 raw = "
            f"{[round(s, 3) for s in trial[:5]]}"
        )


def check_structure(result, device):
    assert result.device == device
    assert result.warmup_forwards == 10
    assert len(result.latency_samples_ms) == 3
    assert all(len(t) == 50 for t in result.latency_samples_ms)
    assert all(s > 0 for t in result.latency_samples_ms for s in t)
    assert result.thread_settings.requested == result.thread_settings.effective


def test_real_cpu_timing_smoke_batch_4():
    config = BenchmarkConfig(device="cpu", batch_size=4)
    model, _ = models.load_model(config)
    tokenizer, tokenizer_meta = models.load_tokenizer(config)
    fixture = verify_fixture(tokenizer, tokenizer_meta)  # before any execution

    inputs = timing.prepare_inputs(config, fixture)
    result = timing.run_timing(config, model, inputs)
    summarize(result)
    check_structure(result, "cpu")
    assert result.timing_mechanism == "perf_counter_ns"


def test_real_cuda_timing_smoke_batch_4():
    # Checked here, after the opt-in gate, so ordinary collection never queries CUDA.
    if not models.cuda_available():
        pytest.skip("no CUDA device available on this machine")
    config = BenchmarkConfig(device="cuda:0", batch_size=4)
    model, _ = models.load_model(config)
    tokenizer, tokenizer_meta = models.load_tokenizer(config)
    fixture = verify_fixture(tokenizer, tokenizer_meta)  # before any execution

    parity, _ = run_parity_check(config, model, fixture)  # moves the model to CUDA
    assert parity.status is ParityStatus.PASSED
    print(
        f"parity passed on {parity.compared_device_name}: "
        f"max_abs_diff={parity.max_abs_diff:.3e}"
    )

    inputs = timing.prepare_inputs(config, fixture)
    result = timing.run_timing(config, model, inputs)
    summarize(result)
    check_structure(result, "cuda:0")
    assert result.timing_mechanism == "cuda_synchronized_perf_counter_ns"
