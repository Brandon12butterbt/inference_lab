"""OPT-IN real CPU/CUDA parity test (needs the cached model and a CUDA GPU).

Skipped in the ordinary ``pytest`` run. To run it (cache-only and offline, per
``conftest.py``)::

    $env:INFERENCE_LAB_RUN_MODEL_INTEGRATION = "1"
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_parity_integration.py -v -s

This is a correctness check only; nothing is timed or benchmarked.
"""

import json
import os

import pytest

from inference_lab import models
from inference_lab.config import BenchmarkConfig
from inference_lab.errors import InferenceLabError
from inference_lab.manifest import verify_fixture
from inference_lab.parity import ParityStatus, run_parity_check

pytestmark = pytest.mark.skipif(
    os.environ.get("INFERENCE_LAB_RUN_MODEL_INTEGRATION") != "1",
    reason="opt-in: set INFERENCE_LAB_RUN_MODEL_INTEGRATION=1 (cache-only, needs the "
    "cached model and a CUDA GPU)",
)


def test_real_cpu_cuda_parity_batch_4_position_10():
    # Checked here, after the opt-in gate, so ordinary collection never queries CUDA.
    if not models.cuda_available():
        pytest.skip("no CUDA device available on this machine")
    config = BenchmarkConfig(device="cuda:0", batch_size=4)
    model, model_meta = models.load_model(config)
    tokenizer, tokenizer_meta = models.load_tokenizer(config)
    fixture = verify_fixture(tokenizer, tokenizer_meta)  # manifest-verified inputs

    try:
        result, settings = run_parity_check(config, model, fixture)
    except InferenceLabError as err:
        print("PARITY ERROR:", err)
        print(json.dumps(err.details, indent=2))
        raise
    print("model metadata:", model_meta.to_dict())
    print("effective precision settings:", json.dumps(settings.to_dict(), indent=2))
    print("parity result:", json.dumps(result.to_dict(), indent=2))

    assert result.status is ParityStatus.PASSED
    assert result.labels_match is True
    assert result.num_violations == 0
    assert result.batch_position == 10
    assert result.fixture_indices == [40, 41, 42, 43]
    assert result.reference_labels == result.candidate_labels
    assert settings.float32_matmul_precision == "highest"
    assert settings.cuda_matmul_allow_tf32 is False
    assert settings.cudnn_allow_tf32 is False
    assert {str(p.device) for p in model.parameters()} == {"cuda:0"}
