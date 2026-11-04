"""Offline unit tests for the timing component (fake forwards, clocks and syncs).

No model, network or CUDA is needed. Nothing here asserts absolute speeds.
"""

import gc
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from inference_lab import models, timing
from inference_lab.config import BenchmarkConfig
from inference_lab.dataset import batch_indices
from inference_lab.errors import ErrorCode, InferenceLabError
from inference_lab.precision import PrecisionSettings

FIXTURE_SIZE = 512
SEQ = 128


def make_fixture():
    """Row i is filled with the value i, so batches reveal their fixture indices."""
    ids = torch.arange(FIXTURE_SIZE, dtype=torch.int64).unsqueeze(1).repeat(1, SEQ)
    return SimpleNamespace(input_ids=ids, attention_mask=ids.clone())


def cfg(device="cpu", batch_size=4, cpu_threads=3):
    return BenchmarkConfig(
        device=device, batch_size=batch_size, cpu_threads=cpu_threads
    )


class FakeModel(nn.Module):
    """Records calls into a shared event list; optionally fails on the n-th call."""

    def __init__(self, events, fail_on_call=None, error=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.events = events
        self.fail_on_call = fail_on_call
        self.error = error or RuntimeError("boom")
        self.calls = 0
        self.inference_flags = []
        self.gc_flags = []
        self.seen_rows = []

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        self.events.append("forward")
        self.inference_flags.append(torch.is_inference_mode_enabled())
        self.gc_flags.append(gc.isenabled())
        self.seen_rows.append(input_ids[:, 0].tolist())
        if self.calls == self.fail_on_call:
            raise self.error
        return SimpleNamespace(logits=None)


BASELINE = PrecisionSettings(
    api="legacy",
    float32_matmul_precision="highest",
    cuda_matmul_allow_tf32=False,
    cudnn_allow_tf32=False,
    fp32_precision_readback={},
    torch_version="fake",
)


@pytest.fixture
def events():
    return []


@pytest.fixture
def env(monkeypatch, events):
    """Fake clock (1 ms per reading), sync, transfer, thread API and precision."""
    state = {"now": 0, "threads": 1, "step_ns": 1_000_000}

    def clock():
        events.append("clock")
        state["now"] += state["step_ns"]
        return state["now"]

    monkeypatch.setattr(timing, "_clock", clock)
    monkeypatch.setattr(timing, "_synchronize", lambda d: events.append(f"sync:{d}"))
    monkeypatch.setattr(
        timing, "_to_device", lambda t, d: (events.append(f"transfer:{d}"), t)[1]
    )
    monkeypatch.setattr(models, "require_cuda_available", lambda device="cuda:0": None)
    monkeypatch.setattr(timing, "apply_fp32_baseline", lambda: BASELINE)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: state.update(threads=n))
    monkeypatch.setattr(torch, "get_num_threads", lambda: state["threads"])
    return state


@pytest.fixture(autouse=True)
def fake_model_device(monkeypatch):
    """The fake model lives on the CPU but reports the device it pretends to be on."""
    monkeypatch.setattr(timing, "_model_device", lambda m: m.fake_device)


def new_model(events, device="cpu", **kwargs):
    model = FakeModel(events, **kwargs).eval()
    model.fake_device = device
    return model


# --- input preparation ---------------------------------------------------------------------


def test_prepares_exactly_60_positions_in_order(env):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    assert [b.position for b in inputs.batches] == list(range(60))
    assert inputs.device == "cpu"
    assert inputs.batch_size == 4


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_fixture_indices_match_batch_indices_and_shape(env, batch_size):
    inputs = timing.prepare_inputs(cfg(batch_size=batch_size), make_fixture())
    for b in inputs.batches:
        expected = batch_indices(batch_size, b.position)
        assert b.fixture_indices == expected
        assert b.input_ids.shape == (batch_size, SEQ)
        assert b.attention_mask.shape == (batch_size, SEQ)
        assert b.input_ids[:, 0].tolist() == expected  # content really is those rows


def test_batch_size_32_wraps_around_the_fixture(env):
    inputs = timing.prepare_inputs(cfg(batch_size=32), make_fixture())
    assert inputs.batches[59].fixture_indices[-1] == (59 * 32 + 31) % 512


def test_cpu_and_cuda_preparation_use_identical_indices(env):
    cpu = timing.prepare_inputs(cfg("cpu"), make_fixture())
    cuda = timing.prepare_inputs(cfg("cuda:0"), make_fixture())
    assert [b.fixture_indices for b in cpu.batches] == [
        b.fixture_indices for b in cuda.batches
    ]
    assert cuda.device == "cuda:0"


def test_cuda_preparation_transfers_each_tensor_then_synchronizes(env, events):
    timing.prepare_inputs(cfg("cuda:0"), make_fixture())
    assert events.count("transfer:cuda:0") == 120  # ids + mask for 60 positions
    assert events[-1] == "sync:cuda:0"
    assert events.count("sync:cuda:0") == 1


def test_cpu_preparation_never_synchronizes(env, events):
    timing.prepare_inputs(cfg("cpu"), make_fixture())
    assert not [e for e in events if e.startswith("sync")]


def test_cuda_preparation_without_cuda_raises_and_does_not_fall_back(env, monkeypatch):
    def unavailable(device="cuda:0"):
        raise InferenceLabError(ErrorCode.CUDA_UNAVAILABLE, "no cuda")

    monkeypatch.setattr(models, "require_cuda_available", unavailable)
    with pytest.raises(InferenceLabError) as err:
        timing.prepare_inputs(cfg("cuda:0"), make_fixture())
    assert err.value.code is ErrorCode.CUDA_UNAVAILABLE


def failing_transfer(error, fail_on_call):
    calls = []

    def transfer(tensor, device):
        calls.append(device)
        if len(calls) == fail_on_call:
            raise error
        return tensor

    return transfer, calls


def test_transfer_oom_maps_to_cuda_oom_with_context(env, monkeypatch):
    transfer, calls = failing_transfer(torch.cuda.OutOfMemoryError("oom"), 7)
    monkeypatch.setattr(timing, "_to_device", transfer)
    with pytest.raises(InferenceLabError) as err:
        timing.prepare_inputs(cfg("cuda:0", batch_size=8), make_fixture())
    assert err.value.code is ErrorCode.CUDA_OOM
    assert "oom" in str(err.value)
    # call 7 is the input_ids of the 4th batch (position 3); 3 batches were complete
    assert err.value.details["stage"] == "transferring batch position 3"
    assert err.value.details["batches_prepared"] == 3
    assert err.value.details["device"] == "cuda:0"
    assert err.value.details["batch_size"] == 8
    assert len(calls) == 7  # no retry


def test_transfer_runtime_error_maps_to_cuda_runtime_error(env, monkeypatch):
    transfer, _ = failing_transfer(RuntimeError("illegal memory access"), 2)
    monkeypatch.setattr(timing, "_to_device", transfer)
    with pytest.raises(InferenceLabError) as err:
        timing.prepare_inputs(cfg("cuda:0"), make_fixture())
    assert err.value.code is ErrorCode.CUDA_RUNTIME_ERROR
    assert "illegal memory access" in str(err.value)
    assert err.value.details["stage"] == "transferring batch position 0"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (torch.cuda.OutOfMemoryError("oom"), ErrorCode.CUDA_OOM),
        (RuntimeError("device-side assert"), ErrorCode.CUDA_RUNTIME_ERROR),
    ],
)
def test_final_synchronize_failure_is_translated(env, monkeypatch, error, code):
    def broken_sync(device):
        raise error

    monkeypatch.setattr(timing, "_synchronize", broken_sync)
    with pytest.raises(InferenceLabError) as err:
        timing.prepare_inputs(cfg("cuda:0"), make_fixture())
    assert err.value.code is code
    assert err.value.details["stage"] == "the final synchronization"
    assert err.value.details["batches_prepared"] == 60


def test_cuda_unavailable_is_not_masked_by_the_new_translation(env, monkeypatch):
    def unavailable(device="cuda:0"):
        raise InferenceLabError(ErrorCode.CUDA_UNAVAILABLE, "no cuda")

    def forbidden(*a):
        raise AssertionError("no transfer or sync may be attempted")

    monkeypatch.setattr(models, "require_cuda_available", unavailable)
    monkeypatch.setattr(timing, "_to_device", forbidden)
    monkeypatch.setattr(timing, "_synchronize", forbidden)
    with pytest.raises(InferenceLabError) as err:
        timing.prepare_inputs(cfg("cuda:0"), make_fixture())
    assert err.value.code is ErrorCode.CUDA_UNAVAILABLE


def test_cpu_preparation_errors_are_not_relabelled_as_cuda(env, monkeypatch):
    transfer, _ = failing_transfer(RuntimeError("cpu problem"), 1)
    monkeypatch.setattr(timing, "_to_device", transfer)
    with pytest.raises(RuntimeError, match="cpu problem"):
        timing.prepare_inputs(cfg("cpu"), make_fixture())


def test_rejects_fixture_with_wrong_shape(env):
    bad = SimpleNamespace(
        input_ids=torch.zeros(512, 64, dtype=torch.int64),
        attention_mask=torch.zeros(512, 64, dtype=torch.int64),
    )
    with pytest.raises(InferenceLabError) as err:
        timing.prepare_inputs(cfg(), bad)
    assert err.value.code is ErrorCode.CONFIG_INVALID


# --- warm-up and trial structure -------------------------------------------------------------


def test_all_preparation_happens_before_any_timing(env, events):
    config = cfg("cuda:0")
    inputs = timing.prepare_inputs(config, make_fixture())
    prep_events = len(events)
    model = new_model(events, "cuda:0")
    timing.run_timing(config, model, inputs)
    assert not [e for e in events[:prep_events] if e in ("forward", "clock")]
    assert not [e for e in events[prep_events:] if e.startswith("transfer")]


def test_run_timing_never_transfers_or_slices(env, monkeypatch, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())

    def forbidden(*a, **k):
        raise AssertionError("must not be called during run_timing")

    monkeypatch.setattr(timing, "_to_device", forbidden)
    monkeypatch.setattr(timing, "batch_indices", forbidden)
    timing.run_timing(cfg(), new_model(events), inputs)


def test_exactly_10_warmups_then_150_timed_forwards(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    model = new_model(events)
    result = timing.run_timing(cfg(), model, inputs)
    assert model.calls == 10 + 3 * 50
    assert result.warmup_forwards == 10
    # the first 10 forwards are warm-up positions 0-9 and record no clock readings
    first_clock = events.index("clock")
    assert events[:first_clock].count("forward") == 10
    assert model.seen_rows[:10] == [batch_indices(4, p) for p in range(10)]


def test_50_samples_in_each_of_3_trials(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    result = timing.run_timing(cfg(), new_model(events), inputs)
    assert len(result.latency_samples_ms) == 3
    assert all(len(trial) == 50 for trial in result.latency_samples_ms)


def test_every_trial_replays_positions_10_to_59_in_order(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    model = new_model(events)
    result = timing.run_timing(cfg(), model, inputs)
    expected = [batch_indices(4, p) for p in range(10, 60)]
    timed = model.seen_rows[10:]
    assert timed[0:50] == expected
    assert timed[50:100] == expected
    assert timed[100:150] == expected
    assert result.measured_positions == list(range(10, 60))


def test_nanoseconds_convert_to_milliseconds(env, events):
    env["step_ns"] = 2_500_000  # every clock reading advances 2.5 ms => 2.5 ms/forward
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    result = timing.run_timing(cfg(), new_model(events), inputs)
    assert all(s == 2.5 for trial in result.latency_samples_ms for s in trial)


def test_conversion_is_end_minus_start_divided_by_a_million(env, events, monkeypatch):
    readings = iter(range(0, 10**12, 1))
    values = []

    def clock():
        values.append(next(readings) * 1_234_567)
        return values[-1]

    monkeypatch.setattr(timing, "_clock", clock)
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    result = timing.run_timing(cfg(), new_model(events), inputs)
    assert result.latency_samples_ms[0][0] == (values[1] - values[0]) / 1_000_000


# --- event order -----------------------------------------------------------------------------


def trial_events(events):
    """Events from the first clock reading on (excludes preparation and warm-up)."""
    return events[events.index("clock") :]


def test_cpu_event_order_is_clock_forward_clock(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    timing.run_timing(cfg(), new_model(events), inputs)
    timed = trial_events(events)
    assert timed == ["clock", "forward", "clock"] * 150


def test_cuda_event_order_is_sync_clock_forward_sync_clock(env, events):
    config = cfg("cuda:0")
    inputs = timing.prepare_inputs(config, make_fixture())
    del events[:]
    timing.run_timing(config, new_model(events, "cuda:0"), inputs)
    warmup, rest = events[:11], events[11:]
    assert warmup == ["forward"] * 10 + ["sync:cuda:0"]  # sync after warm-up
    assert rest == ["sync:cuda:0", "clock", "forward", "sync:cuda:0", "clock"] * 150


def test_cpu_path_never_synchronizes_cuda(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    timing.run_timing(cfg(), new_model(events), inputs)
    assert not [e for e in events if e.startswith("sync")]


# --- inference mode and garbage collection ------------------------------------------------------


def test_every_forward_runs_under_inference_mode(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    model = new_model(events)
    timing.run_timing(cfg(), model, inputs)
    assert model.inference_flags == [True] * 160
    assert not torch.is_inference_mode_enabled()  # left again afterwards


def test_gc_is_disabled_for_timed_forwards_and_collected_before_each_trial(
    env, events, monkeypatch
):
    collects = []
    monkeypatch.setattr(gc, "collect", lambda *a: collects.append(len(events)) or 0)
    assert gc.isenabled()
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    model = new_model(events)
    timing.run_timing(cfg(), model, inputs)
    assert len(collects) == 3
    assert model.gc_flags[:10] == [True] * 10  # warm-up is not a trial
    assert model.gc_flags[10:] == [False] * 150
    assert gc.isenabled()
    # each collection happened before its trial's first timed forward
    forwards_before = [events[:n].count("forward") for n in collects]
    assert forwards_before == [10, 60, 110]


def test_gc_left_disabled_if_it_was_disabled_before(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    gc.disable()
    try:
        timing.run_timing(cfg(), new_model(events), inputs)
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_gc_restored_when_a_forward_raises(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    model = new_model(events, fail_on_call=10 + 25)  # mid first trial
    assert gc.isenabled()
    with pytest.raises(InferenceLabError):
        timing.run_timing(cfg(), model, inputs)
    assert gc.isenabled()


def test_gc_stays_disabled_after_failure_if_it_started_disabled(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    model = new_model(events, fail_on_call=20)
    gc.disable()
    try:
        with pytest.raises(InferenceLabError):
            timing.run_timing(cfg(), model, inputs)
        assert not gc.isenabled()
    finally:
        gc.enable()


# --- sample validation --------------------------------------------------------------------------


GOOD = [[1.0] * 50 for _ in range(3)]


def test_valid_samples_pass():
    timing.validate_samples(GOOD, 3, 50)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_rejects_non_positive_or_non_finite_durations(bad):
    samples = [list(t) for t in GOOD]
    samples[1][7] = bad
    with pytest.raises(InferenceLabError) as err:
        timing.validate_samples(samples, 3, 50)
    assert err.value.code is ErrorCode.INVALID_MEASUREMENT


def test_rejects_missing_samples_and_trials():
    for samples in ([t[:49] for t in GOOD], GOOD[:2], [], [[], [], []]):
        with pytest.raises(InferenceLabError) as err:
            timing.validate_samples(samples, 3, 50)
        assert err.value.code is ErrorCode.INVALID_MEASUREMENT


def test_rejects_non_numeric_samples():
    samples = [list(t) for t in GOOD]
    samples[0][0] = None
    with pytest.raises(InferenceLabError) as err:
        timing.validate_samples(samples, 3, 50)
    assert err.value.code is ErrorCode.INVALID_MEASUREMENT


@pytest.mark.parametrize("step_ns", [0, -1_000_000])
def test_run_timing_rejects_zero_or_negative_elapsed_time(env, events, step_ns):
    env["step_ns"] = step_ns
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    with pytest.raises(InferenceLabError) as err:
        timing.run_timing(cfg(), new_model(events), inputs)
    assert err.value.code is ErrorCode.INVALID_MEASUREMENT


# --- failures ---------------------------------------------------------------------------------------


def test_forward_failure_gives_no_partial_result_and_is_not_retried(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    model = new_model(events, fail_on_call=10 + 50 + 3)  # inside trial 2
    with pytest.raises(InferenceLabError) as err:
        timing.run_timing(cfg(), model, inputs)
    assert err.value.code is ErrorCode.INVALID_MEASUREMENT
    assert model.calls == 10 + 50 + 3  # stopped immediately, no retry


def test_cuda_runtime_error_maps_to_cuda_runtime_error(env, events):
    config = cfg("cuda:0")
    inputs = timing.prepare_inputs(config, make_fixture())
    model = new_model(events, "cuda:0", fail_on_call=15)
    with pytest.raises(InferenceLabError) as err:
        timing.run_timing(config, model, inputs)
    assert err.value.code is ErrorCode.CUDA_RUNTIME_ERROR


def test_cuda_out_of_memory_maps_to_cuda_oom(env, events):
    config = cfg("cuda:0")
    inputs = timing.prepare_inputs(config, make_fixture())
    model = new_model(
        events, "cuda:0", fail_on_call=3, error=torch.cuda.OutOfMemoryError("oom")
    )
    with pytest.raises(InferenceLabError) as err:
        timing.run_timing(config, model, inputs)
    assert err.value.code is ErrorCode.CUDA_OOM


def test_failure_during_warmup_is_reported(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    with pytest.raises(InferenceLabError):
        timing.run_timing(cfg(), new_model(events, fail_on_call=4), inputs)


# --- input/model validation ----------------------------------------------------------------------


def test_rejects_model_on_wrong_device_without_switching(env, events):
    config = cfg("cuda:0")
    inputs = timing.prepare_inputs(config, make_fixture())
    model = new_model(events, "cpu")
    with pytest.raises(InferenceLabError) as err:
        timing.run_timing(config, model, inputs)
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert model.calls == 0


def test_rejects_training_mode_model(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    model = new_model(events).train()
    with pytest.raises(InferenceLabError) as err:
        timing.run_timing(cfg(), model, inputs)
    assert err.value.code is ErrorCode.CONFIG_INVALID


def test_rejects_inputs_prepared_for_another_device_or_batch_size(env, events):
    cpu_inputs = timing.prepare_inputs(cfg("cpu"), make_fixture())
    with pytest.raises(InferenceLabError):
        timing.run_timing(cfg("cuda:0"), new_model(events, "cuda:0"), cpu_inputs)
    with pytest.raises(InferenceLabError):
        timing.run_timing(cfg(batch_size=8), new_model(events), cpu_inputs)


def test_rejects_incomplete_prepared_inputs(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    short = timing.PreparedInputs("cpu", 4, inputs.batches[:59])
    with pytest.raises(InferenceLabError) as err:
        timing.run_timing(cfg(), new_model(events), short)
    assert err.value.code is ErrorCode.CONFIG_INVALID


def test_cuda_config_without_cuda_fails_with_cuda_unavailable(env, events, monkeypatch):
    config = cfg("cuda:0")
    inputs = timing.prepare_inputs(config, make_fixture())

    def unavailable(device="cuda:0"):
        raise InferenceLabError(ErrorCode.CUDA_UNAVAILABLE, "no cuda")

    monkeypatch.setattr(models, "require_cuda_available", unavailable)
    with pytest.raises(InferenceLabError) as err:
        timing.run_timing(config, new_model(events, "cuda:0"), inputs)
    assert err.value.code is ErrorCode.CUDA_UNAVAILABLE


# --- CPU threads ------------------------------------------------------------------------------------


def test_thread_count_is_set_and_read_back_without_capping(env):
    settings = timing.apply_cpu_threads(
        cfg(cpu_threads=64)
    )  # beyond any real core count
    assert settings == timing.ThreadSettings(requested=64, effective=64)
    assert env["threads"] == 64


def test_thread_mismatch_fails_with_config_invalid(env, monkeypatch):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    with pytest.raises(InferenceLabError) as err:
        timing.apply_cpu_threads(cfg(cpu_threads=5))
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.details == {"requested": 5, "effective": 2}


def test_thread_set_failure_is_config_invalid(env, monkeypatch):
    def broken(n):
        raise RuntimeError("cannot")

    monkeypatch.setattr(torch, "set_num_threads", broken)
    with pytest.raises(InferenceLabError) as err:
        timing.apply_cpu_threads(cfg())
    assert err.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_threads_applied_before_measured_work_for_cpu_and_cuda(
    env, events, monkeypatch, device
):
    config = cfg(device, cpu_threads=7)
    inputs = timing.prepare_inputs(config, make_fixture())
    monkeypatch.setattr(
        torch,
        "set_num_threads",
        lambda n: (events.append("set_threads"), env.update(threads=n)),
    )
    model = new_model(events, device)
    result = timing.run_timing(config, model, inputs)
    assert events.count("set_threads") == 1
    assert events.index("set_threads") < events.index("forward")
    assert result.thread_settings == timing.ThreadSettings(7, 7)
    assert model.calls == 160


# --- result -------------------------------------------------------------------------------------------


def test_result_is_json_serializable_and_labelled(env, events):
    inputs = timing.prepare_inputs(cfg(), make_fixture())
    result = timing.run_timing(cfg(), new_model(events), inputs)
    data = json.loads(json.dumps(result.to_dict()))
    assert data["timing_mechanism"] == "perf_counter_ns"
    assert data["precision_settings"]["float32_matmul_precision"] == "highest"
    assert data["thread_settings"] == {"requested": 3, "effective": 3}
    assert data["warmup_positions"] == list(range(10))
    assert len(data["latency_samples_ms"]) == 3


def test_cuda_result_names_the_synchronized_mechanism(env, events):
    config = cfg("cuda:0")
    inputs = timing.prepare_inputs(config, make_fixture())
    result = timing.run_timing(config, new_model(events, "cuda:0"), inputs)
    assert result.timing_mechanism == "cuda_synchronized_perf_counter_ns"
    assert result.device == "cuda:0"
