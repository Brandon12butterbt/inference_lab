"""Smoke check of the deliberately pinned Pandas/NumPy pairing (SPEC section 8)."""

import numpy as np
import pandas as pd


def test_pandas_2_2_3_imports_and_works_with_the_locked_numpy():
    assert pd.__version__ == "2.2.3"  # deliberately pinned (Smart App Control)
    assert tuple(int(p) for p in np.__version__.split(".")[:3]) >= (2, 5, 4)

    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    frame = pd.DataFrame(values, columns=["latency_ms", "throughput"])
    frame["group"] = ["a", "a", "b"]
    summary = frame.groupby("group")["latency_ms"].median()
    assert summary.to_dict() == {"a": 2.0, "b": 5.0}
    assert np.percentile(frame["latency_ms"], 95) == np.percentile(values[:, 0], 95)
    assert frame.to_numpy(dtype=object).shape == (3, 3)
