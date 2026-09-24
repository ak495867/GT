"""
Unit tests for Streaming Gabriel Transform (real-time stream ingestion).
"""

import pytest
import numpy as np
from gabriel_transform.streaming import StreamingGabrielTransform


class TestStreamingGabrielTransform:
    def test_single_and_batch_append(self):
        stream = StreamingGabrielTransform(q=0.5)

        # Append 16 items
        for i in range(16):
            merges = stream.append(float(i))
            assert merges >= 0

        assert stream.total_samples == 16
        # At exactly 16 = 2^4 samples, slot 4 should be occupied, lower slots None
        assert stream.slots[4] is not None
        assert stream.slots[4].size == 16

    def test_query_recent(self):
        stream = StreamingGabrielTransform(q=0.5)
        vals = [10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]
        stream.append_batch(vals)

        # Query recent 4 items (should be mean of [20, 20, 20, 20] = 20.0)
        mean_val, blocks = stream.query_recent(window_size=4)
        assert mean_val == pytest.approx(20.0)
        assert blocks > 0

    def test_horn_summary(self):
        stream = StreamingGabrielTransform(q=0.5)
        stream.append_batch(np.linspace(0, 100, 64))

        summary = stream.get_compressed_horn_summary(epsilon=1e-2)
        assert summary["total_samples"] == 64
        assert summary["compression_ratio"] >= 1.0
        assert len(summary["active_horn_blocks"]) > 0
