import sys
import unittest
from pathlib import Path

import numpy as np


DL_ROOT = Path(__file__).resolve().parents[1] / "Deeplearning"
if str(DL_ROOT) not in sys.path:
    sys.path.insert(0, str(DL_ROOT))

from device_timebase import infer_device_timebase  # noqa: E402


class DeviceTimebaseTests(unittest.TestCase):
    def test_second_timestamps_keep_seconds(self):
        fs, scale, unit = infer_device_timebase(np.arange(1000) / 200.0, 200.0)
        self.assertAlmostEqual(fs, 200.0)
        self.assertEqual(scale, 1.0)
        self.assertEqual(unit, "seconds")

    def test_millisecond_timestamps_are_rescaled(self):
        fs, scale, unit = infer_device_timebase(np.arange(1000) * 5.0, 200.0)
        self.assertAlmostEqual(fs, 200.0)
        self.assertEqual(scale, 1e-3)
        self.assertEqual(unit, "milliseconds")

    def test_implausible_timebase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "采样率异常"):
            infer_device_timebase(np.arange(100), 200.0)


if __name__ == "__main__":
    unittest.main()
