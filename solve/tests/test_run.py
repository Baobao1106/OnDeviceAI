"""solve.run end to end on synthetic files in a temp work dir."""
import os
import tempfile
import time
import unittest

import _paths  # noqa: F401
import _synth
import common
import solve
from schema import validate_poses


def run_in_tmp(matches, slots):
    with tempfile.TemporaryDirectory() as d:
        common.save_json(matches, os.path.join(d, "matches.json"))
        common.save_json(slots, os.path.join(d, "slots.json"))
        solve.run("unused", d, time.time() + 60)
        return validate_poses(common.load_json(os.path.join(d, "poses.json")))


class TestRun(unittest.TestCase):
    def test_writes_valid_poses_json_close_to_truth(self):
        matches, slots, truth = _synth.make_scene(seed=30)
        out = run_in_tmp(matches, slots)
        self.assertEqual(len(out["images"]), 24)
        errs, A = _synth.pose_errors(out["images"], truth)
        self.assertLess(max(errs.values()), 0.6)
        self.assertLess(_synth.tilt_deg(A), 1.5)
        self.assertAlmostEqual(out["hfov_deg"], 74.0, delta=0.4)

    def test_set_without_any_pair_still_writes_poses(self):
        matches, slots, _ = _synth.make_scene(seed=31)
        out = run_in_tmp({**matches, "pairs": []}, slots)
        self.assertEqual(len(out["images"]), 24)
        self.assertTrue(70.0 <= out["hfov_deg"] <= 79.0)


if __name__ == "__main__":
    unittest.main()
