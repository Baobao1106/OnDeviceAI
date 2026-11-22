"""estimate_fov: scan of the shared horizontal FOV."""
import unittest

import _paths  # noqa: F401
import _synth
from solve import fov


class TestEstimateFov(unittest.TestCase):
    def test_recovers_fov_within_half_degree(self):
        for seed, true in ((2, 72.5), (3, 74.0), (4, 76.8)):
            matches, slots, _ = _synth.make_scene(seed=seed, hfov=true)
            got = fov.estimate_fov(matches, slots)
            self.assertAlmostEqual(got, true, delta=0.5, msg=f"seed {seed}")

    def test_no_pairs_returns_middle_of_range(self):
        matches, slots, _ = _synth.make_scene(seed=5)
        matches = {**matches, "pairs": []}
        self.assertAlmostEqual(fov.estimate_fov(matches, slots), 74.5, delta=1.0)

    def test_result_stays_inside_search_range(self):
        matches, slots, _ = _synth.make_scene(seed=6, hfov=79.0)
        got = fov.estimate_fov(matches, slots, lo=70.0, hi=79.0)
        self.assertTrue(70.0 <= got <= 79.0)


if __name__ == "__main__":
    unittest.main()
