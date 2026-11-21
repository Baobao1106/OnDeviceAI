"""bundle_adjust: global poses + FOV from pairwise matches."""
import time
import unittest

import numpy as np

import _paths  # noqa: F401
import _synth
from solve import bundle


def run(seed=7, hfov=74.0, drop=(), start_fov=74.0, **kw):
    matches, slots, truth = _synth.make_scene(seed=seed, hfov=hfov, drop=drop, **kw)
    poses, got = bundle.bundle_adjust(matches, slots, start_fov, time.time() + 60)
    return matches, slots, truth, poses, got


class TestBundle(unittest.TestCase):
    def test_poses_and_fov_match_truth(self):
        _, _, truth, poses, got = run()
        errs, _ = _synth.pose_errors(poses, truth)
        self.assertLess(max(errs.values()), 0.5)
        self.assertLess(float(np.mean(list(errs.values()))), 0.25)
        self.assertAlmostEqual(got, 74.0, delta=0.3)

    def test_fov_is_refined_from_a_wrong_start(self):
        _, _, truth, poses, got = run(seed=8, hfov=75.5, start_fov=72.5)
        self.assertAlmostEqual(got, 75.5, delta=0.4)

    def test_image_without_pairs_gets_slot_prior(self):
        matches, slots, truth, poses, _ = run(seed=9, drop=(5,))
        p = poses["img05.jpg"]
        self.assertFalse(p["registered"])
        self.assertEqual(p["source"], "slot_prior")
        errs, _ = _synth.pose_errors(poses, truth)
        self.assertLess(errs["img05.jpg"], 9.0)                      # jitter is +-5 deg per axis
        self.assertTrue(all(poses[n]["registered"] for n in poses if n != "img05.jpg"))

    def test_all_poses_are_rotations_and_cover_every_image(self):
        matches, _, _, poses, _ = run(seed=10, drop=(3, 11))
        self.assertEqual(sorted(poses), sorted(matches["images"]))
        for p in poses.values():
            R = np.asarray(p["R"])
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)

    def test_a_few_wrong_pairs_do_not_ruin_the_solution(self):
        matches, slots, truth = _synth.make_scene(seed=11)
        rng = np.random.default_rng(0)
        pairs = [dict(p) for p in matches["pairs"]]
        for k in rng.choice(len(pairs), 4, replace=False):
            q = dict(pairs[k])
            q["pts_j"] = rng.uniform([0, 0], matches["image_size"], (len(q["pts_i"]), 2)).tolist()
            q["rotation"] = np.eye(3).tolist()
            q["n_inliers"] = 12
            pairs[k] = q
        poses, _ = bundle.bundle_adjust({**matches, "pairs": pairs}, slots, 74.0, time.time() + 60)
        errs, _ = _synth.pose_errors(poses, truth)
        self.assertLess(float(np.mean(list(errs.values()))), 0.6)

    def test_no_pairs_at_all_uses_slot_priors_only(self):
        matches, slots, _ = _synth.make_scene(seed=12)
        poses, got = bundle.bundle_adjust({**matches, "pairs": []}, slots, 74.0, time.time() + 60)
        self.assertEqual(len(poses), 24)
        self.assertTrue(all(not p["registered"] and p["source"] == "slot_prior" for p in poses.values()))
        self.assertTrue(70.0 <= got <= 79.0)

    def test_expired_deadline_still_returns_valid_poses(self):
        matches, slots, _ = _synth.make_scene(seed=13)
        poses, _ = bundle.bundle_adjust(matches, slots, 74.0, time.time() - 1)
        self.assertEqual(len(poses), 24)

    def test_runs_in_a_few_seconds(self):
        t = time.time()
        run(seed=14)
        self.assertLess(time.time() - t, 15.0)


if __name__ == "__main__":
    unittest.main()
