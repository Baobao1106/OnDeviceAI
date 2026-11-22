import time
import unittest

import numpy as np

import _paths  # noqa: F401
import _synth
from solve import bundle, level

Y = np.array([0.0, 1.0, 0.0])


def tilted_scene(seed, tilt_deg, axis_deg=30.0):
    matches, slots, truth = _synth.make_scene(seed=seed)
    a = np.radians(axis_deg)
    ax = np.array([np.cos(a), 0.0, np.sin(a)])
    t = np.radians(tilt_deg)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    T = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
    poses = {n: {"R": (T @ R).tolist(), "registered": True, "source": "bundle"} for n, R in truth["R"].items()}
    return poses, slots, truth, T


def angle(u, v):
    return np.degrees(np.arccos(np.clip(u @ v, -1, 1)))


class TestLevel(unittest.TestCase):
    def test_estimate_up_finds_tilted_up_axis(self):
        for tilt in (0.0, 3.0, 8.0):
            poses, slots, _, T = tilted_scene(20, tilt)
            self.assertLess(angle(level.estimate_up(poses, slots), T @ Y), 1.5, msg=f"tilt {tilt}")

    def test_level_poses_puts_up_on_y_and_reports_tilt(self):
        poses, slots, truth, _ = tilted_scene(21, 6.0)
        new, up, tilt = level.level_poses(poses, slots)
        self.assertAlmostEqual(tilt, 6.0, delta=1.5)
        _, A = _synth.pose_errors(new, truth)
        self.assertLess(_synth.tilt_deg(A), 1.5)
        self.assertAlmostEqual(float(np.linalg.norm(up)), 1.0, places=6)

    def test_level_keeps_poses_as_rotations_and_flags(self):
        poses, slots, _, _ = tilted_scene(22, 4.0)
        poses["img03.jpg"] = {**poses["img03.jpg"], "registered": False, "source": "slot_prior"}
        new, _, _ = level.level_poses(poses, slots)
        self.assertEqual(sorted(new), sorted(poses))
        self.assertFalse(new["img03.jpg"]["registered"])
        self.assertEqual(new["img03.jpg"]["source"], "slot_prior")
        R = np.asarray(new["img00.jpg"]["R"])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_outlier_slot_does_not_move_up_much(self):
        poses, slots, _, T = tilted_scene(23, 5.0)
        bad = {**slots["images"]["img02.jpg"], "pitch_deg": 45.0}
        slots = {"images": {**slots["images"], "img02.jpg": bad}}
        self.assertLess(angle(level.estimate_up(poses, slots), T @ Y), 2.0)

    def test_too_few_registered_images_falls_back_to_y(self):
        poses, slots, _, _ = tilted_scene(24, 5.0)
        poses = {n: {**p, "registered": k < 2, "source": "bundle" if k < 2 else "slot_prior"}
                 for k, (n, p) in enumerate(sorted(poses.items()))}
        np.testing.assert_allclose(level.estimate_up(poses, slots), Y, atol=1e-9)

    def test_after_bundle_whole_chain_is_level(self):
        matches, slots, truth = _synth.make_scene(seed=25)
        poses, _ = bundle.bundle_adjust(matches, slots, 74.0, time.time() + 60)
        new, _, _ = level.level_poses(poses, slots)
        errs, A = _synth.pose_errors(new, truth)
        self.assertLess(_synth.tilt_deg(A), 1.5)
        self.assertLess(max(errs.values()), 0.6)


if __name__ == "__main__":
    unittest.main()
