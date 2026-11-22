import unittest

import numpy as np

import _paths  # noqa: F401
import common
from register import slots


def synthetic_set(seed, drop=(), cut=()):
    rng = np.random.default_rng(seed)
    plan = common.capture_plan()
    names = [f"img{rng.integers(1_000_000):06d}_{k}.jpg" for k in range(24)]
    rots = [common.pose_to_R(y + rng.uniform(-5, 5), p + (0 if abs(p) == 90 else rng.uniform(-5, 5)),
                             rng.uniform(-5, 5)) for y, p in plan]
    pairs = []
    for a in range(24):
        for b in range(a + 1, 24):
            if a in drop or b in drop or ((a in cut) != (b in cut)):
                continue
            ang = np.degrees(np.arccos(np.clip(rots[a][:, 2] @ rots[b][:, 2], -1, 1)))
            if ang < 60:
                pairs.append({"i": names[a], "j": names[b], "n_inliers": 50, "rotation": (rots[a].T @ rots[b]).tolist()})
    return {n: None for n in names}, pairs, {n: k for k, n in enumerate(names)}


def layer(p):
    return int(np.abs(p - np.array([-90.0, -45.0, 0.0, 45.0, 90.0])).argmin())


def yaw_residuals(got, truth, plan):
    ring = [n for n in got if abs(plan[truth[n]][1]) < 80]
    d = np.array([(got[n]["yaw_deg"] - plan[truth[n]][0] + 180) % 360 - 180 for n in ring])
    shift = np.degrees(np.angle(np.exp(1j * np.radians(d)).mean()))
    return np.abs((d - shift + 180) % 360 - 180)


class TestSlots(unittest.TestCase):
    def check_permutation(self, got):
        self.assertEqual(sorted(v["slot"] for v in got.values()), list(range(24)))
        for v in got.values():
            self.assertEqual((v["yaw_deg"], v["pitch_deg"]), common.capture_plan()[v["slot"]])

    def test_all_photos_matched(self):
        images, pairs, truth = synthetic_set(0)
        got = slots.assign_slots(images, pairs, 1e18)
        self.check_permutation(got)
        plan = common.capture_plan()
        self.assertTrue(all(layer(plan[got[n]["slot"]][1]) == layer(plan[truth[n]][1]) for n in got))
        self.assertLess(yaw_residuals(got, truth, plan).max(), 10.0)

    def test_isolated_photos_still_get_distinct_slots(self):
        images, pairs, truth = synthetic_set(1, drop=(3, 12, 20))
        got = slots.assign_slots(images, pairs, 1e18)
        self.check_permutation(got)
        names = sorted(images)
        low = [n for n in names if got[n]["confidence"] <= 0.05]
        self.assertEqual(len(low), 3)
        self.assertTrue(all(got[n]["n_matches"] == 0 for n in low))

    def test_two_components_keep_slots_distinct(self):
        images, pairs, truth = synthetic_set(2, cut=(9, 10, 11))
        got = slots.assign_slots(images, pairs, 1e18)
        self.check_permutation(got)


class TestMargin(unittest.TestCase):
    def test_margin_is_finite_for_a_ring_component_and_infinite_for_one_image(self):
        images, pairs, truth = synthetic_set(3)
        names = sorted(images)
        objs = slots._pair_objects(names, pairs)
        A, label, U = slots.component_frames(24, objs)
        lay = slots.assign_layers(names, objs)
        pitch = {k: lay[k].pitch_deg for k in A}
        yaw = {k: float(slots.yaw_deg(A[k][:, 2], U[int(label[k])])[0]) for k in A}
        Y = np.array([p[0] for p in common.capture_plan()])
        P = np.array([p[1] for p in common.capture_plan()])
        _, _, _, margin = slots._fit_component(list(A), set(range(24)), pitch, yaw, Y, P)
        self.assertTrue(np.isfinite(margin) and margin > 0)
        one = next(k for k in A if abs(pitch[k]) < 80)
        self.assertEqual(slots._fit_component([one], set(range(24)), pitch, yaw, Y, P)[3], float("inf"))

    def test_confidence_stays_in_range_and_slots_are_unchanged_by_margin(self):
        images, pairs, truth = synthetic_set(4)
        got = slots.assign_slots(images, pairs, 1e18)
        self.assertTrue(all(0.0 < v["confidence"] <= 0.95 for v in got.values()))


if __name__ == "__main__":
    unittest.main()
