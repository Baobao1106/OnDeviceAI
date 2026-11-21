"""register.run must always write valid matches.json / slots.json when a step fails or time runs out."""
import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import _paths  # noqa: F401
import common
import register
from register import features, match, slots, textureless
from schema import validate_matches, validate_slots

NAMES = [f"{k:04x}.jpg" for k in range(24)]
IMAGES = {n: np.zeros((648, 864, 3), np.uint8) for n in NAMES}


def fake_pairs():
    return [{"i": NAMES[0], "j": NAMES[1], "pts_i": [[1.0, 2.0]], "pts_j": [[3.0, 4.0]], "n_inliers": 12}]


def run_register(deadline_in=240.0, **patches):
    """Run register.run with patched steps; returns (matches, slots, stderr text)."""
    stack = contextlib.ExitStack()
    with stack, tempfile.TemporaryDirectory() as work, contextlib.redirect_stderr(io.StringIO()) as err:
        stack.enter_context(mock.patch.object(features, "load_images", return_value=IMAGES))
        for target, kw in patches.items():
            mod, attr = target.split("__")
            stack.enter_context(mock.patch.object({"features": features, "match": match, "slots": slots,
                                                   "textureless": textureless}[mod], attr, **kw))
        register.run("unused", work, time.time() + deadline_in)
        m = validate_matches(json.load(open(os.path.join(work, "matches.json"))))
        s = validate_slots(json.load(open(os.path.join(work, "slots.json"))))
    return m, s, err.getvalue()


def good(**extra):
    base = {"features__extract": {"return_value": (np.zeros((0, 2), np.float32), None)},
            "match__match_all": {"return_value": fake_pairs()}}
    base.update(extra)
    return base


class TestResilience(unittest.TestCase):
    def check_slots(self, s):
        self.assertEqual(sorted(v["slot"] for v in s["images"].values()), list(range(24)))

    def test_slots_step_fails_falls_back_to_file_order(self):
        m, s, err = run_register(**good(**{"slots__assign_slots": {"side_effect": RuntimeError("boom")}}))
        self.check_slots(s)
        self.assertEqual(s["images"][NAMES[5]]["slot"], 5)
        self.assertTrue(all(v["confidence"] == 0.0 for v in s["images"].values()))
        self.assertIn("boom", err)
        self.assertEqual(len(m["pairs"]), 1)                       # matching result is kept

    def test_match_step_fails_gives_no_pairs_but_valid_slots(self):
        m, s, err = run_register(**good(**{"match__match_all": {"side_effect": MemoryError("out of memory")}}))
        self.assertEqual(m["pairs"], [])
        self.check_slots(s)

    def test_one_image_fails_to_extract(self):
        calls = {"n": 0}

        def flaky(img):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("cuda error")
            return np.zeros((0, 2), np.float32), None

        seen = {}

        def capture(images, feats, deadline):
            seen["feats"] = len(feats)
            return []

        m, s, err = run_register(**{"features__extract": {"side_effect": flaky},
                                    "match__match_all": {"side_effect": capture}})
        self.assertEqual(seen["feats"], 23)                         # the failing image is treated as isolated
        self.check_slots(s)

    def test_deadline_already_passed_stops_extraction(self):
        extract = mock.Mock(return_value=(np.zeros((0, 2), np.float32), None))
        m, s, err = run_register(deadline_in=-5.0, **{"features__extract": {"new": extract},
                                                      "match__match_all": {"return_value": []}})
        extract.assert_not_called()
        self.assertIn("hết giờ", err)
        self.check_slots(s)

    def test_invalid_slots_from_assign_slots_are_replaced(self):
        dup = {n: {"slot": 0, "yaw_deg": 0.0, "pitch_deg": 0.0, "confidence": 1.0, "n_matches": 0} for n in NAMES}
        m, s, err = run_register(**good(**{"slots__assign_slots": {"return_value": dup}}))
        self.check_slots(s)


if __name__ == "__main__":
    unittest.main()
