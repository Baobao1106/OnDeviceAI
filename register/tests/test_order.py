import itertools
import unittest

import _paths  # noqa: F401
from register.engine.order import priority_order, split_deadline

N = 8
ALL = list(itertools.combinations(range(N), 2))


def scores():
    return {p: (7 * p[0] + 3 * p[1]) % 11 for p in ALL}


class TestPriorityOrder(unittest.TestCase):
    def test_every_pair_exactly_once(self):
        out = priority_order(N, scores())
        self.assertEqual(sorted(out), ALL)
        self.assertEqual(len(out), len(set(out)))

    def test_strongest_pair_first(self):
        s = scores()
        s[(2, 5)] = 99
        self.assertEqual(priority_order(N, s)[0], (2, 5))

    def test_every_image_gets_its_best_pair_before_any_image_gets_a_second(self):
        s = scores()
        out = priority_order(N, s)
        head = set(out[:N])
        for i in range(N):
            best = max(s[p] for p in ALL if i in p)
            self.assertTrue(any(i in p and s[p] == best for p in head), f"ảnh {i} chưa có cặp tốt nhất trong vòng đầu")

    def test_blind_image_is_still_served_early(self):
        s = {p: (0 if 0 in p else 50) for p in ALL}
        head = priority_order(N, s)[:N]
        self.assertTrue(any(0 in p for p in head))

    def test_is_deterministic(self):
        self.assertEqual(priority_order(N, scores(), seed=3), priority_order(N, scores(), seed=3))

    def test_respects_pair_subset(self):
        subset = [(0, 1), (2, 3), (4, 7)]
        self.assertEqual(sorted(priority_order(N, scores(), subset)), subset)

    def test_missing_scores_count_as_zero_and_pairs_are_kept(self):
        out = priority_order(N, {(1, 2): 5})
        self.assertEqual(out[0], (1, 2))
        self.assertEqual(sorted(out), ALL)

    def test_empty(self):
        self.assertEqual(priority_order(1, {}), [])
        self.assertEqual(priority_order(N, {}, []), [])


class TestSplitDeadline(unittest.TestCase):
    def test_no_deadline_or_no_reserve_keeps_deadline(self):
        self.assertIsNone(split_deadline(100.0, None, 0.2))
        self.assertEqual(split_deadline(100.0, 160.0, 0.0), 160.0)
        self.assertEqual(split_deadline(100.0, 160.0, -1.0), 160.0)

    def test_reserve_takes_that_share_of_the_remaining_time(self):
        self.assertAlmostEqual(split_deadline(100.0, 200.0, 0.2), 180.0)

    def test_reserve_is_capped_and_past_deadline_is_safe(self):
        self.assertAlmostEqual(split_deadline(100.0, 200.0, 5.0), 110.0)
        self.assertEqual(split_deadline(100.0, 50.0, 0.3), 100.0)


if __name__ == "__main__":
    unittest.main()
