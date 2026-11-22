"""Tìm trọng số trong thư mục dự án và lỗi một lần khi thiếu (không cần mô hình hay mạng)."""
import os
import tempfile
import unittest
from unittest import mock

import _paths  # noqa: F401
import register.engine as engine
from register.engine import weights


def make_weights(directory):
    os.makedirs(os.path.join(directory, "checkpoints"))
    for name in weights.WEIGHTS:
        open(os.path.join(directory, "checkpoints", name), "wb").close()


class TestWeights(unittest.TestCase):
    def test_project_dir_is_used_and_hub_dir_is_restored(self):
        import torch
        before = torch.hub.get_dir()
        with tempfile.TemporaryDirectory() as own, mock.patch.dict(os.environ, {weights.ENV_VAR: own}):
            make_weights(own)
            with weights.use() as used:
                self.assertEqual(used, own)
                self.assertEqual(torch.hub.get_dir(), own)
            self.assertEqual(torch.hub.get_dir(), before)

    def test_missing_weights_raise_a_clear_error_without_network(self):
        with tempfile.TemporaryDirectory() as own, tempfile.TemporaryDirectory() as hub, \
                mock.patch.dict(os.environ, {weights.ENV_VAR: own}):
            import torch
            before = torch.hub.get_dir()
            torch.hub.set_dir(hub)
            try:
                with self.assertRaises(weights.WeightsMissing) as ctx:
                    with weights.use():
                        pass
                self.assertIn("fetch_weights", str(ctx.exception))
            finally:
                torch.hub.set_dir(before)

    def test_engine_failure_is_remembered_and_not_retried(self):
        calls = []

        class Boom:
            def __init__(self):
                calls.append(1)
                raise RuntimeError("không tải được")

        saved = (engine._ENGINE, engine._ERROR)
        engine._ENGINE = engine._ERROR = None
        try:
            with mock.patch("register.engine.pairs.PairEngine", Boom):
                for _ in range(3):
                    with self.assertRaises(RuntimeError):
                        engine.get_engine()
            self.assertEqual(len(calls), 1)             # lần sau ném lại lỗi cũ, không tạo lại mô hình
        finally:
            engine._ENGINE, engine._ERROR = saved

    def _get_engine_with_fake_pairs(self, env=None, weights_missing=False):
        """Gọi get_engine với PairEngine giả (ghi lại config) và trạng thái module đã đặt lại."""
        import contextlib
        import io
        built = []

        class Fake:
            def __init__(self, config=None):
                built.append(config)

        saved = (engine._ENGINE, engine._ERROR, engine._MODE)
        engine._ENGINE = engine._ERROR = engine._MODE = None
        err = io.StringIO()
        try:
            patches = [mock.patch("register.engine.pairs.PairEngine", Fake), mock.patch.dict(os.environ, env or {})]
            if weights_missing:
                patches.append(mock.patch.object(weights, "use", side_effect=weights.WeightsMissing("thiếu test")))
            with contextlib.ExitStack() as stack, contextlib.redirect_stderr(err):
                for p in patches:
                    stack.enter_context(p)
                eng = engine.get_engine()
                again = engine.get_engine()
            return eng, again, built, engine.engine_mode(), err.getvalue()
        finally:
            engine._ENGINE, engine._ERROR, engine._MODE = saved

    def test_missing_weights_fall_back_to_sift_only_once(self):
        eng, again, built, mode, err = self._get_engine_with_fake_pairs(weights_missing=True)
        self.assertEqual(mode, "sift-only")
        self.assertIs(eng, again)
        self.assertEqual(len(built), 1)                              # chỉ tạo một lần, không thử lại
        self.assertFalse(built[0].use_superpoint)
        self.assertTrue(built[0].use_sift)
        self.assertIn("dự phòng", err)

    def test_env_forces_sift_only_without_touching_weights(self):
        eng, _, built, mode, err = self._get_engine_with_fake_pairs(env={engine.SIFT_ONLY_ENV: "1"})
        self.assertEqual(mode, "sift-only")
        self.assertFalse(built[0].use_superpoint)

    def test_full_engine_when_weights_present(self):
        with tempfile.TemporaryDirectory() as own:
            make_weights(own)
            eng, _, built, mode, err = self._get_engine_with_fake_pairs(env={weights.ENV_VAR: own})
        self.assertEqual(mode, "full")
        self.assertEqual(built, [None])                              # PairEngine() mặc định: SuperPoint + RootSIFT
        self.assertEqual(err, "")

    def test_pinned_sizes_and_hashes_are_well_formed(self):
        for name, (size, sha, url) in weights.WEIGHTS.items():
            self.assertGreater(size, 1_000_000)
            self.assertEqual(len(sha), 64)
            self.assertTrue(url.startswith("https://github.com/cvg/LightGlue/releases/"))


if __name__ == "__main__":
    unittest.main()
