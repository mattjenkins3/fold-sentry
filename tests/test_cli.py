"""CLI exit code and JSON regression tests."""
import contextlib
import io
import json
from pathlib import Path
import unittest

from fold_sentry.cli import main

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_clean_exit_zero(self):
        code, out, err = self.invoke(str(ROOT / "examples/clean.csv"), "--embargo-hours", "2")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "PASS")
        self.assertEqual(err, "")

    def test_leaky_exit_one(self):
        code, out, err = self.invoke(str(ROOT / "examples/leaky.csv"), "--embargo-hours", "2")
        self.assertEqual(code, 1)
        self.assertEqual(len(json.loads(out)["findings"]), 4)
        self.assertEqual(err, "")

    def test_bad_manifest_exit_two(self):
        code, out, err = self.invoke("/missing/manifest.csv")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertEqual(json.loads(err)["status"], "ERROR")

    def test_nan_embargo_exit_two(self):
        code, out, err = self.invoke(str(ROOT / "examples/clean.csv"), "--embargo-hours", "nan")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("embargo", json.loads(err)["error"])

    def test_output_is_repeatable(self):
        args = (str(ROOT / "examples/leaky.csv"), "--embargo-hours", "2")
        self.assertEqual(self.invoke(*args), self.invoke(*args))
