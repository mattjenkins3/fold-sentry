"""Behavioral tests for honest split-manifest audits."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from fold_sentry import ManifestError, audit, read_manifest
from fold_sentry.audit import parse_instant

ROOT = Path(__file__).resolve().parents[1]
HEADER = ("fold", "role", "sample_id", "group_id", "feature_at_utc", "label_available_utc")


def record(role, sample, group, feature, label=None, fold="f"):
    return (fold, role, sample, group, feature, label or feature)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "manifest.csv"

    def manifest(self, records, header=HEADER):
        with self.path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(records)
        return read_manifest(self.path)

    def clean(self):
        return [record("train", "a", "x", "2026-01-01T00:00:00Z"),
                record("test", "b", "y", "2026-01-03T00:00:00Z")]

    def codes(self, rows, **opts):
        return {x["code"] for x in audit(self.manifest(rows), **opts)["findings"]}

    def test_clean_pass(self):
        result = audit(self.manifest(self.clean()))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["folds"][0]["first_test_utc"], "2026-01-03T00:00:00Z")

    def test_example_clean_passes_with_embargo(self):
        self.assertEqual(audit(read_manifest(ROOT / "examples/clean.csv"), embargo_hours=2)["status"], "PASS")

    def test_example_leaky_reports_all_four(self):
        result = audit(read_manifest(ROOT / "examples/leaky.csv"), embargo_hours=2)
        self.assertEqual({f["code"] for f in result["findings"]},
                         {"DUPLICATE_SAMPLE", "GROUP_OVERLAP", "LABEL_NOT_AVAILABLE", "TRAIN_NOT_BEFORE_TEST"})

    def test_group_overlap_detected(self):
        rows = self.clean()
        rows[1] = record("test", "b", "x", "2026-01-03T00:00:00Z")
        self.assertEqual(self.codes(rows), {"GROUP_OVERLAP"})

    def test_group_policy_allow_suppresses_only_overlap(self):
        rows = [record("train", "a", "x", "2026-01-01T00:00:00Z", "2026-01-04T00:00:00Z"),
                record("test", "b", "x", "2026-01-03T00:00:00Z")]
        self.assertEqual(self.codes(rows, group_policy="allow"), {"LABEL_NOT_AVAILABLE"})

    def test_duplicate_within_role(self):
        rows = self.clean() + [record("train", "a", "z", "2026-01-01T00:00:00Z")]
        self.assertEqual(self.codes(rows), {"DUPLICATE_SAMPLE"})

    def test_duplicate_across_roles(self):
        rows = self.clean()
        rows[1] = record("test", "a", "y", "2026-01-03T00:00:00Z")
        self.assertEqual(self.codes(rows), {"DUPLICATE_SAMPLE"})

    def test_same_sample_in_distinct_folds_allowed(self):
        rows = self.clean() + [record("train", "a", "x", "2026-01-01T00:00:00Z", fold="g"),
                               record("test", "b", "y", "2026-01-03T00:00:00Z", fold="g")]
        self.assertEqual(self.codes(rows), set())

    def test_train_at_first_test_is_flagged(self):
        rows = [record("train", "a", "x", "2026-01-03T00:00:00Z"),
                record("test", "b", "y", "2026-01-03T00:00:00Z")]
        self.assertEqual(self.codes(rows), {"TRAIN_NOT_BEFORE_TEST"})

    def test_train_after_first_test_is_flagged(self):
        rows = [record("train", "a", "x", "2026-01-04T00:00:00Z"),
                record("test", "b", "y", "2026-01-03T00:00:00Z")]
        self.assertEqual(self.codes(rows), {"TRAIN_NOT_BEFORE_TEST", "LABEL_NOT_AVAILABLE"})

    def test_label_exactly_at_cutoff_allowed(self):
        rows = [record("train", "a", "x", "2026-01-01T00:00:00Z", "2026-01-02T22:00:00Z"),
                record("test", "b", "y", "2026-01-03T00:00:00Z")]
        self.assertEqual(self.codes(rows, embargo_hours=2), set())

    def test_label_one_second_after_cutoff_flagged(self):
        rows = [record("train", "a", "x", "2026-01-01T00:00:00Z", "2026-01-02T22:00:01Z"),
                record("test", "b", "y", "2026-01-03T00:00:00Z")]
        self.assertEqual(self.codes(rows, embargo_hours=2), {"LABEL_NOT_AVAILABLE"})

    def test_offset_normalized_to_utc(self):
        stamp = parse_instant("2026-01-01T00:00:00-07:00", 2, "t")
        self.assertEqual(stamp, datetime(2026, 1, 1, 7, tzinfo=timezone.utc))

    def test_naive_timestamp_rejected(self):
        rows = [record("train", "a", "x", "2026-01-01T00:00:00"), self.clean()[1]]
        with self.assertRaisesRegex(ManifestError, "explicit timezone"):
            self.manifest(rows)

    def test_invalid_timestamp_rejected(self):
        rows = [record("train", "a", "x", "tomorrow"), self.clean()[1]]
        with self.assertRaisesRegex(ManifestError, "ISO 8601"):
            self.manifest(rows)

    def test_label_before_prediction_rejected(self):
        rows = [record("train", "a", "x", "2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"), self.clean()[1]]
        with self.assertRaisesRegex(ManifestError, "precedes"):
            self.manifest(rows)

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ManifestError, "missing columns"):
            self.manifest([("f", "train", "a", "x", "2026-01-01T00:00:00Z")], HEADER[:-1])

    def test_duplicate_header_rejected(self):
        with self.assertRaisesRegex(ManifestError, "duplicate header"):
            self.manifest([], HEADER + ("fold",))

    def test_invalid_role_rejected(self):
        rows = [record("validate", "a", "x", "2026-01-01T00:00:00Z")]
        with self.assertRaisesRegex(ManifestError, "role must"):
            self.manifest(rows)

    def test_missing_value_rejected(self):
        rows = [record("train", "", "x", "2026-01-01T00:00:00Z")]
        with self.assertRaisesRegex(ManifestError, "nonempty"):
            self.manifest(rows)

    def test_extra_columns_ignored(self):
        rows = [r + ("note",) for r in self.clean()]
        self.assertEqual(audit(self.manifest(rows, HEADER + ("note",)))["status"], "PASS")

    def test_extra_values_rejected(self):
        rows = [r + ("surprise",) for r in self.clean()]
        with self.assertRaisesRegex(ManifestError, "more fields"):
            self.manifest(rows)

    def test_empty_csv_rejected(self):
        with self.assertRaisesRegex(ManifestError, "no observations"):
            self.manifest([])

    def test_missing_test_role_rejected(self):
        rows = [record("train", "a", "x", "2026-01-01T00:00:00Z")]
        with self.assertRaisesRegex(ManifestError, "train and test"):
            audit(self.manifest(rows))

    def test_missing_train_role_rejected(self):
        rows = [record("test", "a", "x", "2026-01-01T00:00:00Z")]
        with self.assertRaisesRegex(ManifestError, "train and test"):
            audit(self.manifest(rows))

    def test_negative_embargo_rejected(self):
        with self.assertRaisesRegex(ManifestError, "embargo_hours"):
            audit(self.manifest(self.clean()), embargo_hours=-1)

    def test_nonfinite_embargo_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ManifestError):
                audit(self.manifest(self.clean()), embargo_hours=value)

    def test_invalid_group_policy_rejected(self):
        with self.assertRaises(ManifestError):
            audit(self.manifest(self.clean()), group_policy="magic")

    def test_example_limit_applies(self):
        rows = [record("train", "x" + str(i), "same", "2026-01-01T00:00:00Z") for i in range(8)]
        rows.append(record("test", "t", "same", "2026-01-03T00:00:00Z"))
        result = audit(self.manifest(rows), max_examples=1)
        self.assertEqual(next(f for f in result["findings"] if f["code"] == "GROUP_OVERLAP")["examples"], ["same"])

    def test_max_examples_rejected(self):
        for n in (0, 21, -1):
            with self.subTest(n=n), self.assertRaises(ManifestError):
                audit(self.manifest(self.clean()), max_examples=n)

    def test_deterministic_finding_order(self):
        result = audit(read_manifest(ROOT / "examples/leaky.csv"), embargo_hours=2)
        self.assertEqual([f["code"] for f in result["findings"]], sorted(f["code"] for f in result["findings"]))

    def test_bom_csv_supported(self):
        self.path.write_text("\ufeff" + ",".join(HEADER) + "\n" +
                             "\n".join(",".join(r) for r in self.clean()) + "\n", encoding="utf-8")
        self.assertEqual(audit(read_manifest(self.path))["status"], "PASS")

    def test_nonexistent_file_rejected(self):
        with self.assertRaisesRegex(ManifestError, "cannot read"):
            read_manifest(self.path / "missing")
