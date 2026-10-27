"""Auditable, deliberately narrow cross-validation boundary checks.

The manifest is an assertion of which observations and labels a model used,
not an inspection of a model pipeline. No third-party dependencies.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

REQUIRED = (
    "fold", "role", "sample_id", "group_id", "feature_at_utc", "label_available_utc"
)


class ManifestError(ValueError):
    """Invalid manifest or audit configuration; not a leakage finding."""


@dataclass(frozen=True)
class Row:
    fold: str
    role: str
    sample_id: str
    group_id: str
    feature_at: datetime
    label_available: datetime


def parse_instant(value: str, line: int, field: str) -> datetime:
    """Accept timezone-aware ISO 8601, normalize to UTC; never infer a timezone."""
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise ManifestError(f"line {line}: {field} must be an ISO 8601 instant") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ManifestError(f"line {line}: {field} needs an explicit timezone offset or Z")
    return stamp.astimezone(timezone.utc)


def read_manifest(path: str | Path) -> list[Row]:
    """Read CSV strictly enough that a malformed manifest cannot silently pass."""
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ManifestError("CSV has no header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ManifestError("CSV has duplicate header names")
            missing = set(REQUIRED) - set(reader.fieldnames)
            if missing:
                raise ManifestError("missing columns: " + ", ".join(sorted(missing)))
            rows: list[Row] = []
            for line, raw in enumerate(reader, start=2):
                if None in raw:
                    raise ManifestError(f"line {line}: more fields than header")
                values = {k: (raw[k] or "").strip() for k in REQUIRED}
                if any(not v for v in values.values()):
                    raise ManifestError(f"line {line}: required fields must be nonempty")
                if values["role"] not in ("train", "test"):
                    raise ManifestError(f"line {line}: role must be train or test")
                feature_at = parse_instant(values["feature_at_utc"], line, "feature_at_utc")
                label_available = parse_instant(values["label_available_utc"], line, "label_available_utc")
                if label_available < feature_at:
                    raise ManifestError(f"line {line}: label_available_utc precedes feature_at_utc")
                rows.append(Row(values["fold"], values["role"], values["sample_id"],
                                values["group_id"], feature_at, label_available))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"cannot read CSV manifest: {exc}") from exc
    if not rows:
        raise ManifestError("CSV contains no observations")
    return rows


def _finding(fold: str, code: str, count: int, examples: list[str], detail: str,
             max_examples: int) -> dict:
    return {"fold": fold, "code": code, "count": count,
            "examples": sorted(examples)[:max_examples], "detail": detail}


def audit(rows: list[Row], *, embargo_hours: float = 0,
          group_policy: str = "disjoint", max_examples: int = 5) -> dict:
    """Check train/test boundaries independently in each fold.

    A training label is considered usable at its label_available_utc instant.
    For a test starting at t, training labels must be available by t-embargo.
    """
    if not math.isfinite(embargo_hours) or embargo_hours < 0 or embargo_hours > 1_000_000:
        raise ManifestError("embargo_hours must be finite, nonnegative and <= 1000000")
    if group_policy not in ("disjoint", "allow"):
        raise ManifestError("group_policy must be disjoint or allow")
    if not isinstance(max_examples, int) or not 1 <= max_examples <= 20:
        raise ManifestError("max_examples must be between 1 and 20")
    if not rows:
        raise ManifestError("manifest has no rows")
    folds: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        folds[row.fold].append(row)
    findings: list[dict] = []
    fold_summary: list[dict] = []
    for fold in sorted(folds):
        entries = folds[fold]
        train = [r for r in entries if r.role == "train"]
        test = [r for r in entries if r.role == "test"]
        if not train or not test:
            raise ManifestError(f"fold {fold!r} must contain train and test observations")
        first_test = min(r.feature_at for r in test)
        cutoff = first_test - timedelta(hours=embargo_hours)
        fold_summary.append({"fold": fold, "train": len(train), "test": len(test),
                             "first_test_utc": first_test.isoformat().replace("+00:00", "Z"),
                             "label_cutoff_utc": cutoff.isoformat().replace("+00:00", "Z")})
        counts = Counter(r.sample_id for r in entries)
        duplicates = [sample for sample, n in counts.items() if n > 1]
        if duplicates:
            findings.append(_finding(fold, "DUPLICATE_SAMPLE", len(duplicates), duplicates,
                                     "Sample identifiers repeat within this fold.", max_examples))
        if group_policy == "disjoint":
            shared = set(r.group_id for r in train) & set(r.group_id for r in test)
            if shared:
                findings.append(_finding(fold, "GROUP_OVERLAP", len(shared), list(shared),
                                         "Train and test share group identifiers.", max_examples))
        future = [r.sample_id for r in train if r.feature_at >= first_test]
        if future:
            findings.append(_finding(fold, "TRAIN_NOT_BEFORE_TEST", len(future), future,
                                     "Train feature timestamps are at/after first test feature timestamp.",
                                     max_examples))
        late = [r.sample_id for r in train if r.label_available > cutoff]
        if late:
            findings.append(_finding(fold, "LABEL_NOT_AVAILABLE", len(late), late,
                                     "Train labels become available after first test minus embargo.",
                                     max_examples))
    findings.sort(key=lambda x: (x["fold"], x["code"]))
    return {"status": "FAIL" if findings else "PASS", "folds": fold_summary,
            "rows": len(rows), "group_policy": group_policy,
            "embargo_hours": embargo_hours, "findings": findings}
