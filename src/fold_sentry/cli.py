"""Command-line interface with stable exit codes and JSON output."""
from __future__ import annotations

import argparse
import json
import sys
from .audit import ManifestError, audit, read_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit ML train/test boundary manifests")
    parser.add_argument("manifest", help="CSV with fold, role, sample_id, group_id and UTC instants")
    parser.add_argument("--embargo-hours", type=float, default=0.0,
                        help="Minimum time between train label availability and first test (default: 0)")
    parser.add_argument("--group-policy", choices=("disjoint", "allow"), default="disjoint",
                        help="Disjoint is for unseen entities; allow for known-entity forecasting")
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = audit(read_manifest(args.manifest), embargo_hours=args.embargo_hours,
                       group_policy=args.group_policy, max_examples=args.max_examples)
    except ManifestError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1
