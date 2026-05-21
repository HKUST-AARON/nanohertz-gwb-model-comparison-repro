#!/usr/bin/env python3
"""Summarize staged full-timing evidence outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {}
    for stage_dir in sorted(args.base_out.glob("*p")):
        stage = stage_dir.name
        results = {}
        for evidence_path in sorted(stage_dir.glob("*/evidence.json")):
            model = evidence_path.parent.name
            results[model] = json.loads(evidence_path.read_text())
        if not results:
            continue
        item = {"results": results}
        if "hd_powerlaw" in results:
            base = results["hd_powerlaw"]
            for model, result in results.items():
                if model == "hd_powerlaw":
                    continue
                delta = result["logz"] - base["logz"]
                delta_err = math.sqrt(result["logzerr"] ** 2 + base["logzerr"] ** 2)
                item[f"delta_logz_{model}_minus_hd_powerlaw"] = delta
                item[f"delta_logzerr_{model}_minus_hd_powerlaw"] = delta_err
        summary[stage] = item
    out = args.base_out / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
