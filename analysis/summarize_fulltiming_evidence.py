#!/usr/bin/env python3
"""Summarize staged full-timing evidence outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_out", type=Path, nargs="+")
    return parser.parse_args()


def summarize_base(base_out: Path):
    summary = {}
    for stage_dir in sorted(base_out.glob("*p")):
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
    return summary


def aggregate_summaries(summaries):
    aggregate = {}
    for base_name, summary in summaries.items():
        for stage, item in summary.items():
            for key, value in item.items():
                if not key.startswith("delta_logz_") or key.startswith("delta_logzerr_"):
                    continue
                err_key = key.replace("delta_logz_", "delta_logzerr_")
                entry = aggregate.setdefault(stage, {}).setdefault(key, {"values": [], "errors": []})
                entry["values"].append(value)
                if err_key in item:
                    entry["errors"].append(item[err_key])
                entry.setdefault("sources", []).append(base_name)

    for stage_items in aggregate.values():
        for entry in stage_items.values():
            values = entry["values"]
            errors = entry["errors"]
            mean = sum(values) / len(values)
            if len(values) > 1:
                scatter = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            else:
                scatter = 0.0
            entry["n"] = len(values)
            entry["mean"] = mean
            entry["seed_scatter"] = scatter
            if errors:
                entry["rms_numerical_error"] = math.sqrt(sum(error**2 for error in errors) / len(errors))
    return aggregate


def main() -> None:
    args = parse_args()
    summaries = {base_out.name: summarize_base(base_out) for base_out in args.base_out}
    if len(args.base_out) == 1:
        out = args.base_out[0] / "summary.json"
        result = next(iter(summaries.values()))
    else:
        out = args.base_out[0].parent / "fulltiming_combined_summary.json"
        result = {"runs": summaries, "aggregate": aggregate_summaries(summaries)}
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
