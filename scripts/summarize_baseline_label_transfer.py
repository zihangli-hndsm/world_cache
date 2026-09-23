"""Summarize label-transfer utility for geometry-only correspondence controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mean_iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    labels = np.union1d(ground_truth, prediction)
    scores = []
    for label in labels:
        actual = ground_truth == label
        guessed = prediction == label
        union = np.logical_or(actual, guessed).sum()
        if union:
            scores.append(float(np.logical_and(actual, guessed).sum() / union))
    return float(np.mean(scores)) if scores else float("nan")


def metrics(frame: pd.DataFrame, retained: np.ndarray) -> dict[str, float]:
    selected = frame.loc[retained]
    count = len(selected)
    if count == 0:
        return {
            "queries": int(len(frame)),
            "retained": 0,
            "coverage": 0.0,
            "semantic_accuracy": float("nan"),
            "semantic_mIoU": float("nan"),
            "instance_accuracy": float("nan"),
            "instance_mIoU": float("nan"),
        }
    query_semantics = selected.query_semantic_id.to_numpy()
    predicted_semantics = selected.selected_reference_semantic_id.to_numpy()
    query_instances = selected.query_instance_id.to_numpy()
    predicted_instances = selected.selected_reference_instance_id.to_numpy()
    return {
        "queries": int(len(frame)),
        "retained": int(count),
        "coverage": float(count / len(frame)) if len(frame) else 0.0,
        "semantic_accuracy": float(np.mean(query_semantics == predicted_semantics)),
        "semantic_mIoU": mean_iou(query_semantics, predicted_semantics),
        "instance_accuracy": float(np.mean(query_instances == predicted_instances)),
        "instance_mIoU": mean_iou(query_instances, predicted_instances),
    }


def load_diagnostics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "query_semantic_id",
        "selected_reference_semantic_id",
        "query_instance_id",
        "selected_reference_instance_id",
        "query_evaluable",
        "consensus_inlier",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks diagnostic columns: {sorted(missing)}")
    frame = frame[frame.query_evaluable.astype(bool)].reset_index(drop=True)
    return frame


def collect(root: Path, filename: str, method: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob(f"*/{filename}")):
        pair = path.parent.name
        frame = load_diagnostics(path)
        for selector, retained in (
            ("all_correspondences", np.ones(len(frame), dtype=bool)),
            ("consensus_inlier", frame.consensus_inlier.to_numpy(bool)),
        ):
            rows.append(
                {
                    "method": method,
                    "pair": pair,
                    "dataset": "target" if pair == "target" else "source",
                    "selector": selector,
                    **metrics(frame, retained),
                }
            )
    if not rows:
        raise ValueError(f"no diagnostics found under {root} for {filename}")
    return pd.DataFrame(rows)


def aggregate_exact(root: Path, filename: str, method: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob(f"*/{filename}")):
        pair = path.parent.name
        frame = load_diagnostics(path)
        dataset = "target" if pair == "target" else "source"
        for selector, retained in (
            ("all_correspondences", np.ones(len(frame), dtype=bool)),
            ("consensus_inlier", frame.consensus_inlier.to_numpy(bool)),
        ):
            rows.append({"method": method, "pair": pair, "dataset": dataset, "selector": selector, "frame": frame, "retained_mask": retained})
    summary = []
    for dataset in ("source", "target"):
        for selector in ("all_correspondences", "consensus_inlier"):
            groups = [row for row in rows if row["dataset"] == dataset and row["selector"] == selector]
            all_frame = pd.concat([row["frame"] for row in groups], ignore_index=True)
            retained_mask = np.concatenate([row["retained_mask"] for row in groups])
            base = metrics(all_frame, retained_mask)
            base.update({"method": method, "dataset": dataset, "selector": selector, "pairs": len(groups)})
            summary.append(base)
    return pd.DataFrame(summary)


def latex_table(summary: pd.DataFrame) -> str:
    labels = {
        ("source", "ViT-G consensus"): "Source / ViT-G",
        ("source", "FPFH one-way"): "Source / FPFH",
        ("source", "FCGF top-20"): "Source / FCGF",
        ("target", "ViT-G consensus"): "Target / ViT-G",
        ("target", "FPFH one-way"): "Target / FPFH",
        ("target", "FCGF top-20"): "Target / FCGF",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset / matcher & Cov. & Sem. mIoU & Inst. mIoU & Sem. acc. \\",
        r" & (\%) & (\%) & (\%) & (\%) \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        label = labels[(row.dataset, row.method)]
        lines.append(
            f"{label} & {100 * row.coverage:.1f} & {100 * row.semantic_mIoU:.1f} & "
            f"{100 * row.instance_mIoU:.1f} & {100 * row.semantic_accuracy:.1f} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpfh-root", type=Path, required=True)
    parser.add_argument("--fcgf-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fpfh = aggregate_exact(args.fpfh_root, "query_diagnostics_fpfh_one_way.csv", "FPFH one-way")
    fcgf = aggregate_exact(args.fcgf_root, "query_diagnostics_k20.csv", "FCGF top-20")
    baseline = pd.concat([fpfh, fcgf], ignore_index=True)

    # DINO values are copied from the authoritative transfer artifact so this
    # comparison cannot silently redefine the headline method's denominator.
    dino = pd.DataFrame(
        [
            {"method": "ViT-G consensus", "dataset": "source", "selector": "consensus_inlier", "pairs": 16, "queries": 15048, "retained": 4572, "coverage": 4572 / 15048, "semantic_accuracy": 0.850, "semantic_mIoU": 0.415, "instance_accuracy": 0.850, "instance_mIoU": 0.390},
            {"method": "ViT-G consensus", "dataset": "target", "selector": "consensus_inlier", "pairs": 1, "queries": 1069, "retained": 727, "coverage": 727 / 1069, "semantic_accuracy": 0.935, "semantic_mIoU": 0.855, "instance_accuracy": 0.935, "instance_mIoU": 0.723},
        ]
    )
    table = pd.concat([dino, baseline], ignore_index=True)
    table = table[table.selector == "consensus_inlier"].copy()
    order = {("source", "ViT-G consensus"): 0, ("source", "FPFH one-way"): 1, ("source", "FCGF top-20"): 2, ("target", "ViT-G consensus"): 3, ("target", "FPFH one-way"): 4, ("target", "FCGF top-20"): 5}
    table["order"] = [order[(row.dataset, row.method)] for row in table.itertuples()]
    table = table.sort_values("order").drop(columns="order")
    table.to_csv(args.output_dir / "summary.csv", index=False)
    table.to_csv(args.output_dir / "baseline_transfer_table.csv", index=False)
    (args.output_dir / "baseline_transfer_table.tex").write_text(latex_table(table), encoding="utf-8")
    (args.output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "known_alignment_used_for_selection": False,
                "methods": ["ViT-G consensus", "FPFH one-way", "FCGF top-20"],
                "source_pairs": 16,
                "target_pairs": 1,
                "note": "FPFH uses voxelized points; coverage is therefore not equal-density with ViT-G or FCGF.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
