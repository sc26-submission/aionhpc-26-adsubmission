#!/usr/bin/env python3
"""Generate LaTeX tables from an evaluation-matrix summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


WORKLOAD_ORDER = ["conversation", "synthetic", "toolagent"]
MODEL_ORDER = ["llama32_3b", "llama31_8b", "qwen3_8b"]
POLICY_ORDER = ["fixed", "static", "adaptive"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "logs" / "evaluation_matrix" / "summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "figures" / "evaluation_matrix",
    )
    return parser.parse_args()


def latex_policy(policy: str) -> str:
    if policy == "adaptive":
        return r"\SYSTEM{}"
    if policy == "static":
        return "Static P/D"
    return "Persistent-Only"


def esc(value: object) -> str:
    return str(value).replace("&", r"\&")


def fmt(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def available_order(df: pd.DataFrame, column: str, preferred: list[str]) -> list[str]:
    present = set(df[column].astype(str))
    ordered = [x for x in preferred if x in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def write_all_policies(df: pd.DataFrame, path: Path) -> None:
    indexed = df.set_index(["workload", "model", "policy"])
    workloads = available_order(df, "workload", WORKLOAD_ORDER)
    models = available_order(df, "model", MODEL_ORDER)

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Trace-driven performance across workloads and model configurations.}",
        r"    \label{tab:eval-matrix-all}",
        r"    \fontsize{8.5pt}{10pt}\selectfont",
        r"    \renewcommand{\arraystretch}{1.08}",
        r"    \begin{tabular}{lllrrrrrr}",
        r"        \toprule",
        r"        Workload & Model & Policy & \shortstack{Complete\\by 1 h (\%)} & \shortstack{p95 TTFT\\(s)} & \shortstack{p95 E2E\\(s)} & \shortstack{p95 prefill\\wait (s)} & \shortstack{p95 decode\\wait (s)} & Retries/1K \\",
        r"        \midrule",
    ]

    first_workload = True
    for workload in workloads:
        workload_rows = df[df["workload"] == workload]
        if workload_rows.empty:
            continue
        if not first_workload:
            lines.append(r"        \midrule")
        first_workload = False
        first_row_in_workload = True

        for model in models:
            if not any(
                (workload_rows["model"] == model)
                & (workload_rows["policy"].isin(POLICY_ORDER))
            ):
                continue
            first_model_row = True
            for policy in POLICY_ORDER:
                key = (workload, model, policy)
                if key not in indexed.index:
                    continue
                row = indexed.loc[key]
                workload_cell = esc(row["workload_label"]) if first_row_in_workload else ""
                model_cell = esc(row["model_label"]) if first_model_row else ""
                lines.append(
                    "        "
                    f"{workload_cell} & {model_cell} & {latex_policy(policy)} & "
                    f"{100*float(row['completion_by_cutoff_fraction']):.1f} & "
                    f"{fmt(row['ttft_p95_s'])} & "
                    f"{fmt(row['e2e_p95_s'])} & "
                    f"{fmt(row.get('prefill_wait_p95_s', float('nan')))} & "
                    f"{fmt(row.get('decode_wait_p95_s', float('nan')))} & "
                    f"{fmt(row.get('retries_per_1000_requests', float('nan')))} \\\\"
                )
                first_row_in_workload = False
                first_model_row = False
            lines.append(r"        \addlinespace[1pt]")

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_static_vs_system(df: pd.DataFrame, path: Path) -> None:
    indexed = df.set_index(["workload", "model", "policy"])
    workloads = available_order(df, "workload", WORKLOAD_ORDER)
    models = available_order(df, "model", MODEL_ORDER)

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \caption{Static P/D and \SYSTEM{} across workloads and model configurations.}",
        r"    \label{tab:eval-matrix}",
        r"    \fontsize{8.5pt}{10pt}\selectfont",
        r"    \renewcommand{\arraystretch}{1.08}",
        r"    \begin{tabular}{llrrrrrr}",
        r"        \toprule",
        r"        & & \multicolumn{2}{c}{p95 TTFT (s)} & \multicolumn{2}{c}{p95 E2E (s)} & \multicolumn{2}{c}{Complete by 1 h (\%)} \\",
        r"        \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"        Workload & Model & Static & \SYSTEM{} & Static & \SYSTEM{} & Static & \SYSTEM{} \\",
        r"        \midrule",
    ]

    first_workload = True
    for workload in workloads:
        subset = df[df["workload"] == workload]
        if subset.empty:
            continue
        if not first_workload:
            lines.append(r"        \midrule")
        first_workload = False
        first_row = True
        for model in models:
            skey = (workload, model, "static")
            akey = (workload, model, "adaptive")
            if skey not in indexed.index or akey not in indexed.index:
                continue
            static = indexed.loc[skey]
            adaptive = indexed.loc[akey]
            workload_cell = esc(static["workload_label"]) if first_row else ""
            lines.append(
                "        "
                f"{workload_cell} & {esc(static['model_label'])} & "
                f"{fmt(static['ttft_p95_s'])} & {fmt(adaptive['ttft_p95_s'])} & "
                f"{fmt(static['e2e_p95_s'])} & {fmt(adaptive['e2e_p95_s'])} & "
                f"{100*float(static['completion_by_cutoff_fraction']):.1f} & "
                f"{100*float(adaptive['completion_by_cutoff_fraction']):.1f} \\\\"
            )
            first_row = False

    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    all_path = args.output_dir / "evaluation_matrix_all_policies.tex"
    compact_path = args.output_dir / "evaluation_matrix_static_vs_system.tex"
    write_all_policies(df, all_path)
    write_static_vs_system(df, compact_path)
    print(f"Wrote {all_path}")
    print(f"Wrote {compact_path}")


if __name__ == "__main__":
    main()
