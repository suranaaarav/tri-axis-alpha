"""
Build lead-4M accounting targets to be predicted in Stage 1 (no leakage).

What it does
------------
- Reads: ret_sample.csv (only needed cols) and acc_ratios.csv (list of ratios)
- Shifts each ratio forward by 4 months using char_eom -> ret_eom alignment
- Renames to *_lead4 and merges to (id, ret_eom) monthly grid
- Writes: outputs/lead4_targets_acc.csv
"""
from __future__ import annotations

import pandas as pd
from typing import List
from .utils import load_config, ensure_dir


def _read_ratio_list(acc_path: str) -> List[str]:
    ratios = pd.read_csv(acc_path)
    col = None
    # accept either 'Variable' (as provided) or 'variable'
    for candidate in ["Variable", "variable"]:
        if candidate in ratios.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(
            "acc_ratios.csv must contain a column named 'Variable' (or 'variable')."
        )
    return ratios[col].astype(str).tolist()


def build_lead_targets(cfg_path: str) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    data_path = cfg["data"]["ret_sample_csv"]
    acc_path = cfg["data"]["acc_ratios_csv"]
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    ratio_list = _read_ratio_list(acc_path)

    # We only read the minimal columns for memory safety
    base_cols = ["id", "ret_eom", "char_eom"]
    usecols = list(dict.fromkeys(base_cols + ratio_list))  # preserve order & dedupe

    # Parse dates if present; tolerate missing columns gracefully
    parse_dates = [c for c in ["ret_eom", "char_eom"] if c in usecols]

    data = pd.read_csv(
        data_path,
        usecols=lambda c: (c in usecols),  # only necessary columns
        parse_dates=parse_dates,
        low_memory=True,
    )

    # Basic sanity checks
    for col in base_cols:
        if col not in data.columns:
            raise ValueError(
                f"Required column '{col}' not found in ret_sample.csv. Present: {list(data.columns)}"
            )
    missing_ratios = [r for r in ratio_list if r not in data.columns]
    if missing_ratios:
        # Keep going but warn; those columns will be absent in output
        print(
            f"[lead_fundamentals] Warning: {len(missing_ratios)} ratio columns missing in ret_sample.csv "
            f"(e.g., {missing_ratios[:5]} ...). They will be skipped."
        )
        ratio_list = [r for r in ratio_list if r in data.columns]

    # Move ratios forward by 4 months:
    # char_eom at t becomes a target aligned to ret_eom at t+4 months
    ratios = data[["id", "char_eom"] + ratio_list].copy()
    ratios["char_eom"] = ratios["char_eom"] + pd.DateOffset(months=4)
    # Rename for clarity and to avoid accidental use as contemporaneous features
    rename_map = {c: f"{c}_lead4" for c in ratio_list}
    ratios = ratios.rename(columns=rename_map)
    ratios = ratios.rename(columns={"char_eom": "ret_eom"})

    # Define the (id, ret_eom) grid we’ll supervise on
    # Use unique combinations from the main file to avoid exploding rows
    id_date = data[["id", "ret_eom"]].drop_duplicates()

    # Merge the shifted targets back
    targets = id_date.merge(
        ratios[["id", "ret_eom"] + list(rename_map.values())],
        on=["id", "ret_eom"],
        how="left",
    )

    # Persist
    out_fp = f"{out_dir}/lead4_targets_acc.csv"
    targets.to_csv(out_fp, index=False)
    print(f"[lead_fundamentals] Saved lead targets -> {out_fp} | shape={targets.shape}")

    return targets


if __name__ == "__main__":
    # Allow standalone execution for quick tests
    build_lead_targets("config.yaml")
