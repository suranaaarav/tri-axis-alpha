"""Build lead-4M accounting targets and prepare a table to be *predicted* in Stage 1.
We do **not** use true lead ratios as features in the return model (that would be look-ahead).
Instead we save a clean target table that Stage 1 models will try to predict.
"""
import pandas as pd
from .utils import load_config, ensure_dir

def build_lead_targets(cfg_path:str):
    cfg = load_config(cfg_path)
    data_path = cfg["data"]["ret_sample_csv"]
    acc_path  = cfg["data"]["acc_ratios_csv"]
    out_dir   = cfg["output_dir"]
    ensure_dir(out_dir)

    data = pd.read_csv(
        data_path, parse_dates=["date", "ret_eom", "char_date", "char_eom"], low_memory=False
    )
    ratio_list = pd.read_csv(acc_path)["Variable"].tolist()

    keep_cols = ["id", "ret_eom", "char_eom"] + ratio_list
    ratios = data[keep_cols].copy()
    # Move ratios forward by 4 months using char_eom -> ret_eom alignment
    ratios["char_eom"] = ratios["char_eom"] + pd.DateOffset(months=4)
    ratios = ratios.rename(columns={c: f"{c}_lead4" for c in ratio_list})
    ratios = ratios.rename(columns={"char_eom": "ret_eom"})  # aligns with return month

    # Merge back to (id, ret_eom) to define supervised targets
    targets = data[["id", "ret_eom"]].drop_duplicates()
    targets = targets.merge(ratios[["id", "ret_eom"] + [f"{c}_lead4" for c in ratio_list]],
                            on=["id","ret_eom"], how="left")

    targets.to_csv(f"{out_dir}/lead4_targets_acc.csv", index=False)
    return targets
