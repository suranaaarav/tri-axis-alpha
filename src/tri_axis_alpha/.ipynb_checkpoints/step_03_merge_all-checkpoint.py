"""
Step 03: Merge quant, fundamentals, text and regime features into one modeling dataset.
"""
import pandas as pd
from tri_axis_alpha.data.utils import load_config, ensure_dir

def merge_all(cfg_path: str) -> pd.DataFrame:
    print("[MERGE] Loading config...")
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)

    # ---------- Load base quant/fundamental panel ----------
    print("[MERGE] Loading quant+fundamentals...")
    qf = pd.read_csv(f"{out_dir}/quant_features_full.csv", parse_dates=["ret_eom"])
    # Should contain gvkey, ret_eom, engineered quant/fundamental features, and future returns like ret_fwd_1m, ret_fwd_3m

    # ---------- Load text features ----------
    print("[MERGE] Loading text features...")
    try:
        txt = pd.read_csv(f"{out_dir}/text_features_advanced.csv", parse_dates=["ret_eom"])
    except FileNotFoundError:
        print("   !! No text_features_advanced.csv found — continuing without text data.")
        txt = pd.DataFrame(columns=["gvkey","ret_eom"])

    # ---------- Load regime features ----------
    print("[MERGE] Loading regime posteriors + predictions...")
    try:
        reg_post = pd.read_csv(f"{out_dir}/regime_posteriors.csv", parse_dates=["ret_eom"])
        reg_pred = pd.read_csv(f"{out_dir}/regime_preds.csv", parse_dates=["ret_eom"])
    except FileNotFoundError:
        print("   !! No regime files found — continuing without regime data.")
        reg_post = pd.DataFrame(columns=["ret_eom"])
        reg_pred = pd.DataFrame(columns=["ret_eom"])

    # ---------- Merge all ----------
    df = qf.copy()

    if not txt.empty:
        # company-level text join
        df = df.merge(txt, on=["gvkey","ret_eom"], how="left")

    if not reg_post.empty:
        df = df.merge(reg_post, on="ret_eom", how="left")
    if not reg_pred.empty:
        # Align next-month predictions to current month features
        df = df.merge(reg_pred, on="ret_eom", how="left")

    # Drop super-sparse columns (optional safeguard)
    na_thresh = 0.99
    keep_cols = [c for c in df.columns if df[c].isna().mean() < na_thresh]
    dropped = set(df.columns) - set(keep_cols)
    if dropped:
        print(f"[MERGE] Dropping {len(dropped)} mostly-empty cols: {list(dropped)[:5]}...")
    df = df[keep_cols]

    # Save
    out_fp = f"{out_dir}/model_dataset.csv"
    df.to_csv(out_fp, index=False)
    print(f"[MERGE] Saved merged modeling dataset -> {out_fp} | shape={df.shape}")
    return df

if __name__ == "__main__":
    merge_all("config.yaml")
