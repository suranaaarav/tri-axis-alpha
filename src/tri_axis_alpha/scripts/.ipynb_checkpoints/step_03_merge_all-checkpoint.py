"""
Step 03: Merge quant, fundamentals, text and regime features into one modeling dataset (memory-safe, chunked).
"""

import pandas as pd
from tri_axis_alpha.data.utils import load_config, ensure_dir

CHUNK_SIZE = 2000  # number of gvkeys per batch to keep memory low

def merge_all(cfg_path: str) -> pd.DataFrame:
    print("[MERGE] Loading config...")
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)

    # ---------- Base quant + fundamentals ----------
    print("[MERGE] Loading quant+fundamentals...")
    qf = pd.read_csv(f"{out_dir}/quant_features_full.csv", parse_dates=["ret_eom"])
    try:
        lead = pd.read_csv(f"{out_dir}/lead4_targets_acc.csv", parse_dates=["ret_eom"])
        if {"id", "ret_eom"}.issubset(lead.columns) and "id" in qf.columns:
            qf = qf.merge(lead, on=["id", "ret_eom"], how="left")
            print("[MERGE] Lead fundamentals merged.")
    except FileNotFoundError:
        print("[MERGE] lead4_targets_acc.csv not found — skipping fundamentals.")

    # ---------- Text ----------
    print("[MERGE] Loading text features...")
    try:
        txt = pd.read_csv(f"{out_dir}/text_features_advanced.csv", parse_dates=["ret_eom"])
        print(f"    -> Text features: {txt.shape}")
    except FileNotFoundError:
        txt = pd.DataFrame(columns=["gvkey", "ret_eom"])
        print("[MERGE] No text features found.")

    # ---------- Regime ----------
    print("[MERGE] Loading regime files...")
    try:
        reg_post = pd.read_csv(f"{out_dir}/regime_posteriors.csv", parse_dates=["ret_eom"])
    except FileNotFoundError:
        reg_post = pd.DataFrame(columns=["ret_eom"])
        print("[MERGE] No regime_posteriors.csv.")
    try:
        reg_pred = pd.read_csv(f"{out_dir}/regime_preds.csv", parse_dates=["ret_eom"])
    except FileNotFoundError:
        reg_pred = pd.DataFrame(columns=["ret_eom"])
        print("[MERGE] No regime_preds.csv.")

    out_fp = f"{out_dir}/model_dataset.csv"
    first_write = True

    gvkeys = qf["gvkey"].dropna().unique()
    print(f"[MERGE] Processing {len(gvkeys)} gvkeys in batches of {CHUNK_SIZE}...")

    for i in range(0, len(gvkeys), CHUNK_SIZE):
        batch = gvkeys[i : i + CHUNK_SIZE]
        df = qf[qf["gvkey"].isin(batch)].copy()

        # --- Merge text ---
        if not txt.empty:
            df = df.merge(txt, on=["gvkey", "ret_eom"], how="left")
        # --- Merge regime ---
        if not reg_post.empty:
            df = df.merge(reg_post, on="ret_eom", how="left")
        if not reg_pred.empty:
            df = df.merge(reg_pred, on="ret_eom", how="left")

        # Drop super sparse columns in this batch (optional safeguard)
        na_thresh = 0.99
        keep_cols = [c for c in df.columns if df[c].isna().mean() < na_thresh]
        if len(keep_cols) < len(df.columns):
            dropped = set(df.columns) - set(keep_cols)
            print(f"   -> Dropped {len(dropped)} mostly-empty cols in batch {i//CHUNK_SIZE+1}")
        df = df[keep_cols]

        # Append to CSV
        df.to_csv(out_fp, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False
        print(f"   -> Saved batch {i//CHUNK_SIZE+1} ({len(batch)} gvkeys)")

    print(f"[MERGE] Done -> {out_fp}")
    return None

if __name__ == "__main__":
    merge_all("config.yaml")
