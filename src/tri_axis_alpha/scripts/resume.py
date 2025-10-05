"""
resume_text_features_after_crash.py
Resumes after yearly files were processed and cached in tmp_text/
"""

import os, glob, pandas as pd
from sklearn.decomposition import PCA
from tri_axis_alpha.data.utils import load_config, ensure_dir

PCA_DIM = 12

def resume(cfg_path="config.yaml"):
    print("[RESUME] Loading cached yearly files...")
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)
    tmp_dir = os.path.join(out_dir, "tmp_text")

    files = sorted(glob.glob(os.path.join(tmp_dir, "text_*.feather")))
    if not files:
        raise FileNotFoundError(f"No cached yearly files found in {tmp_dir}")

    dfs = [pd.read_feather(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[RESUME] Concatenated {len(all_df):,} rows from {len(files)} yearly files.")

    # ---- Fix the crash line ----
    all_df["ret_eom"] = pd.to_datetime(all_df["fdate"], errors="coerce").dt.to_period("M").dt.to_timestamp() + pd.offsets.MonthEnd(0)

    # ---- PCA on embeddings (if present) ----
    emb_cols = [c for c in all_df.columns if c.startswith("emb_")]
    if emb_cols:
        print(f"[RESUME] Running PCA on {len(emb_cols)} emb dims -> {PCA_DIM}")
        red = PCA(n_components=min(PCA_DIM,len(emb_cols))).fit_transform(all_df[emb_cols].fillna(0))
        pca_df = pd.DataFrame(red, columns=[f"pca_emb_{i+1}" for i in range(red.shape[1])])
        all_df = pd.concat([all_df.drop(columns=emb_cols), pca_df], axis=1)

    # ---- Monthly aggregation ----
    agg = all_df.groupby(["gvkey","ret_eom"]).mean(numeric_only=True).reset_index()
    out_fp = os.path.join(out_dir, "text_features_advanced.csv")
    agg.to_csv(out_fp, index=False)
    print(f"[RESUME] Saved final -> {out_fp} | shape={agg.shape}")

if __name__ == "__main__":
    resume()
