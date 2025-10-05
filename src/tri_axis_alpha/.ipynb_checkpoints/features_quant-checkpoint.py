"""
Memory-safe quantitative feature engineering with batching and missing-column guards.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from .utils import load_config, ensure_dir

# ---- Helpers ----
def _winsor(s: pd.Series, p: float = 0.01) -> pd.Series:
    """Winsorize at p and 1-p quantile."""
    if s.isna().all():
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def _process_month(df: pd.DataFrame, factors: list[str], sector: str | None, batch_size: int = 500) -> pd.DataFrame:
    """Process one monthly cross section with batching and safety guards."""
    out = df.copy(deep=False)  # shallow copy (memory safe)
    month = out["ret_eom"].iloc[0].date()
    print(f"[QUANT] Processing {month} with {len(out)} rows")

    for i in range(0, len(factors), batch_size):
        batch = factors[i : i + batch_size]
        print(f"    -> Batch {i} to {i+len(batch)-1}")

        for f in batch:
            if f not in out.columns:
                # column not present in this month
                out[f] = 0
                out[f"{f}_miss"] = 1
                out[f"{f}_rank"] = 0
                continue

            s = out[f]
            if not isinstance(s, pd.Series):
                # just in case s is scalar
                out[f] = 0
                out[f"{f}_miss"] = 1
                out[f"{f}_rank"] = 0
                continue

            miss = s.isna().astype(np.int8)
            s = _winsor(s)
            s = s.fillna(s.median())

            if sector and sector in out.columns:
                try:
                    sector_mean = out.groupby(sector)[f].transform("mean")
                    s = s - sector_mean
                except Exception:
                    pass

            mu, sd = s.mean(), s.std()
            if sd and not np.isnan(sd):
                s = (s - mu) / sd
            else:
                s = 0

            if isinstance(s, (int, float)):
                out[f] = s
                out[f"{f}_miss"] = miss
                out[f"{f}_rank"] = 0
            else:
                r = s.rank(method="dense")
                r_scaled = 0 if r.max() <= 1 else (r - 1) / (r.max() - 1) * 2 - 1
                out[f] = s
                out[f"{f}_miss"] = miss
                out[f"{f}_rank"] = r_scaled

    return out

# ---- Main ----
def build_quant_features(cfg_path: str) -> pd.DataFrame:
    print("[QUANT] Loading config & inputs...")
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    ret_path = cfg["data"]["ret_sample_csv"]
    ratio_path = cfg["data"]["acc_ratios_csv"]
    sector_path = cfg["data"].get("sector_info_csv", None)

    # Factor list
    factors = pd.read_csv(ratio_path)["Variable"].dropna().tolist()
    print(f"    -> Found {len(factors)} accounting ratios")

    # Load in chunks
    print(f"[QUANT] Streaming {ret_path}")
    chunks = []
    for chunk in pd.read_csv(ret_path, parse_dates=["ret_eom"], low_memory=True, chunksize=500000):
        chunks.append(chunk)
    raw = pd.concat(chunks, ignore_index=True)
    print(f"    -> Raw shape {raw.shape}")

    # Sector info if available
    sector = None
    if sector_path:
        try:
            sec = pd.read_csv(sector_path)
            if "gvkey" in sec.columns and "sector" in sec.columns:
                raw = raw.merge(sec[["gvkey", "sector"]], on="gvkey", how="left")
                sector = "sector"
                print("[QUANT] Sector column merged.")
        except Exception as e:
            print("[QUANT] Sector merge failed:", e)

    # Process month by month
    monthly_out = []
    for month, month_df in tqdm(raw.groupby("ret_eom"), desc="Monthly processing"):
        try:
            processed = _process_month(month_df, factors, sector)
            monthly_out.append(processed)
        except Exception as e:
            print(f"    !! Failed {month}: {e}")

    final = pd.concat(monthly_out, ignore_index=True)
    out_fp = f"{out_dir}/quant_features_full.csv"
    final.to_csv(out_fp, index=False)
    print(f"[QUANT] Saved -> {out_fp} | shape={final.shape}")
    return final
