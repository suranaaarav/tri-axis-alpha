"""
Step 02a (Advanced): Build monthly regime features
- Market index stats (returns, vol, skew, kurt, drawdowns, momentum)
- Cross-sectional breadth/dispersion/liquidity from quant panel
- Aggregated text sentiment/uncertainty
- Memory-safe (chunked reads), robust to schema variants
Outputs: outputs/regime_features_adv.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from tri_axis_alpha.data.utils import load_config, ensure_dir

# -----------------------------
# Helpers
# -----------------------------
def _month_endify_date(s: pd.Series) -> pd.Series:
    return s.values.astype("datetime64[M]") + MonthEnd(0)

def _rolling_stats(df: pd.DataFrame, col: str, wins=(3, 6, 12, 24)) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for w in wins:
        out[f"{col}_mean_{w}m"] = df[col].rolling(w).mean()
        out[f"{col}_vol_{w}m"]  = df[col].rolling(w).std()
        out[f"{col}_skew_{w}m"] = df[col].rolling(w).skew()
        out[f"{col}_kurt_{w}m"] = df[col].rolling(w).kurt()
        out[f"{col}_mom_{w}m"]  = df[col].rolling(w).sum()
    return out

def _drawdown_series(ret: pd.Series, win: int = 36) -> pd.Series:
    cum = (1.0 + ret.fillna(0)).cumprod()
    rolling_peak = cum.rolling(win, min_periods=1).max()
    return cum / rolling_peak - 1.0

def _safe_usecols(path: str, need: list[str]) -> list[str]:
    hdr = pd.read_csv(path, nrows=0).columns.tolist()
    return [c for c in need if c in hdr]

# -----------------------------
# Main
# -----------------------------
def build_regime_features_advanced(cfg_path: str) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)

    # ---------- 1) MARKET INDEX ----------
    mkt_path = cfg["data"]["market_csv"]
    mkt = pd.read_csv(mkt_path, low_memory=False)

    if "date" in mkt.columns:
        mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce")
        mkt = mkt.rename(columns={"date": "ret_eom"})
        mkt["ret_eom"] = _month_endify_date(mkt["ret_eom"])
    elif {"year","month"}.issubset(mkt.columns):
        mkt["ret_eom"] = pd.to_datetime(mkt["year"].astype(int).astype(str) + "-" +
                                        mkt["month"].astype(int).astype(str) + "-01") + MonthEnd(0)
    else:
        raise ValueError("market_index needs either a 'date' column or 'year' & 'month'.")

    if "mkt_rf" not in mkt.columns:
        # try common aliases
        for alt in ["mkt", "mkt_excess", "market_rf", "market_ret"]:
            if alt in mkt.columns:
                mkt = mkt.rename(columns={alt: "mkt_rf"})
                break
        if "mkt_rf" not in mkt.columns:
            raise ValueError("market_index must contain 'mkt_rf' (monthly market excess return).")

    mkt = mkt[["ret_eom","mkt_rf"]].dropna().sort_values("ret_eom").reset_index(drop=True)
    mkt = mkt.drop_duplicates(subset=["ret_eom"])

    # Rolling stats
    mkt_idx = mkt.set_index("ret_eom")
    roll = _rolling_stats(mkt_idx, "mkt_rf", wins=(3,6,12,24))

    # Drawdowns (12m, 36m) & speed
    dd12 = _drawdown_series(mkt_idx["mkt_rf"], win=12)
    dd36 = _drawdown_series(mkt_idx["mkt_rf"], win=36)
    ddspd6 = dd36.diff().rolling(6).mean()

    market_block = pd.concat([mkt_idx, roll, dd12.rename("dd_12m"), dd36.rename("dd_36m"), ddspd6.rename("dd_speed_6m")], axis=1)\
                    .reset_index()

    # ---------- 2) CROSS-SECTIONAL STATS (quant panel) ----------
    panel_path = cfg["data"]["ret_sample_csv"]
    need = [
        "ret_eom","stock_ret","ami_126d","zero_trades_21d","zero_trades_126d",
        "bidaskhl_21d","dolvol_126d","rskew_21d","ivol_capm_21d"
    ]
    usecols = _safe_usecols(panel_path, need)
    if not {"ret_eom","stock_ret"}.issubset(set(usecols)):
        raise ValueError("ret_sample_csv must contain at least 'ret_eom' and 'stock_ret'.")

    cs_blocks = []
    for ch in pd.read_csv(panel_path, usecols=usecols, parse_dates=["ret_eom"],
                          chunksize=1_000_000, low_memory=True):
        g = ch.groupby("ret_eom")
        a = g.agg(
            breadth_pos=("stock_ret", lambda s: np.mean((s > 0).astype(float))),
            xsec_mean_ret=("stock_ret","mean"),
            xsec_std_ret=("stock_ret","std"),
            xsec_skew_ret=("stock_ret","skew"),
            ivol21_mean=("ivol_capm_21d","mean"),
            rskew21_mean=("rskew_21d","mean"),
            ami_mean=("ami_126d","mean"),
            zero21_mean=("zero_trades_21d","mean"),
            zero126_mean=("zero_trades_126d","mean"),
            bidaskhl_mean=("bidaskhl_21d","mean"),
            dolvol_mean=("dolvol_126d","mean"),
            n_stocks=("stock_ret","size"),
        ).reset_index()
        cs_blocks.append(a)

    xsec = pd.concat(cs_blocks, ignore_index=True)\
             .groupby("ret_eom", as_index=False).mean(numeric_only=True)

    # Liquidity stress proxy (higher => tighter liquidity)
    xsec["liq_stress"] = (
        xsec["zero21_mean"].fillna(xsec["zero21_mean"].median()) * 0.40 +
        xsec["zero126_mean"].fillna(xsec["zero126_mean"].median()) * 0.30 +
        xsec["bidaskhl_mean"].fillna(xsec["bidaskhl_mean"].median()) * 0.20 -
        xsec["dolvol_mean"].fillna(xsec["dolvol_mean"].median()) * 0.10
    )

    # ---------- 3) TEXT AGGREGATES (optional) ----------
    txt_agg = None
    txt_path = os.path.join(out_dir, "text_features_advanced.csv")
    if os.path.exists(txt_path):
        tf = pd.read_csv(txt_path, parse_dates=["ret_eom"])
        txt_agg = tf.groupby("ret_eom", as_index=False).agg(
            lm_unc_mean=("lm_unc","mean"),
            lm_neg_mean=("lm_neg","mean"),
            lm_pos_mean=("lm_pos","mean")
        )
        print("[REGIME-FEATS-ADV] Merged text aggregates.")
    else:
        print("[REGIME-FEATS-ADV] text_features_advanced.csv not found; skipping text aggregates.")

    # ---------- 4) MERGE & STANDARDIZE ----------
    df = market_block.merge(xsec, on="ret_eom", how="left")
    if txt_agg is not None:
        df = df.merge(txt_agg, on="ret_eom", how="left")

    df = df.sort_values("ret_eom").reset_index(drop=True)

    # Time-wise standardization for regime predictors (keep ret_eom, mkt_rf)
    keep = {"ret_eom", "mkt_rf"}
    feats = [c for c in df.columns if c not in keep]
    for c in feats:
        mu, sd = df[c].mean(), df[c].std()
        if pd.notna(sd) and sd > 0:
            df[c] = (df[c] - mu) / sd

    out_fp = f"{out_dir}/regime_features_adv.csv"
    df.to_csv(out_fp, index=False)
    print(f"[REGIME-FEATS-ADV] saved -> {out_fp} | shape={df.shape}")
    return df

if __name__ == "__main__":
    build_regime_features_advanced("config.yaml")
