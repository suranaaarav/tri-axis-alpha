"""
Regime-aware, risk- & turnover-aware portfolio optimizer (Step 05).

What this does (monthly loop):
  1) Build a blended alpha (ensemble) from all model predictions, with model-weights updated by
     trailing IC (12m) and regime probabilities (bear/neutral/bull) for next month.
  2) Choose a tradable universe (liquidity & size filter), cap universe size for stability.
  3) Build a risk model: Ledoit–Wolf shrinkage covariance on trailing 12m returns for the universe.
  4) Solve a convex optimization:
        minimize   0.5 w' Σ w  - λ * α' w  + γ * ||w - w_prev||^2
        subject to sum(w) = net_exposure(regime)  (regime-aware net)
                   sector-neutral (Aw = 0), if sector info available
                   bounds l_i ≤ w_i ≤ u_i (size/liquidity-based)
     using SciPy SLSQP (no external solver needed).
  5) Apply volatility targeting to 10% annual vol, compute realized returns, turnover, and stats.
  6) Save monthly weights & returns, plus a summary CSV.

Inputs it expects:
  - outputs/predictions_step04.csv    (gvkey, ret_eom, target + pred_* columns, stock_ret for history)
  - outputs/regime_preds.csv          (ret_eom, regime_pred_next_proba_0/1/2 ; 0=bear,1=neutral,2=bull)
  - outputs/quant_features_full.csv   (to join: sector, liquidity like dolvol_126d or turnover_126d, size 'me')
  - config.yaml with data.mkt_index_csv (ret_eom,mkt_rf)

Outputs:
  - outputs/opt_portfolio_weights.csv
  - outputs/opt_portfolio_returns.csv
  - outputs/opt_portfolio_summary.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm

# -----------------------------
# Hyperparameters (tweak here)
# -----------------------------
ANNUAL_VOL_TARGET = 0.10            # volatility targeting
ROLL_IC_WINDOW    = 12              # months for trailing information coefficient per model
RISK_WINDOW       = 12              # months of trailing returns for covariance
MAX_UNIVERSE      = 400             # cap names per month for stability/speed
MIN_LIQ_PCTL      = 0.30            # keep stocks above this percentile of liquidity (per month)
LBOUND            = -0.02           # per-name min weight
UBOUND            =  0.02           # per-name max weight
TC_BPS            = 25              # roundtrip transaction cost (bps) used for reporting only
LAMBDA_ALPHA      = 3.0             # preference for alpha (linear)
GAMMA_TURN        = 5.0             # turnover penalty (||w - w_prev||^2)

# Regime net exposure mapping (target sum of weights)
# You can make these more/less aggressive
REGIME_NET = dict(
    bear    = -0.10,
    neutral =  0.00,
    bull    =  0.10
)

# Map HMM order (0=bear,1=neutral,2=bull). If you changed in Step 02, adjust here
REGIME_INDEX = {0: "bear", 1: "neutral", 2: "bull"}

# -----------------------------
# Helpers
# -----------------------------
def _capm_alpha(rets: pd.Series, mkt: pd.Series) -> Tuple[float, float]:
    df = pd.DataFrame({"y": rets, "mkt": mkt}).dropna()
    if len(df) < 24:
        return np.nan, np.nan
    X = sm.add_constant(df["mkt"])
    m = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags":3})
    return float(m.params.get("const", np.nan)), float(m.tvalues.get("const", np.nan))

def _annualize(series: pd.Series) -> Tuple[float, float]:
    mu = series.mean() * 12.0
    vol = series.std() * np.sqrt(12.0)
    return float(mu), float(vol)

def _vol_target(series: pd.Series, target_ann_vol: float) -> pd.Series:
    current = series.std() * np.sqrt(12.0)
    if current <= 1e-12 or np.isnan(current):
        return series
    return series * (target_ann_vol / current)

def _ledoit_cov(ret_mat: pd.DataFrame) -> np.ndarray:
    # rows = dates, cols = assets
    if ret_mat.shape[0] < 6:  # not enough history
        # fallback diag
        v = ret_mat.var().fillna(ret_mat.var().median()).values
        return np.diag(np.where(v>0, v, np.nanmedian(v)))
    lw = LedoitWolf().fit(ret_mat.fillna(0.0).values)
    return lw.covariance_

def _sector_neutral_matrix(sectors: pd.Series, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    # Build A such that A @ w = 0 enforces sector neutrality (sum of weights per sector = 0)
    s = sectors.reindex(names)
    cats = [c for c in s.dropna().unique()]
    A = np.zeros((len(cats), len(names)))
    for i, cat in enumerate(cats):
        mask = (s == cat).values
        A[i, mask] = 1.0
    return A, cats

def _rolling_ic(pred_df: pd.DataFrame, model_col: str, target: str, window: int) -> pd.Series:
    ics = []
    for dt, g in pred_df.groupby("ret_eom"):
        if g[model_col].notna().sum() > 10 and g[target].notna().sum() > 10:
            ic = g[[model_col, target]].rank().corr().iloc[0,1]  # Spearman via ranks
            ics.append((dt, ic))
    ic_s = pd.Series({d:i for d,i in ics}).sort_index()
    return ic_s.rolling(window).mean()

def _blend_alpha(cross: pd.DataFrame, model_cols: List[str], ic_weights: Dict[str, float]) -> pd.Series:
    # z-score each model cross-sectionally, then weighted sum by ic_weights
    Z = []
    W = []
    for m in model_cols:
        x = cross[m]
        if x.notna().sum() < 5:
            continue
        z = (x - x.mean())/x.std(ddof=0) if x.std(ddof=0) > 1e-12 else x*0
        Z.append(z)
        W.append(ic_weights.get(m, 0.0))
    if not Z:
        return pd.Series(0.0, index=cross.index)
    Z = np.vstack([z.values for z in Z]).T  # [n, k]
    W = np.array(W)
    if np.allclose(W.sum(), 0):
        W = np.ones_like(W)/len(W)
    blend = (Z @ (W / (np.sum(np.abs(W)) + 1e-12)))
    return pd.Series(blend, index=cross.index)

def _regime_net_exposure(reg_row: pd.Series) -> float:
    # expected net exposure = sum_k p_k * net_k
    # where k in {bear, neutral, bull}
    ps = []
    for k in [0,1,2]:
        p = reg_row.get(f"regime_pred_next_proba_{k}", np.nan)
        ps.append(0.0 if np.isnan(p) else float(p))
    if len(ps) < 3:
        # if missing, neutral
        return REGIME_NET["neutral"]
    ps = np.array(ps)
    ps = ps / (ps.sum() + 1e-12)
    nets = np.array([REGIME_NET[REGIME_INDEX[k]] for k in range(3)])
    return float(np.dot(ps, nets))

def _qp_objective(w: np.ndarray, Sigma: np.ndarray, alpha: np.ndarray, w_prev: np.ndarray,
                  lambda_alpha: float, gamma_turn: float) -> Tuple[float, np.ndarray]:
    # 0.5 w' Σ w  - λ α' w  + γ ||w - w_prev||^2
    diff = w - w_prev
    val = 0.5 * (w @ Sigma @ w) - lambda_alpha * (alpha @ w) + gamma_turn * (diff @ diff)
    # gradient
    grad = (Sigma @ w) - lambda_alpha * alpha + 2.0 * gamma_turn * diff
    return val, grad

# -----------------------------
# Main entry point
# -----------------------------
def build_optimized_portfolios(cfg_path: str):
    from tri_axis_alpha.data.utils import load_config, ensure_dir
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)

    # Load core data
    preds = pd.read_csv(f"{out_dir}/predictions_step04.csv", parse_dates=["ret_eom"])
    target = "ret_fwd_1m" if "ret_fwd_1m" in preds.columns else "stock_ret"
    model_cols = [c for c in preds.columns if c.startswith("pred_")]
    if not model_cols:
        raise RuntimeError("No model prediction columns (pred_*) found in predictions_step04.csv")

    # Join regime predictions (for net exposure)
    reg = None
    reg_path = os.path.join(out_dir, "regime_preds.csv")
    if os.path.exists(reg_path):
        reg = pd.read_csv(reg_path, parse_dates=["ret_eom"])
        preds = preds.merge(reg, on="ret_eom", how="left")
        print("[OPT] Regime predictions loaded.")
    else:
        print("[OPT] Regime predictions not found; net exposure will be 0.")

    # Join quant features for universe filters & sectors
    qf = pd.read_csv(os.path.join(out_dir, "quant_features_full.csv"), parse_dates=["ret_eom"])
    # Attempt to discover useful columns
    # liquidity: prefer 'dolvol_126d' then 'turnover_126d'
    liq_col = "dolvol_126d" if "dolvol_126d" in qf.columns else ("turnover_126d" if "turnover_126d" in qf.columns else None)
    size_col = "me" if "me" in qf.columns else ("market_equity" if "market_equity" in qf.columns else None)
    sector_col = None
    for cand in ["gsector", "gics_sector", "sic2", "sector", "gic_sector"]:
        if cand in qf.columns:
            sector_col = cand
            break
    cols_to_merge = ["gvkey","ret_eom"] + [c for c in [liq_col, size_col, sector_col] if c]
    qf_small = qf[cols_to_merge].drop_duplicates()
    preds = preds.merge(qf_small, on=["gvkey","ret_eom"], how="left")

    # Market series for evaluation
    mkt = pd.read_csv(cfg["data"]["market_csv"], parse_dates=["ret_eom"])
    mkt.columns = ["ret_eom", "mkt_rf"]
    mkt = mkt.set_index("ret_eom")["mkt_rf"]

    # Compute trailing IC series per model (for weights)
    ic_table = {}
    base_ic = {}
    # Create a compact frame for IC computation
    ic_df = preds[["ret_eom","gvkey",target] + model_cols].dropna(subset=[target]).copy()
    for m in model_cols:
        ics = _rolling_ic(ic_df, m, target, ROLL_IC_WINDOW)  # monthly series
        ic_table[m] = ics
        base_ic[m] = float(ics.dropna().median()) if ics.notna().any() else 0.0

    months = sorted(preds["ret_eom"].dropna().unique())
    weights_records = []
    ret_records = []

    # Keep rolling past returns table for covariance
    # We'll build a pivot of realized returns (target or stock_ret lag?) — use realized 'stock_ret'
    if "stock_ret" in preds.columns:
        ret_hist = preds.pivot_table(index="ret_eom", columns="gvkey", values="stock_ret", aggfunc="mean")
    else:
        # fallback to target shifted by -1 if it is actual realized next-month return
        tmp = preds.pivot_table(index="ret_eom", columns="gvkey", values=target, aggfunc="mean")
        ret_hist = tmp.shift(1)

    w_prev = None  # previous month weights (Series)
    for t, dt in enumerate(months):
        month_df = preds[preds["ret_eom"] == dt].copy()

        # Skip if no predictions this month
        if month_df[model_cols].isna().all(axis=None):
            continue

        # 1) Universe filter by liquidity percentile & cap size (if available)
        if liq_col and month_df[liq_col].notna().sum() > 20:
            threshold = month_df[liq_col].quantile(MIN_LIQ_PCTL)
            month_df = month_df[month_df[liq_col] >= threshold]
        # cap universe size by highest absolute ensemble alpha later; first keep enough names
        # if size available, prefer larger caps when oversubscribed
        if size_col and len(month_df) > MAX_UNIVERSE*2:
            month_df = month_df.sort_values(size_col, ascending=False).head(MAX_UNIVERSE*2)

        # 2) Build model weights for this month using IC and regime tilts
        #    regime tilt: increase weight on models with higher recent IC in current predicted regime
        if reg is not None:
            reg_row = reg[reg["ret_eom"] == dt]
            if len(reg_row) == 1:
                p_bear   = float(reg_row.filter(like="proba_0").iloc[0,0]) if "regime_pred_next_proba_0" in reg_row.columns else 1/3
                p_neutral= float(reg_row.filter(like="proba_1").iloc[0,0]) if "regime_pred_next_proba_1" in reg_row.columns else 1/3
                p_bull   = float(reg_row.filter(like="proba_2").iloc[0,0]) if "regime_pred_next_proba_2" in reg_row.columns else 1/3
            else:
                p_bear = p_neutral = p_bull = 1/3
        else:
            p_bear = p_neutral = p_bull = 1/3

        # base IC weights from trailing window, rescaled to positive
        icw = {}
        for m in model_cols:
            ic_series = ic_table[m]
            val = ic_series.loc[:dt].iloc[-1] if dt in ic_series.index else base_ic[m]
            icw[m] = max(val, 0.0)
        # normalize
        s = sum(icw.values()) + 1e-12
        for m in icw:
            icw[m] = icw[m] / s

        # 3) Blend alpha (z-scored per model cross-sectionally, weighted by icw)
        alpha = _blend_alpha(month_df.set_index("gvkey"), model_cols, icw)
        # keep names to a manageable size by absolute alpha
        if len(alpha) > MAX_UNIVERSE:
            keep = alpha.abs().sort_values(ascending=False).head(MAX_UNIVERSE).index
            month_df = month_df.set_index("gvkey").loc[keep].reset_index()
            alpha = alpha.loc[keep]

        names = alpha.index.tolist()

        # 4) Risk model: trailing RISK_WINDOW months covariance on the same names
        hist_end   = dt - pd.offsets.MonthEnd(1)
        hist_start = hist_end - pd.DateOffset(months=RISK_WINDOW-1)
        R = ret_hist.loc[(ret_hist.index >= hist_start) & (ret_hist.index <= hist_end), names]
        Sigma = _ledoit_cov(R)

        # 5) Regime-aware net exposure
        reg_row = preds[preds["ret_eom"] == dt].iloc[0] if reg is not None and dt in reg["ret_eom"].values else pd.Series()
        net_target = _regime_net_exposure(reg_row) if len(reg_row) else REGIME_NET["neutral"]

        # 6) Bounds per name (can adapt by size/liquidity)
        lb = np.full(len(names), LBOUND)
        ub = np.full(len(names), UBOUND)

        # 7) Optional sector neutrality
        A_eq = None
        if "gics_sector" in month_df.columns or "gsector" in month_df.columns or "sector" in month_df.columns:
            sector_col = "gics_sector" if "gics_sector" in month_df.columns else ("gsector" if "gsector" in month_df.columns else "sector")
            month_df = month_df.set_index("gvkey")
            A, cats = _sector_neutral_matrix(month_df[sector_col], names)
            A_eq = A  # shape (S, N)
            month_df = month_df.reset_index()

        # 8) Previous weights aligned
        if w_prev is None:
            w_prev_vec = np.zeros(len(names))
        else:
            w_prev_vec = w_prev.reindex(names).fillna(0.0).values

        # 9) Solve convex problem with SLSQP
        # objective & gradient wrapper
        alpha_vec = alpha.values.astype(float)
        def fun(w):
            val, grad = _qp_objective(w, Sigma, alpha_vec, w_prev_vec, LAMBDA_ALPHA, GAMMA_TURN)
            return val
        def jac(w):
            val, grad = _qp_objective(w, Sigma, alpha_vec, w_prev_vec, LAMBDA_ALPHA, GAMMA_TURN)
            return grad

        # Equality constraints: sum(w)=net_target, and sector neutrality if provided
        cons = [dict(type="eq", fun=lambda w, nt=net_target: np.sum(w) - nt,
                          jac=lambda w: np.ones_like(w))]
        if A_eq is not None and A_eq.size > 0:
            for i in range(A_eq.shape[0]):
                row = A_eq[i,:].copy()
                cons.append(dict(type="eq",
                                 fun=lambda w, r=row: float(r @ w),
                                 jac=lambda w, r=row: r))

        bounds = tuple((lb[i], ub[i]) for i in range(len(names)))
        w0 = np.clip(w_prev_vec, lb, ub)  # warm-start from previous

        res = minimize(fun, w0, method="SLSQP", jac=jac, bounds=bounds, constraints=cons,
                       options=dict(maxiter=500, ftol=1e-9, disp=False))
        if not res.success:
            # fallback: project w_prev to constraints (zero alpha)
            w_opt = np.clip(w_prev_vec, lb, ub)
            # enforce sum(w)=net_target by simple shift
            shift = (net_target - w_opt.sum())/len(w_opt)
            w_opt = np.clip(w_opt + shift, lb, ub)
        else:
            w_opt = res.x

        # 10) Compute realized return and turnover
        month_df = month_df.set_index("gvkey")
        realized = month_df[target].reindex(names).fillna(0.0).values
        r_month = float(np.dot(w_opt, realized))
        turnover = np.sum(np.abs(w_opt - w_prev_vec)) / 2.0 if w_prev is not None else np.nan

        # Store weights & return
        weights_records.append(pd.DataFrame({
            "ret_eom": dt, "gvkey": names, "weight": w_opt
        }))
        ret_records.append(dict(ret_eom=dt, return_=r_month, turnover=turnover))

        # Next loop warm start
        w_prev = pd.Series(w_opt, index=names)

        print(f"[OPT] {dt.date()}  n={len(names)}  ret={r_month:.4f}  net={w_opt.sum():+.3f}  to={turnover if turnover==turnover else 0:.3f}")

    # Build outputs
    weights_df = pd.concat(weights_records, ignore_index=True) if weights_records else pd.DataFrame(columns=["ret_eom","gvkey","weight"])
    rets_df = pd.DataFrame(ret_records).sort_values("ret_eom") if ret_records else pd.DataFrame(columns=["ret_eom","return_","turnover"])

    # Vol targeting & net of simple TC estimate
    rets = rets_df.set_index("ret_eom")["return_"]
    rets_vt = _vol_target(rets, ANNUAL_VOL_TARGET)
    # naive TC deduction proxy using average turnover * bps
    to_series = rets_df.set_index("ret_eom")["turnover"].fillna(0.0)
    tc_monthly = (TC_BPS/10000.0) * to_series  # bps to decimal
    rets_net = rets_vt - tc_monthly

    # Metrics
    mu_g, vol_g = _annualize(rets_net)
    sharpe = mu_g / (vol_g + 1e-12)
    # Market alignment
    mkt_aligned = mkt.reindex(rets_net.index)
    alpha, tstat = _capm_alpha(rets_net, mkt_aligned)
    mdd = (np.log1p(rets_net).cumsum() - np.log1p(rets_net).cumsum().cummax()).min()

    summary = pd.DataFrame([dict(
        ann_return=mu_g, ann_vol=vol_g, sharpe=sharpe,
        capm_alpha=alpha, alpha_t=tstat,
        max_drawdown=float(mdd),
        avg_turnover=float(to_series.mean())
    )])

    # Save
    weights_out = os.path.join(out_dir, "opt_portfolio_weights.csv")
    returns_out = os.path.join(out_dir, "opt_portfolio_returns.csv")
    summary_out = os.path.join(out_dir, "opt_portfolio_summary.csv")

    weights_df.to_csv(weights_out, index=False)
    pd.DataFrame({"ret_eom": rets_net.index, "return": rets_net.values}).to_csv(returns_out, index=False)
    summary.to_csv(summary_out, index=False)

    print("[OPT] Saved:")
    print(f"      weights  -> {weights_out}  (rows={len(weights_df)})")
    print(f"      returns  -> {returns_out}  (rows={len(rets_net)})")
    print(f"      summary  -> {summary_out}")
    print(summary.to_string(index=False))
