"""
Step 04: Train return-prediction models with expanding window (parallel + memory safe).
- Models: OLS, Ridge, Lasso, ElasticNet, LightGBM (if installed), XGBoost (if installed)
- Target: ret_fwd_1m if present, else stock_ret
- Reads only the months needed for each fold (no full-file load)
- Parallelizes folds across CPU cores
- Outputs:
    outputs/predictions_step04.csv
    outputs/model_oos_summary_step04.csv
    outputs/model_metrics_step04.csv
    outputs/feature_importance_{lgbm|xgb}.csv
"""

from __future__ import annotations
import os, time, warnings
import numpy as np
import pandas as pd
from typing import List, Set
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from tri_axis_alpha.data.utils import load_config, ensure_dir

# ---- Optional tree libs ----
_HAS_LGBM = False
_HAS_XGB  = False
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    pass
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42
MIN_TRAIN_MONTHS = 96        # at least 8 years before first test
VAL_MONTHS        = 12       # last 12 months of train for validation
LINEAR_ALPHAS     = np.logspace(-4, 2, 12)
CHUNK_SIZE_ROWS   = 400_000  # tune for machine

# --- User control ---
RUN_MODELS = ["ols","ridge","lasso","elasticnet","lgbm","xgb"]  # comment out to run fewer
N_JOBS     = -1  # -1 = all cores

# -----------------------------
# Helpers
# -----------------------------
def _pick_target(cols: List[str]) -> str:
    return "ret_fwd_1m" if "ret_fwd_1m" in cols else "stock_ret"

EXCLUDE_COLS_BASE = {"gvkey","id","ret_eom","date","year","month"}

def _infer_feature_list_from_header(csv_path: str, target: str) -> List[str]:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    drop = set(EXCLUDE_COLS_BASE | {target})
    keep_like = (
        "_rank","_miss","_lead4","lm_","pca_emb_","fog","flesch",
        "regime_post_", "regime_pred_next_proba_", "regime_pred_next_class",
    )
    feats = []
    for c in header:
        if c in drop: 
            continue
        if any(c.startswith(pfx) for pfx in keep_like) or any(c.endswith(sfx) for sfx in ("_rank","_miss","_lead4")):
            feats.append(c)
    for base in ["me","market_equity","dolvol_126d","turnover_126d","beta_60m","ivol_capm_21d","rskew_21d"]:
        if base in header and base not in feats and base not in drop:
            feats.append(base)
    return sorted(list(dict.fromkeys(feats)))

def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    return df

def _read_month_index(csv_path: str) -> List[pd.Timestamp]:
    months = []
    for ch in pd.read_csv(csv_path, usecols=["ret_eom"], parse_dates=["ret_eom"],
                          chunksize=CHUNK_SIZE_ROWS, low_memory=False):
        months.append(ch["ret_eom"])
    m = pd.concat(months, ignore_index=True).dropna().unique()
    m = np.sort(m)
    return list(m)

def _load_slice(csv_path: str, month_set: Set[pd.Timestamp], cols: List[str], target: str) -> pd.DataFrame:
    use = ["gvkey","ret_eom",target] + cols
    use = sorted(list(dict.fromkeys(use)))
    out = []
    dtypes = {"gvkey": "Int64"}
    for ch in pd.read_csv(csv_path, usecols=lambda c: c in use, parse_dates=["ret_eom"],
                          chunksize=CHUNK_SIZE_ROWS, low_memory=False, dtype=dtypes):
        ch = ch[ch["ret_eom"].isin(month_set)]
        if ch.empty: continue
        out.append(_downcast_floats(ch))
    if not out:
        return pd.DataFrame(columns=use)
    return pd.concat(out, ignore_index=True)

def compute_oos_r2(y_true, y_pred):
    mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
    if mask.sum() == 0: return float("nan")
    y, yhat = y_true[mask], y_pred[mask]
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return 1 - sse / (sst + 1e-12)

def _monthly_ic(df: pd.DataFrame, pred_col: str, target_col: str) -> float:
    ics = []
    for dt, g in df.groupby("ret_eom"):
        if g[pred_col].notna().sum() > 10 and g[target_col].notna().sum() > 10:
            ic = spearmanr(g[pred_col], g[target_col], nan_policy="omit").correlation
            if pd.notna(ic): ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan

def _tune_linear(model_cls, Xtr, ytr, Xva, yva, alphas=LINEAR_ALPHAS, l1_ratio=None):
    best_alpha, best_score = None, np.inf
    for a in alphas:
        if model_cls is ElasticNet and l1_ratio is not None:
            m = model_cls(alpha=a, l1_ratio=l1_ratio, fit_intercept=True, max_iter=20000, random_state=RANDOM_SEED)
        else:
            m = model_cls(alpha=a, fit_intercept=True, max_iter=20000) if model_cls in (Lasso, ElasticNet) else model_cls(alpha=a, fit_intercept=True)
        m.fit(Xtr, ytr)
        mse = mean_squared_error(yva, m.predict(Xva))
        if mse < best_score:
            best_alpha, best_score = a, mse
    if model_cls is ElasticNet and l1_ratio is not None:
        return model_cls(alpha=best_alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=20000, random_state=RANDOM_SEED)
    elif model_cls in (Lasso, ElasticNet):
        return model_cls(alpha=best_alpha, fit_intercept=True, max_iter=20000)
    else:
        return model_cls(alpha=best_alpha, fit_intercept=True)

# -----------------------------
# Per-fold worker
# -----------------------------
def _train_one_fold(i, test_month, train_end, val_start, months, data_fp, feats, target):
    t0 = time.time()
    print(f"\n[FOLD] {i - MIN_TRAIN_MONTHS + 1}/{len(months)-MIN_TRAIN_MONTHS} "
          f"| Test={pd.to_datetime(test_month).date()} | Train<= {pd.to_datetime(train_end).date()} | Val>= {pd.to_datetime(val_start).date()}")
    train_months = {m for m in months if m < val_start}
    val_months   = {m for m in months if (m >= val_start) and (m <= train_end)}
    test_months  = {test_month}

    df_tr = _load_slice(data_fp, train_months, feats, target).dropna(subset=[target])
    df_va = _load_slice(data_fp, val_months,   feats, target).dropna(subset=[target])
    df_te = _load_slice(data_fp, test_months,  feats, target)
    if df_tr.empty or df_va.empty or df_te.empty:
        print("    [SKIP] insufficient data"); return None

    for d in [df_tr, df_va, df_te]:
        d.sort_values(["ret_eom","gvkey"], inplace=True)
    med = df_tr[feats].median()
    for d in [df_tr, df_va, df_te]:
        d[feats] = d[feats].fillna(med).fillna(0)

    scaler = StandardScaler()
    Xtr, Xva, Xte = scaler.fit_transform(df_tr[feats]), scaler.transform(df_va[feats]), scaler.transform(df_te[feats])
    ytr, yva = df_tr[target].values.astype(np.float32), df_va[target].values.astype(np.float32)
    fold_preds = []

    if "ols" in RUN_MODELS:
        print("    [MODEL] OLS")
        m = LinearRegression(); m.fit(Xtr, ytr)
        fold_preds.append(("ols", m.predict(Xte)))
    if "ridge" in RUN_MODELS:
        print("    [MODEL] Ridge")
        m = _tune_linear(Ridge, Xtr, ytr, Xva, yva); m.fit(np.vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
        fold_preds.append(("ridge", m.predict(Xte)))
    if "lasso" in RUN_MODELS:
        print("    [MODEL] Lasso")
        m = _tune_linear(Lasso, Xtr, ytr, Xva, yva); m.fit(np.vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
        fold_preds.append(("lasso", m.predict(Xte)))
    if "elasticnet" in RUN_MODELS:
        print("    [MODEL] ElasticNet")
        m = _tune_linear(ElasticNet, Xtr, ytr, Xva, yva, l1_ratio=0.5); m.fit(np.vstack([Xtr,Xva]), np.concatenate([ytr,yva]))
        fold_preds.append(("elasticnet", m.predict(Xte)))
    if "lgbm" in RUN_MODELS and _HAS_LGBM:
        print("    [MODEL] LightGBM")
        lgb_tr, lgb_va = lgb.Dataset(Xtr,label=ytr), lgb.Dataset(Xva,label=yva,reference=lgb_tr)
        params = dict(objective="regression", metric="rmse", learning_rate=0.05,num_leaves=63,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,min_data_in_leaf=100,
                      seed=RANDOM_SEED, verbose=-1)
        m = lgb.train(params, lgb_tr, num_boost_round=3000, valid_sets=[lgb_tr, lgb_va],
                      early_stopping_rounds=200, verbose_eval=False)
        fold_preds.append(("lgbm", m.predict(Xte, num_iteration=m.best_iteration)))
    if "xgb" in RUN_MODELS and _HAS_XGB:
        print("    [MODEL] XGBoost")
        xg_tr, xg_va, xg_te = xgb.DMatrix(Xtr,label=ytr), xgb.DMatrix(Xva,label=yva), xgb.DMatrix(Xte)
        params = dict(objective="reg:squarederror", eta=0.05, max_depth=7, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, seed=RANDOM_SEED, eval_metric="rmse")
        m = xgb.train(params, xg_tr, num_boost_round=3000, evals=[(xg_tr,"train"),(xg_va,"val")],
                      early_stopping_rounds=200, verbose_eval=False)
        fold_preds.append(("xgb", m.predict(xg_te, iteration_range=(0, m.best_iteration+1))))

    fold = df_te[["gvkey","ret_eom",target]].copy()
    for name, p in fold_preds: fold[f"pred_{name}"] = p.astype(np.float32)
    print(f"    [DONE] {len(fold)} rows | {time.time()-t0:.1f}s")
    return fold

# -----------------------------
# Main
# -----------------------------
def train_models(cfg_path: str):
    print("[TRAIN] Loading config...")
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)

    data_fp = os.path.join(out_dir, "model_dataset.csv")
    header = pd.read_csv(data_fp, nrows=0).columns.tolist()
    target = _pick_target(header)
    feats = _infer_feature_list_from_header(data_fp, target)
    print(f"[TRAIN] Target: {target} | Feature count: {len(feats)}")

    months = _read_month_index(data_fp)
    print(f"[TRAIN] Months available: {len(months)}")
    if len(months) < MIN_TRAIN_MONTHS + 1:
        raise ValueError("Not enough history to start expanding-window training.")

    # --- Parallel loop ---
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_train_one_fold)(i, months[i], months[i-1], months[max(i-VAL_MONTHS-1,0)],
                                 months, data_fp, feats, target)
        for i in range(MIN_TRAIN_MONTHS, len(months))
    )
    preds_all = [r for r in results if r is not None]

    if not preds_all:
        raise RuntimeError("No predictions were produced.")

    preds = pd.concat(preds_all, ignore_index=True)
    preds.to_csv(os.path.join(out_dir, "predictions_step04.csv"), index=False)
    print(f"[TRAIN] Saved predictions -> {out_dir}/predictions_step04.csv | shape={preds.shape}")

    # ---- Metrics ----
    models = [c.replace("pred_","") for c in preds.columns if c.startswith("pred_")]
    rows, overall_r2 = [], {}
    for m in models:
        y, yhat = preds[target].values, preds[f"pred_{m}"].values
        r2  = compute_oos_r2(y, yhat)
        mse = mean_squared_error(y[~np.isnan(yhat)], yhat[~np.isnan(yhat)])
        ic  = _monthly_ic(preds[["ret_eom",target,f"pred_{m}"]].dropna(), f"pred_{m}", target)
        rows.append(dict(model=m, oos_r2=r2, oos_mse=mse, monthly_ic=ic))
        overall_r2[m] = r2

    summ = pd.DataFrame(rows).sort_values(["monthly_ic","oos_r2"], ascending=[False,False])
    summ.to_csv(os.path.join(out_dir, "model_oos_summary_step04.csv"), index=False)
    print(f"[TRAIN] Saved OOS summary -> {out_dir}/model_oos_summary_step04.csv")
    print(summ.to_string(index=False))

    avg_ic = np.nanmean([r["monthly_ic"] for r in rows if not np.isnan(r["monthly_ic"])])
    best = max(overall_r2, key=overall_r2.get)
    pd.DataFrame([
        {"metric":"best_model", "value":best},
        {"metric":"best_model_oos_r2", "value":overall_r2.get(best,np.nan)},
        {"metric":"avg_monthly_ic", "value":avg_ic}
    ]).to_csv(os.path.join(out_dir, "model_metrics_step04.csv"), index=False)
    print(f"[TRAIN] Saved model_metrics_step04.csv with OOS R² and IC")

if __name__ == "__main__":
    train_models("config.yaml")
