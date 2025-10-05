"""
Step 04: Train return-prediction models with expanding window.
- Models: OLS, Ridge, Lasso, ElasticNet, LightGBM (if installed), XGBoost (if installed)
- Target: ret_fwd_1m if present, else stock_ret
- Expanding window by month with held-out validation (last 12m of train) for tuning/early-stopping
- Outputs:
    outputs/predictions_step04.csv
    outputs/model_oos_summary_step04.csv
    outputs/feature_importance_{lgbm|xgb}.csv (if available)
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from tri_axis_alpha.data.utils import load_config, ensure_dir

# Optional tree libs
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
MIN_TRAIN_MONTHS = 96        # at least 8 years history before first test
VAL_MONTHS        = 12       # last 12 months of train used for validation
LINEAR_ALPHAS     = np.logspace(-4, 2, 12)  # for Ridge/Lasso/EN grid

def _pick_target(cols: list[str]) -> str:
    if "ret_fwd_1m" in cols:
        return "ret_fwd_1m"
    return "stock_ret"

def _build_feature_list(df: pd.DataFrame, target: str) -> list[str]:
    # Numerical features only; drop ids/target/date-like columns
    drop_like = {"gvkey", "id", "ret_eom", "date", "year", "month", target}
    # Also drop text columns accidentally non-numeric
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c not in drop_like]
    # Drop fully NA / near-constant columns
    kept = []
    for c in feats:
        s = df[c]
        if s.isna().mean() >= 0.99:
            continue
        # near-constant
        if s.dropna().nunique() <= 1:
            continue
        kept.append(c)
    return kept

def _tune_linear(model_cls, Xtr, ytr, Xva, yva, alphas=LINEAR_ALPHAS, l1_ratio=None):
    best_alpha, best_score = None, np.inf
    for a in alphas:
        if model_cls is ElasticNet and l1_ratio is not None:
            m = model_cls(alpha=a, l1_ratio=l1_ratio, fit_intercept=True, max_iter=20000, random_state=RANDOM_SEED)
        else:
            m = model_cls(alpha=a, fit_intercept=True, max_iter=20000) if model_cls in (Lasso, ElasticNet) else model_cls(alpha=a, fit_intercept=True)
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        mse = mean_squared_error(yva, pred)
        if mse < best_score:
            best_alpha, best_score = a, mse
    if model_cls is ElasticNet and l1_ratio is not None:
        return model_cls(alpha=best_alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=20000, random_state=RANDOM_SEED)
    elif model_cls in (Lasso, ElasticNet):
        return model_cls(alpha=best_alpha, fit_intercept=True, max_iter=20000)
    else:
        return model_cls(alpha=best_alpha, fit_intercept=True)

def _monthly_ic(df: pd.DataFrame, pred_col: str, target_col: str) -> float:
    # Average monthly Spearman IC
    ics = []
    for dt, g in df.groupby("ret_eom"):
        if g[pred_col].notna().sum() > 10 and g[target_col].notna().sum() > 10:
            ic = spearmanr(g[pred_col], g[target_col], nan_policy="omit").correlation
            if pd.notna(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan

def train_models(cfg_path: str):
    print("[TRAIN] Loading config and dataset...")
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)

    data_fp = os.path.join(out_dir, "model_dataset.csv")
    df = pd.read_csv(data_fp, parse_dates=["ret_eom"])
    df = df.sort_values(["ret_eom", "gvkey"]).reset_index(drop=True)

    target = _pick_target(df.columns.tolist())
    print(f"[TRAIN] Target set to: {target}")

    feats = _build_feature_list(df, target)
    print(f"[TRAIN] Using {len(feats)} numeric features.")

    # Timeline
    months = sorted(df["ret_eom"].dropna().unique())
    if len(months) < MIN_TRAIN_MONTHS + 1:
        raise ValueError("Not enough history to start expanding-window training.")

    # Collect predictions
    preds_all = []

    # Optional: feature importance trackers
    lgbm_imps = []
    xgb_imps  = []

    # Expanding window loop
    for i in range(MIN_TRAIN_MONTHS, len(months)):
        test_month = months[i]
        train_end  = months[i - 1]
        val_start  = months[max(i - VAL_MONTHS - 1, 0)]  # inclusive

        tr_mask = df["ret_eom"] < val_start
        va_mask = (df["ret_eom"] >= val_start) & (df["ret_eom"] <= train_end)
        te_mask = df["ret_eom"] == test_month

        df_tr = df.loc[tr_mask, ["gvkey","ret_eom",target] + feats].dropna(subset=[target])
        df_va = df.loc[va_mask, ["gvkey","ret_eom",target] + feats].dropna(subset=[target])
        df_te = df.loc[te_mask, ["gvkey","ret_eom",target] + feats]  # keep even if target NA

        if df_tr.empty or df_va.empty or df_te.empty:
            print(f"[TRAIN] Skipping {test_month.date()} (insufficient data).")
            continue

        # Scale features (fit on train only)
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(df_tr[feats].values)
        Xva = scaler.transform(df_va[feats].values)
        Xte = scaler.transform(df_te[feats].values)

        ytr = df_tr[target].values
        yva = df_va[target].values

        fold_preds = []

        # ---- OLS ----
        ols = LinearRegression(fit_intercept=True)
        ols.fit(Xtr, ytr)
        p_ols = ols.predict(Xte)
        fold_preds.append(("ols", p_ols))

        # ---- Ridge ----
        ridge = _tune_linear(Ridge, Xtr, ytr, Xva, yva, LINEAR_ALPHAS)
        ridge.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
        p_rdg = ridge.predict(Xte)
        fold_preds.append(("ridge", p_rdg))

        # ---- Lasso ----
        lasso = _tune_linear(Lasso, Xtr, ytr, Xva, yva, LINEAR_ALPHAS)
        lasso.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
        p_las = lasso.predict(Xte)
        fold_preds.append(("lasso", p_las))

        # ---- Elastic Net (0.5) ----
        en = _tune_linear(ElasticNet, Xtr, ytr, Xva, yva, LINEAR_ALPHAS, l1_ratio=0.5)
        en.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
        p_en = en.predict(Xte)
        fold_preds.append(("elasticnet", p_en))

        # ---- LightGBM (optional) ----
        if _HAS_LGBM:
            lgb_tr = lgb.Dataset(Xtr, label=ytr)
            lgb_va = lgb.Dataset(Xva, label=yva, reference=lgb_tr)
            params = dict(
                objective="regression",
                metric="rmse",
                learning_rate=0.05,
                num_leaves=63,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                min_data_in_leaf=100,
                seed=RANDOM_SEED,
                verbose=-1,
            )
            lgbm = lgb.train(
                params,
                lgb_tr,
                num_boost_round=5000,
                valid_sets=[lgb_tr, lgb_va],
                valid_names=["train","valid"],
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            p_lgb = lgbm.predict(Xte, num_iteration=lgbm.best_iteration)
            fold_preds.append(("lgbm", p_lgb))

            # Track importances
            imps = pd.DataFrame({
                "feature": feats,
                "importance": lgbm.feature_importance(importance_type="gain"),
                "ret_eom": pd.to_datetime(test_month),
            })
            lgbm_imps.append(imps)

        # ---- XGBoost (optional) ----
        if _HAS_XGB:
            xg_tr = xgb.DMatrix(Xtr, label=ytr)
            xg_va = xgb.DMatrix(Xva, label=yva)
            xg_te = xgb.DMatrix(Xte)

            params = dict(
                objective="reg:squarederror",
                eta=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                seed=RANDOM_SEED,
                eval_metric="rmse",
                nthread=0
            )
            watch = [(xg_tr, "train"), (xg_va, "valid")]
            xgbm = xgb.train(
                params,
                xg_tr,
                num_boost_round=5000,
                evals=watch,
                early_stopping_rounds=200,
                verbose_eval=False,
            )
            p_xgb = xgbm.predict(xg_te, iteration_range=(0, xgbm.best_iteration+1))
            fold_preds.append(("xgb", p_xgb))

            # Track importances
            imps = pd.DataFrame({
                "feature": feats,
                "importance": list(xgbm.get_score(importance_type="gain").values()) + [0]*(len(feats)-len(xgbm.get_score())),
                "ret_eom": pd.to_datetime(test_month),
            })
            xgb_imps.append(imps)

        # collect predictions for this month
        fold = df_te[["gvkey","ret_eom",target]].copy()
        for name, p in fold_preds:
            fold[f"pred_{name}"] = p
        preds_all.append(fold)

        print(f"[TRAIN] {str(test_month.date())} — done (n={len(fold)})")

    # Concatenate OOS predictions
    if not preds_all:
        raise RuntimeError("No predictions were produced. Check data coverage and parameters.")
    preds = pd.concat(preds_all, ignore_index=True)
    out_pred_fp = os.path.join(out_dir, "predictions_step04.csv")
    preds.to_csv(out_pred_fp, index=False)
    print(f"[TRAIN] Saved predictions -> {out_pred_fp}  | shape={preds.shape}")

    # Save feature importances (if any)
    if _HAS_LGBM and lgbm_imps:
        pd.concat(lgbm_imps, ignore_index=True).to_csv(os.path.join(out_dir, "feature_importance_lgbm.csv"), index=False)
    if _HAS_XGB and xgb_imps:
        pd.concat(xgb_imps, ignore_index=True).to_csv(os.path.join(out_dir, "feature_importance_xgb.csv"), index=False)

    # --------- OOS summary metrics ----------
    models = [c.replace("pred_","") for c in preds.columns if c.startswith("pred_")]
    rows = []
    for m in models:
        y = preds[target].values
        yhat = preds[f"pred_{m}"].values
        mask = ~np.isnan(y) & ~np.isnan(yhat)
        if mask.sum() == 0:
            continue
        r2 = r2_score(y[mask], yhat[mask])
        mse = mean_squared_error(y[mask], yhat[mask])
        df_m = preds[["ret_eom", target, f"pred_{m}"]].dropna()
        ic = _monthly_ic(df_m, f"pred_{m}", target)
        rows.append(dict(model=m, oos_r2=r2, oos_mse=mse, monthly_ic=ic))

    summ = pd.DataFrame(rows).sort_values(["monthly_ic","oos_r2"], ascending=[False, False])
    out_sum_fp = os.path.join(out_dir, "model_oos_summary_step04.csv")
    summ.to_csv(out_sum_fp, index=False)
    print(f"[TRAIN] Saved OOS summary -> {out_sum_fp}")
    print(summ.to_string(index=False))

if __name__ == "__main__":
    train_models("config.yaml")
