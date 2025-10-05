"""
Step 02b (Advanced): HMM regime discovery + expanding-window predictor
- Fit Hidden Markov Model (Gaussian) on regime features
- Order regimes by mean mkt_rf (bear..bull) and export posterior probabilities
- Train an expanding-window Gradient Boosting classifier to predict next-month regime
Outputs:
  - outputs/regime_labels.csv      (inferred class, mapped names)
  - outputs/regime_posteriors.csv  (posterior P_t(state))
  - outputs/regime_preds.csv       (predicted class/proba for t+1)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import GradientBoostingClassifier
from tri_axis_alpha.data.utils import load_config, ensure_dir
from pandas.tseries.offsets import MonthEnd


# -----------------------------
# Helpers
# -----------------------------
def _order_states_by_mkt_rf(ret_eom: pd.Series, mkt_rf: pd.Series, states: np.ndarray) -> dict[int,int]:
    tmp = pd.DataFrame({"state": states, "mkt_rf": mkt_rf.values, "ret_eom": ret_eom.values})
    mean_by_s = tmp.groupby("state")["mkt_rf"].mean().sort_values()  # low..high
    return {int(s): rank for rank, s in enumerate(mean_by_s.index)}

def _expanding_classifier_preds(df: pd.DataFrame, feat_cols: list[str], y_col: str, date_col: str):
    """
    Expanding-window classifier:
      - For each date (month) T, train on < T, predict on T (one step ahead when we shift)
    Returns dataframe with columns: [ret_eom, yhat_class, yhat_proba_k...]
    """
    out = []
    dates = sorted(df[date_col].unique())
    for i in range(24, len(dates)):  # burn-in 24 months
        train_dates = dates[:i]
        test_date   = dates[i]
        tr = df[df[date_col].isin(train_dates)].dropna(subset=[y_col])
        te = df[df[date_col] == test_date]
        if len(tr) < 100 or len(te) < 1:
            continue
        X_tr = tr[feat_cols].fillna(0).values
        y_tr = tr[y_col].astype(int).values
        X_te = te[feat_cols].fillna(0).values

        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        yhat = np.argmax(proba, axis=1)

        te_out = te[[date_col]].copy()
        te_out["yhat_class"] = yhat
        for k in range(proba.shape[1]):
            te_out[f"yhat_proba_{k}"] = proba[:, k]
        out.append(te_out)
    if out:
        return pd.concat(out, ignore_index=True)
    else:
        return pd.DataFrame(columns=[date_col,"yhat_class"])

# -----------------------------
# Main
# -----------------------------
def fit_hmm_and_predict(cfg_path: str, n_states: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)

    # Load features
    rf_path = f"{out_dir}/regime_features_adv.csv"
    rf = pd.read_csv(rf_path, parse_dates=["ret_eom"]).sort_values("ret_eom")
    feat_cols = [c for c in rf.columns if c not in ["ret_eom"]]

    # ---------- 1) HMM DISCOVERY ----------
    X = rf[feat_cols].fillna(0).values
    hmm = GaussianHMM(n_components=n_states, covariance_type="full", random_state=42, n_iter=200)
    hmm.fit(X)
    states = hmm.predict(X)

    # Posterior probabilities P_t(state)
    # (hmm.predict_proba is not in hmmlearn; compute via score_samples -> posteriors in .posterior)
    # Workaround: use hmm._compute_posteriors through score_samples
    # score_samples returns logprob, posteriors
    _, posteriors = hmm.score_samples(X)

    # Order & label states (bear..bull)
    ordering = _order_states_by_mkt_rf(rf["ret_eom"], rf["mkt_rf"], states)
    inv_map = {old:new for old,new in ordering.items()}
    states_ranked = np.array([ordering[int(s)] for s in states])

    name_map = {0:"bear", 1:"neutral", 2:"bull"} if n_states == 3 else {0:"bear", 1:"bull"}
    names = [name_map[min(s, max(name_map.keys()))] for s in states_ranked]

    labels = rf[["ret_eom"]].copy()
    labels["regime_inferred"] = states
    labels["regime_rank"] = states_ranked
    labels["regime_name"] = names
    labels.to_csv(f"{out_dir}/regime_labels.csv", index=False)
    print(f"[REGIME-HMM] labels -> {out_dir}/regime_labels.csv | {labels.shape}")

    # Posterior probs aligned and re-ordered to bear..bull
    post_df = rf[["ret_eom"]].copy()
    for s_old in range(n_states):
        s_new = ordering[s_old]
        post_df[f"regime_post_{s_new}"] = posteriors[:, s_old]
    post_df.to_csv(f"{out_dir}/regime_posteriors.csv", index=False)
    print(f"[REGIME-HMM] posteriors -> {out_dir}/regime_posteriors.csv | {post_df.shape}")

    # ---------- 2) NEXT-MONTH PREDICTION (EXPANDING) ----------
    # Features for classifier: the regime features themselves + posterior probs (soft persistence)
    cls_df = rf.merge(post_df, on="ret_eom", how="left").merge(labels[["ret_eom","regime_rank"]], on="ret_eom", how="left")
    # Predict y_{t+1}: shift target backward (so at time t we know next month's class)
    cls_df = cls_df.sort_values("ret_eom").reset_index(drop=True)
    cls_df["regime_next"] = cls_df["regime_rank"].shift(-1)

    # Candidate classifier features
    cls_feat_cols = [c for c in cls_df.columns if c not in ["ret_eom","regime_rank","regime_next","regime_inferred","regime_name"]]

    preds_oos = _expanding_classifier_preds(cls_df, cls_feat_cols, "regime_next", "ret_eom")
    # Shift predictions by +0 months because _expanding_classifier_preds already builds prediction for current month
    # We want the prediction made at t to be used for t+1; so shift predictions forward by 1 month when merging later
    preds_oos = preds_oos.sort_values("ret_eom").reset_index(drop=True)
    preds_oos.to_csv(f"{out_dir}/regime_preds_raw.csv", index=False)

    # Build a tidy next-month prediction table aligned to current month
    preds_aligned = preds_oos.copy()
    preds_aligned.columns = ["ret_eom"] + ["regime_pred_next_class"] + [f"regime_pred_next_proba_{k}" for k in range(n_states)]
    # NOTE: If you prefer alignment strictly as "prediction at t for t+1", you can shift here:
    preds_aligned["ret_eom"] = (preds_aligned["ret_eom"] + MonthEnd(1)).values
    preds_aligned.to_csv(f"{out_dir}/regime_preds.csv", index=False)
    print(f"[REGIME-HMM] preds -> {out_dir}/regime_preds.csv | {preds_aligned.shape}")

    return labels, post_df, preds_aligned

if __name__ == "__main__":
    fit_hmm_and_predict("config.yaml", n_states=3)
