"""
Step 06 – Evaluation & Reporting
Evaluates optimized and regime-aware portfolio outputs from Step 05.
"""

import os, io, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
from pandas.tseries.offsets import MonthEnd
from tri_axis_alpha.data.utils import load_config, ensure_dir

def _ann(series): return series.mean()*12, series.std()*np.sqrt(12)
def _sortino(series):
    neg = series[series<0]
    return np.nan if neg.std()==0 else series.mean()*12/(neg.std()*np.sqrt(12))
def _max_dd(series):
    cum = np.log1p(series).cumsum()
    return (cum - cum.cummax()).min()
def _nw_alpha(series, mkt):
    df = pd.DataFrame({"y":series, "mkt":mkt}).dropna()
    if len(df)<24: return np.nan, np.nan
    X=sm.add_constant(df["mkt"]); res=sm.OLS(df["y"],X).fit(cov_type="HAC",cov_kwds={"maxlags":3})
    return float(res.params.get("const",np.nan)), float(res.tvalues.get("const",np.nan))
def _turnover(wdf):
    if wdf.empty: return pd.Series(dtype=float)
    out=[]; prev=None
    for dt,g in wdf.groupby("ret_eom"):
        w=g.set_index("gvkey")["weight"]
        if prev is None: out.append((dt,np.nan))
        else: out.append((dt,0.5*float((w.reindex(prev.index,fill_value=0)-prev.reindex(w.index,fill_value=0)).abs().sum())))
        prev=w
    return pd.Series(dict(out)).sort_index()
def _img(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight",dpi=140); plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def evaluate(cfg_path:str="config.yaml"):
    cfg=load_config(cfg_path)
    out=cfg["output_dir"]; ensure_dir(out); figs=os.path.join(out,"figs"); ensure_dir(figs)

    # Market returns
    mkt=pd.read_csv(cfg["data"]["market_csv"],parse_dates=["ret_eom"]).set_index("ret_eom")["mkt_rf"]

    # Candidate series
    candidates=[]
    if os.path.exists(os.path.join(out,"opt_portfolio_returns.csv")):
        candidates.append(("OPT",os.path.join(out,"opt_portfolio_returns.csv"),"return"))
    for f in os.listdir(out):
        if f.startswith("portfolio_regime_"): candidates.append((f.replace("portfolio_regime_","REG_"),os.path.join(out,f),"return"))
        if f.startswith("portfolio_returns_"): candidates.append((f.replace("portfolio_returns_","SIMPLE_"),os.path.join(out,f),"long_short"))

    weights=pd.DataFrame()
    if os.path.exists(os.path.join(out,"opt_portfolio_weights.csv")):
        weights=pd.read_csv(os.path.join(out,"opt_portfolio_weights.csv"),parse_dates=["ret_eom"])
        turnover=_turnover(weights)
    else: turnover=pd.Series(dtype=float)

    summary=[]
    for name,fp,col in candidates:
        df=pd.read_csv(fp,parse_dates=["ret_eom"]).set_index("ret_eom").sort_index()
        if col not in df: col=df.select_dtypes(include=[np.number]).columns[-1]
        s=df[col].dropna()
        ar,av=_ann(s); sharpe=ar/(av+1e-9); sortino=_sortino(s); mdd=_max_dd(s)
        alpha,t=_nw_alpha(s,mkt.reindex(s.index))
        sk=skew(s, nan_policy='omit'); ku=kurtosis(s,nan_policy='omit')
        to=turnover.reindex(s.index).mean() if name=="OPT" and not turnover.empty else np.nan
        summary.append(dict(strategy=name,start=str(s.index.min().date()),end=str(s.index.max().date()),months=len(s),
                            ann_ret=ar,ann_vol=av,sharpe=sharpe,sortino=sortino,max_dd=mdd,
                            alpha=alpha,alpha_t=t,hit=(s>0).mean(),skew=sk,kurtosis=ku,avg_turnover=to))
        # plots
        cum=np.exp(np.log1p(s).cumsum())
        plt.figure(figsize=(8,4)); plt.plot(cum); plt.title(f"{name} cumulative"); plt.tight_layout()
        plt.savefig(os.path.join(figs,f"{name}_cum.png")); plt.close()

        dd=(np.log1p(s).cumsum()-np.log1p(s).cumsum().cummax())
        plt.figure(figsize=(8,3)); plt.plot(dd); plt.title(f"{name} drawdown"); plt.tight_layout()
        plt.savefig(os.path.join(figs,f"{name}_dd.png")); plt.close()

    pd.DataFrame(summary).to_csv(os.path.join(out,"eval_summary.csv"),index=False)
    print(f"[EVAL] Saved eval_summary.csv with {len(summary)} strategies")
    print(f"[EVAL] Plots in {figs}")
