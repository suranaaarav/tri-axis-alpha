"""
Fast & memory-friendly text feature builder.
- Auto GPU/CPU model switch
- Token cap = 128 for speed
- Year-by-year caching
- Lexicon counts + readability + PCA compression
"""

import os, re, glob, torch
import pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from .utils import load_config, ensure_dir

# -----------------------------
# Settings
# -----------------------------
MODEL_GPU = "yiyanghkust/finbert-tone"          # FinBERT if GPU
MODEL_CPU = "sentence-transformers/all-MiniLM-L6-v2"  # fast on CPU
MAX_TOKENS = 128
BATCH_SIZE = 32
PCA_DIM = 12

# -----------------------------
# Helpers
# -----------------------------
def _count(text, vocab):
    if not isinstance(text, str): return 0
    return sum(1 for w in re.findall(r"[A-Za-z']+", text.lower()) if w in vocab)

def _readability(text):
    if not isinstance(text,str) or not text: return dict(fog=0,flesch=0)
    words=text.split(); sents=max(text.count('.'),1)
    syll=sum(len(re.findall(r'[aeiouyAEIOUY]+',w)) for w in words)
    fog=0.4*(len(words)/max(sents,1)+100*sum(len(w)>2 for w in words)/len(words))
    flesch=206.835-1.015*len(words)/max(sents,1)-84.6*syll/max(len(words),1)
    return dict(fog=fog,flesch=flesch)

def _embed_texts(model, tokenizer, texts, device):
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=MAX_TOKENS,
                         return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**toks)
            vec = out.last_hidden_state.mean(1).cpu()
        all_vecs.append(vec)
    return torch.cat(all_vecs,0).numpy()

# -----------------------------
# Main
# -----------------------------
def build_text_features(cfg_path:str):
    print("[TEXT] Loading config...")
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)
    txt_dir = cfg["data"]["text_dir"]

    # Load LM lexicons
    print("[TEXT] Loading LM lexicons...")
    lm_dir="data/raw/lexicons"
    def _load(name): 
        return set(pd.read_csv(f"{lm_dir}/{name}.csv")["word"].str.lower())
    LM = {
        "pos": _load("LM_Positive"),
        "neg": _load("LM_Negative"),
        "unc": _load("LM_Uncertainty"),
        "lit": _load("LM_Litigation"),
        "mod": _load("LM_Modal"),
    }

    # Model auto switch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = MODEL_GPU if device=="cuda" else MODEL_CPU
    print(f"[TEXT] Using model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    files = sorted(glob.glob(os.path.join(txt_dir,"*.pkl")))
    print(f"[TEXT] Found {len(files)} yearly text files")
    out_years = []

    tmp_dir = os.path.join(out_dir, "tmp_text")
    ensure_dir(tmp_dir)

    for fp in tqdm(files, desc="Yearly text files"):
        year = os.path.basename(fp).split('.')[0][-4:]
        out_fp_tmp = os.path.join(tmp_dir, f"text_{year}.feather")
        if os.path.exists(out_fp_tmp):
            print(f"  -> Skipping {year} (already processed)")
            out_years.append(pd.read_feather(out_fp_tmp))
            continue

        print(f"  -> Processing {year}")
        df = pd.read_pickle(fp)
        df.rename(columns={"date":"fdate"},inplace=True)
        df["fdate"]=pd.to_datetime(df["fdate"],errors="coerce")

        # Lexicon counts + readability
        for k,vocab in LM.items():
            df[f"lm_{k}"]=df["mgmt"].apply(lambda x:_count(x,vocab))
        df[["fog","flesch"]] = df["mgmt"].apply(_readability).apply(pd.Series)

        # Combine mgmt + rf text
        texts = (df["mgmt"].fillna("") + " " + df["rf"].fillna("")).tolist()
        if texts:
            print(f"    -> Embedding {len(texts)} filings")
            emb = _embed_texts(model, tokenizer, texts, device)
            for i in range(emb.shape[1]):
                df[f"emb_{i}"]=emb[:,i]

        # Temporal deltas
        df = df.sort_values(["gvkey","fdate"])
        emb_cols=[c for c in df.columns if c.startswith("emb_")]
        for col in emb_cols:
            df[f"d_{col}"]=df.groupby("gvkey")[col].diff()
        df["months_since"]=df.groupby("gvkey")["fdate"].diff().dt.days/30

        # Save partial year
        df.reset_index(drop=True).to_feather(out_fp_tmp)
        out_years.append(df)
        print(f"    -> Saved {out_fp_tmp}")

    print("[TEXT] Concatenating all years...")
    all_df = pd.concat(out_years, ignore_index=True)

    # PCA on embeddings
    emb_cols=[c for c in all_df.columns if c.startswith("emb_")]
    if emb_cols:
        print(f"[TEXT] Running PCA on {len(emb_cols)} emb dims -> {PCA_DIM}")
        pca=PCA(n_components=PCA_DIM)
        red=pca.fit_transform(all_df[emb_cols].fillna(0))
        for i in range(red.shape[1]):
            all_df[f"pca_emb_{i+1}"]=red[:,i]
        all_df.drop(columns=emb_cols,inplace=True)

    # Aggregate monthly
    all_df["ret_eom"]=all_df["fdate"].values.astype("datetime64[M]")+pd.offsets.MonthEnd(0)
    agg=all_df.groupby(["gvkey","ret_eom"]).mean(numeric_only=True).reset_index()

    out_fp = f"{out_dir}/text_features_advanced.csv"
    agg.to_csv(out_fp,index=False)
    print(f"[TEXT] Saved final -> {out_fp} | shape={agg.shape}")
    return agg

if __name__ == "__main__":
    build_text_features("config.yaml")
