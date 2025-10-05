"""
Advanced text feature engineering with GPU/CPU fallback and batching.
If GPU available -> FinBERT (finance-tuned)
If only CPU -> DistilBERT (faster)
"""
import os, re, glob, torch
import pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from .utils import load_config, ensure_dir

FINBERT = "yiyanghkust/finbert-tone"
LIGHT_MODEL = "distilbert-base-uncased"  # fallback if no GPU

BATCH = 16  # adjust if memory issues

def _count(text, vocab):
    return sum(1 for w in re.findall(r"[A-Za-z']+", str(text).lower()) if w in vocab)

def _readability(text):
    if not isinstance(text, str) or not text:
        return dict(fog=0, flesch=0)
    words = text.split()
    sents = max(text.count('.'), 1)
    syll = sum(len(re.findall(r'[aeiouyAEIOUY]+', w)) for w in words)
    fog = 0.4 * (len(words) / max(sents, 1) + 100 * sum(len(w) > 2 for w in words) / len(words))
    flesch = 206.835 - 1.015 * len(words) / max(sents, 1) - 84.6 * syll / max(len(words), 1)
    return dict(fog=fog, flesch=flesch)

def build_text_features(cfg_path: str, pca_dim=20):
    print("[TEXT] Loading config & lexicons...")
    cfg = load_config(cfg_path)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)
    txt_dir = cfg["data"]["text_dir"]

    # Load lexicons
    lm_dir = "data/raw/lexicons"
    def _load(name):
        print(f"    -> Loading lexicon {name}")
        return set(pd.read_csv(f"{lm_dir}/{name}.csv")["word"].str.lower())
    LM_POS, LM_NEG, LM_UNC, LM_LIT, LM_MOD = [
        _load(x) for x in ["LM_Positive", "LM_Negative", "LM_Uncertainty", "LM_Litigation", "LM_Modal"]
    ]

    # Model selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model_name = FINBERT
    else:
        print("[TEXT] GPU not found -> falling back to lightweight DistilBERT for speed.")
        model_name = LIGHT_MODEL

    print(f"[TEXT] Loading model: {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    feats = []
    files = sorted(glob.glob(os.path.join(txt_dir, "*.pkl")))
    print(f"[TEXT] Found {len(files)} yearly text files")
    for fp in tqdm(files, desc="Yearly Text Files"):
        print(f"  -> Processing file: {os.path.basename(fp)}")
        df = pd.read_pickle(fp)
        df.rename(columns={"date": "fdate"}, inplace=True)
        df["fdate"] = pd.to_datetime(df["fdate"], errors="coerce")

        # LM counts + readability
        for cat, vocab in {"pos": LM_POS, "neg": LM_NEG, "unc": LM_UNC, "lit": LM_LIT, "mod": LM_MOD}.items():
            df[f"lm_{cat}"] = df["mgmt"].apply(lambda x: _count(x, vocab))
        df[["fog", "flesch"]] = df["mgmt"].apply(_readability).apply(pd.Series)

        # Embeddings
        texts = (df["mgmt"].fillna("") + " " + df["rf"].fillna("")).tolist()
        if texts:
            print(f"    -> Encoding {len(texts)} filings")
            all_emb = []
            for i in range(0, len(texts), BATCH):
                batch = texts[i:i+BATCH]
                toks = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**toks).last_hidden_state.mean(1).cpu().numpy()
                all_emb.append(out)
            emb = np.vstack(all_emb)
            for i in range(emb.shape[1]):
                df[f"emb_{i}"] = emb[:, i]

        # Temporal diffs
        df = df.sort_values(["gvkey", "fdate"])
        for i in range(emb.shape[1]):
            if f"emb_{i}" in df:
                df[f"d_emb_{i}"] = df.groupby("gvkey")[f"emb_{i}"].diff()
        df["months_since"] = df.groupby("gvkey")["fdate"].diff().dt.days / 30
        feats.append(df)

    print("[TEXT] Concatenating and reducing embeddings with PCA...")
    all = pd.concat(feats, ignore_index=True)
    emb_cols = [c for c in all.columns if c.startswith("emb_")]
    if emb_cols:
        pca = PCA(n_components=min(pca_dim, len(emb_cols)))
        red = pca.fit_transform(all[emb_cols].fillna(0))
        for i in range(red.shape[1]):
            all[f"pca_emb_{i+1}"] = red[:, i]
        all.drop(columns=emb_cols, inplace=True)

    all["ret_eom"] = all["fdate"].values.astype("datetime64[M]") + pd.offsets.MonthEnd(0)
    agg = all.groupby(["gvkey", "ret_eom"]).mean(numeric_only=True).reset_index()
    out_fp = f"{out_dir}/text_features_advanced.csv"
    agg.to_csv(out_fp, index=False)
    print(f"[TEXT] Saved -> {out_fp} | shape={agg.shape}")
    return agg
