import os, yaml

def load_config(cfg_path:str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)
