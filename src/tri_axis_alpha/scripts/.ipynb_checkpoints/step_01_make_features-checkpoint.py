from tri_axis_alpha.data.features_quant import build_quant_features
from tri_axis_alpha.data.features_text import build_text_features

if __name__ == "__main__":
    print("[step_01] Building QUANT features...")
    build_quant_features("config.yaml")
    #print("[step_01] Building TEXT features...")
    #build_text_features("config.yaml")
