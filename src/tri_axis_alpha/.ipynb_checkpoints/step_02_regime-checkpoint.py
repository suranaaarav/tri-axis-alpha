from tri_axis_alpha.modeling.regime_features import build_regime_features
from tri_axis_alpha.modeling.regime_model import discover_and_predict_regimes

if __name__ == "__main__":
    build_regime_features("config.yaml")
    discover_and_predict_regimes("config.yaml", n_regimes=3)
