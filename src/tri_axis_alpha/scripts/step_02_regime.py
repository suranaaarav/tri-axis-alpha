from tri_axis_alpha.models.regime_features_advanced import build_regime_features_advanced
from tri_axis_alpha.models.regime_model import fit_hmm_and_predict

if __name__ == "__main__":
    build_regime_features_advanced("config.yaml")
    fit_hmm_and_predict("config.yaml", n_states=3)