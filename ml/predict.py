"""
Deceptive-Net – Inference Module
==================================
Provides:
  - predict_fraud_prob()    → LightGBM fraud probability [0,1]
  - predict_anomaly_score() → Autoencoder anomaly score [0,1]
  - combined_score()        → Ensemble: 0.7*lgbm + 0.3*ae
  - explain_prediction()    → Top-5 features by permutation importance
  - get_model_metrics()     → Training-time performance metrics
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

_BASE         = os.path.dirname(__file__)
_LGBM_PATH    = os.path.join(_BASE, "models", "lgbm_classifier.pkl")
_SCALER_PATH  = os.path.join(_BASE, "models", "scaler.pkl")
_AE_PATH      = os.path.join(_BASE, "models", "autoencoder.pkl")
_IMP_PATH     = os.path.join(_BASE, "models", "shap_values.json")
_FEAT_PATH    = os.path.join(_BASE, "models", "feature_names.json")
_METRICS_PATH = os.path.join(_BASE, "models", "metrics.json")

_lgbm_model    = None
_scaler        = None
_ae_model      = None
_feature_names = None


def _load_all():
    global _lgbm_model, _scaler, _ae_model, _feature_names
    if _lgbm_model is None:
        if not os.path.exists(_LGBM_PATH):
            raise FileNotFoundError("Models not found. Run ml/train_model.py first.")
        _lgbm_model    = joblib.load(_LGBM_PATH)
        _scaler        = joblib.load(_SCALER_PATH)
        _ae_model      = joblib.load(_AE_PATH)
        with open(_FEAT_PATH) as f:
            _feature_names = json.load(f)


def _to_scaled(transaction_data: dict) -> tuple:
    """Returns (scaled_df, scaled_np_array)."""
    _load_all()
    row = {k: transaction_data.get(k, 0) for k in _feature_names}
    df  = pd.DataFrame([row], columns=_feature_names)
    arr = _scaler.transform(df)
    return pd.DataFrame(arr, columns=_feature_names), arr


def predict_fraud_prob(transaction_data: dict) -> float:
    df, _ = _to_scaled(transaction_data)
    return float(_lgbm_model.predict_proba(df)[0][1])


def predict_anomaly_score(transaction_data: dict) -> float:
    _, arr = _to_scaled(transaction_data)
    return float(_ae_model.anomaly_score(arr)[0])


def combined_score(transaction_data: dict) -> float:
    lgbm = predict_fraud_prob(transaction_data)
    ae   = predict_anomaly_score(transaction_data)
    return round(0.70 * lgbm + 0.30 * ae, 4)


def explain_prediction(transaction_data: dict) -> dict:
    """
    Returns top-5 feature contributions using global permutation importance
    scaled by the current transaction's absolute feature values.
    """
    _load_all()
    if not os.path.exists(_IMP_PATH):
        return {}
    with open(_IMP_PATH) as f:
        global_imp = json.load(f)

    df_scaled, _ = _to_scaled(transaction_data)
    feat_vals    = df_scaled.iloc[0].to_dict()

    local = {
        feat: round(abs(feat_vals.get(feat, 0)) * imp, 5)
        for feat, imp in global_imp.items()
    }
    top5 = sorted(local.items(), key=lambda x: x[1], reverse=True)[:5]
    return dict(top5)


def get_model_metrics() -> dict:
    if not os.path.exists(_METRICS_PATH):
        return {}
    with open(_METRICS_PATH) as f:
        return json.load(f)
