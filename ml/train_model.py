"""
Deceptive-Net – ML Training Pipeline
=====================================
Models trained:
  1. LightGBM classifier  (primary fraud classifier)
  2. SimpleAutoencoder     (anomaly-detection layer via sklearn MLPRegressor)

SHAP replaced with lightweight permutation-based feature importance
to avoid the numba/llvmlite 56 MB dependency.

Evaluation metrics: PR-AUC, ROC-AUC, F1, Precision, Recall
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LGBM_PATH          = os.path.join(MODEL_DIR, "lgbm_classifier.pkl")
SCALER_PATH        = os.path.join(MODEL_DIR, "scaler.pkl")
AE_PATH            = os.path.join(MODEL_DIR, "autoencoder.pkl")
METRICS_PATH       = os.path.join(MODEL_DIR, "metrics.json")
FEAT_IMP_PATH      = os.path.join(MODEL_DIR, "shap_values.json")   # reused key name for API compat
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

# ── feature schema ────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "transaction_amount",
    "user_age",
    "account_age_days",
    "device_type",
    "distance_from_home",
    "num_transactions_24h",
    "avg_txn_amount_7d",
    "failed_attempts",
    "is_foreign_transaction",
    "hour_of_day",
    "day_of_week",
    "credit_score",
    "monthly_income",
    "num_cards",
    "email_is_free",
    "phone_mobile",
    "has_chip",
    "pin_changed_recently",
    "velocity_change",
    "merchant_risk_score",
]


# ── synthetic data ────────────────────────────────────────────────────────────
def generate_synthetic_data(num_samples: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = num_samples
    data = {
        "transaction_amount":      rng.exponential(scale=120, size=n),
        "user_age":                rng.integers(18, 80, size=n),
        "account_age_days":        rng.integers(1, 3650, size=n),
        "device_type":             rng.choice([0, 1, 2], size=n, p=[0.65, 0.25, 0.10]),
        "distance_from_home":      rng.lognormal(mean=2.0, sigma=1.2, size=n),
        "num_transactions_24h":    rng.integers(1, 50, size=n),
        "avg_txn_amount_7d":       rng.exponential(scale=80, size=n),
        "failed_attempts":         rng.integers(0, 5, size=n),
        "is_foreign_transaction":  rng.integers(0, 2, size=n),
        "hour_of_day":             rng.integers(0, 24, size=n),
        "day_of_week":             rng.integers(0, 7, size=n),
        "credit_score":            rng.integers(300, 850, size=n),
        "monthly_income":          rng.lognormal(mean=8.5, sigma=0.6, size=n),
        "num_cards":               rng.integers(1, 8, size=n),
        "email_is_free":           rng.integers(0, 2, size=n),
        "phone_mobile":            rng.integers(0, 2, size=n),
        "has_chip":                rng.integers(0, 2, size=n),
        "pin_changed_recently":    rng.integers(0, 2, size=n),
        "velocity_change":         rng.exponential(scale=0.3, size=n),
        "merchant_risk_score":     rng.uniform(0, 1, size=n),
    }
    df = pd.DataFrame(data)

    score = (
        0.25 * (df["transaction_amount"] / df["transaction_amount"].max()) +
        0.15 * (1 - df["account_age_days"] / df["account_age_days"].max()) +
        0.15 * (df["distance_from_home"] / df["distance_from_home"].max()) +
        0.10 * (df["failed_attempts"] / df["failed_attempts"].max()) +
        0.10 * df["is_foreign_transaction"] +
        0.10 * df["velocity_change"] / df["velocity_change"].max() +
        0.10 * df["merchant_risk_score"] +
        0.05 * df["email_is_free"]
    )
    score += rng.normal(0, 0.08, size=n)
    threshold = np.percentile(score, 99)
    df["is_fraud"] = (score > threshold).astype(int)
    return df


# ── LightGBM ──────────────────────────────────────────────────────────────────
def train_lgbm(X_train, y_train, X_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = {
        "objective":         "binary",
        "metric":            ["binary_logloss", "auc"],
        "n_estimators":      500,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "scale_pos_weight":  scale_pos_weight,
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(int)
    metrics = {
        "roc_auc": round(roc_auc_score(y_val, probs), 4),
        "pr_auc":  round(average_precision_score(y_val, probs), 4),
        "report":  classification_report(y_val, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
    }
    return model, metrics


# ── Autoencoder ───────────────────────────────────────────────────────────────
class SimpleAutoencoder:
    """
    Reconstruction-error based anomaly detector.
    Uses sklearn MLPRegressor — no numba/llvmlite dependency.
    """
    def __init__(self, encoding_dim: int = 10):
        from sklearn.neural_network import MLPRegressor
        self.ae = MLPRegressor(
            hidden_layer_sizes=(encoding_dim,),
            max_iter=200,
            random_state=42,
            verbose=False,
        )
        self.threshold: float = 0.0

    def fit(self, X_normal: np.ndarray):
        self.ae.fit(X_normal, X_normal)
        errors = self._errors(X_normal)
        self.threshold = float(np.percentile(errors, 95))
        return self

    def _errors(self, X: np.ndarray) -> np.ndarray:
        return np.mean((X - self.ae.predict(X)) ** 2, axis=1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        e   = self._errors(X)
        cap = max(e.max(), self.threshold * 2, 1e-9)
        return np.clip(e / cap, 0, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._errors(X) > self.threshold).astype(int)


# ── Permutation feature importance (SHAP-replacement) ─────────────────────────
def compute_feature_importance(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Compute sklearn permutation importance as lightweight SHAP alternative.
    Returns dict: {feature: normalised_importance}
    """
    from sklearn.inspection import permutation_importance as perm_imp
    from sklearn.metrics import roc_auc_score as auc_score

    result = perm_imp(model, X_val, y_val, n_repeats=5, random_state=42,
                      scoring="roc_auc", n_jobs=-1)
    imps = result.importances_mean
    # Normalise to [0, 1]
    max_imp = max(imps.max(), 1e-9)
    return {
        name: round(float(max(imp, 0) / max_imp), 5)
        for name, imp in zip(FEATURE_NAMES, imps)
    }


# ── main ──────────────────────────────────────────────────────────────────────
def train_and_save():
    print("=" * 60)
    print("  Deceptive-Net – Model Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Generating synthetic dataset (50,000 records)...")
    df = generate_synthetic_data()
    print(f"      Fraud prevalence: {df['is_fraud'].mean()*100:.2f}%")

    X = df[FEATURE_NAMES]
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n[2/5] Scaling features...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    X_tr_df = pd.DataFrame(X_tr_s, columns=FEATURE_NAMES)
    X_te_df = pd.DataFrame(X_te_s, columns=FEATURE_NAMES)

    print("\n[3/5] Training LightGBM classifier...")
    lgbm_model, lgbm_metrics = train_lgbm(X_tr_df, y_train, X_te_df, y_test)
    joblib.dump(lgbm_model, LGBM_PATH)
    print(f"      ROC-AUC: {lgbm_metrics['roc_auc']}  |  PR-AUC: {lgbm_metrics['pr_auc']}")

    print("\n[4/5] Training Autoencoder (anomaly detection)...")
    X_tr_normal = X_tr_s[y_train.values == 0]
    ae = SimpleAutoencoder(encoding_dim=10)
    ae.fit(X_tr_normal)
    ae_roc = roc_auc_score(y_test, ae.anomaly_score(X_te_s))
    joblib.dump(ae, AE_PATH)
    lgbm_metrics["ae_roc_auc"] = round(ae_roc, 4)
    print(f"      AE ROC-AUC: {ae_roc:.4f}")

    print("\n[5/5] Computing feature importance (permutation-based)...")
    feat_imp = compute_feature_importance(lgbm_model, X_te_df, y_test)
    with open(FEAT_IMP_PATH, "w") as f:
        json.dump(feat_imp, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(lgbm_metrics, f, indent=2)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f)

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  LightGBM  ROC-AUC : {lgbm_metrics['roc_auc']}")
    print(f"  LightGBM  PR-AUC  : {lgbm_metrics['pr_auc']}")
    print(f"  Autoencoder ROC   : {lgbm_metrics['ae_roc_auc']}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()
