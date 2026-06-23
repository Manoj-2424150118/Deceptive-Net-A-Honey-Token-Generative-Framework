# Deceptive-Net: Architecture & API Documentation

## System Architecture

```
Browser
  │
  ▼
┌────────────────────┐      ┌──────────────────────────┐
│  Nginx (port 8080) │      │  FastAPI (port 8000)     │
│  Static Frontend   │─────▶│  REST + WebSocket API    │
│  index.html        │      │  JWT Auth & RBAC         │
│  dashboard.html    │      │  Rate Limiting (slowapi)  │
└────────────────────┘      │  Audit Logger             │
                            │  ML Inference Module      │
                            └──────────┬───────────────┘
                                       │
                            ┌──────────▼───────────────┐
                            │  ML Models (ml/models/)   │
                            │  - LightGBM Classifier    │
                            │  - SimpleAutoencoder      │
                            │  - SHAP Explainer         │
                            │  - StandardScaler         │
                            └──────────────────────────┘
```

## ML Pipeline

### Dataset
Synthetic dataset (50,000 records) generated to mirror the NeurIPS 2022 Bank Account Fraud (BAF) dataset schema. Features include: transaction amounts, account age, device type, velocity, distance from home, and 15+ additional financial risk signals.

### Models
1. **LightGBM Classifier**: Primary fraud classifier. Handles class imbalance with `scale_pos_weight`. Trained with early stopping on validation set.
2. **SimpleAutoencoder** (sklearn MLPRegressor): Trained on normal transactions only. Reconstruction error > 95th percentile triggers anomaly flag.
3. **Ensemble**: `score = 0.70 * LightGBM_prob + 0.30 * AE_anomaly_score`

### Evaluation
- Primary metric: **PR-AUC** (Precision-Recall Area Under Curve)
- Secondary: **ROC-AUC**
- Rationale: Accuracy is meaningless for <1% fraud prevalence. PR-AUC captures the trade-off between catching fraud (recall) and avoiding false positives (precision).

## API Endpoints

### Authentication
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/token` | None | Login — returns JWT |
| GET | `/api/me` | Any | Current user info |

### Data
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/transactions?limit=N` | viewer+ | Transaction list (max 200) |
| GET | `/api/predict/{txn_id}` | analyst+ | ML fraud prediction |
| GET | `/api/explain/{txn_id}` | analyst+ | SHAP feature contributions |

### Admin
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/metrics` | admin | Model training metrics |
| GET | `/api/shap_importance` | admin | Global SHAP feature importance |
| GET | `/api/audit/logs?n=N` | admin | Recent audit log entries |
| GET | `/api/audit/alerts` | admin | Alert/Warn severity logs only |
| GET | `/api/genai_report` | admin | AI-generated threat analysis |

### Real-Time
| Protocol | Path | Auth | Description |
|----------|------|------|-------------|
| WebSocket | `/ws/transactions` | None* | Live transaction stream (3s interval) |

*WebSocket auth can be added via query param token in production.

## Roles

| Role | Access |
|------|--------|
| `admin` | All endpoints |
| `analyst` | Transactions, predictions, SHAP explanations |
| `viewer` | Transactions only |

## Setup

```bash
# Build and run
docker-compose up --build

# Access
# Dashboard:  http://localhost:8080
# API Docs:   http://localhost:8000/docs

# Default credentials
# admin   / admin123
# analyst / analyst123
# viewer  / viewer123
```
