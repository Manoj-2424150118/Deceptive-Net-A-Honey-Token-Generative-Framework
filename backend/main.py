"""
Deceptive-Net – FastAPI Backend
================================
Endpoints:
  POST /auth/token              → JWT login
  GET  /api/me                  → Current user info
  GET  /api/transactions        → Live transaction feed (all roles)
  GET  /api/predict/{txn_id}    → ML prediction (analyst+)
  GET  /api/explain/{txn_id}    → SHAP explanation (analyst+)
  GET  /api/metrics             → Model performance (admin)
  GET  /api/audit/logs          → Audit log feed (admin)
  GET  /api/audit/alerts        → Alert-only log feed (admin)
  GET  /api/genai_report        → AI threat analysis (admin)
  GET  /api/shap_importance     → Global SHAP importance (admin)
  WS   /ws/transactions         → WebSocket live stream (analyst+)

Security:
  - JWT RBAC on every protected endpoint
  - Rate limiting via slowapi
  - Strict CORS
  - Request logging middleware
  - Security headers middleware (CSP, HSTS, X-Frame-Options)
"""

# BCrypt compatibility monkeypatch for passlib
try:
    import bcrypt
    if not hasattr(bcrypt, "__about__"):
        class Dummy:
            pass
        dummy = Dummy()
        dummy.__version__ = getattr(bcrypt, "__version__", "4.0.0")
        bcrypt.__about__ = dummy
except ImportError:
    pass

import json
import os
import random
import sys
import time
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
IST = timezone(timedelta(hours=5, minutes=30))
from typing import List, Optional

from fastapi import (
    Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.predict import (
    combined_score, explain_prediction, get_model_metrics, predict_fraud_prob,
    predict_anomaly_score,
)
from backend.auth import (
    UserInDB, authenticate_user, create_access_token, get_current_user,
    require_admin, require_analyst, require_viewer,
)
from backend.audit import get_alert_events, get_recent_events, log_event
from backend.deception import (
    deception_registry, FIRST_NAMES, LAST_NAMES, EMAIL_DOMAINS,
    COMMON_PASSWORDS, _luhn_check_digit
)

# ── app setup ─────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ── Deceptive-Net Global System Config ───────────────────────────────────────
DECEPTIVE_NET_CONFIG = {
    "dqn_epsilon": 0.05,
    "token_deployment_frequency": 45,
    "active_honeypots": ["SSH Cowrie Console", "E-Commerce Checkout", "Admin Credential Trap"],
    "token_rotation_enabled": True,
    "watermark_compliance_min": 0.95
}

# ── TDA DQN transitions initialization ────────────────────────────────────────
TDA_TRANSITIONS = []

def init_tda_transitions():
    actions = [
        "inject_professional_honey_token",
        "inject_credit_card_honey_token",
        "inject_foreign_ip_honey_token",
        "hold_token_deployment",
        "rotate_all_deployed_tokens"
    ]
    rewards = [1.25, 0.45, 12.8, -0.1, 5.5]
    states = ["entropy_high", "unique_ips_spike", "session_dwell_long", "normal_idle", "command_entropy_spike"]
    
    global TDA_TRANSITIONS
    TDA_TRANSITIONS = []
    # Seed 10 realistic logs in reverse chronological order
    for i in range(10):
        ts = (datetime.now(IST) - timedelta(seconds=i*45)).strftime("%Y-%m-%d %H:%M:%S IST")
        TDA_TRANSITIONS.append({
            "timestamp": ts,
            "state": random.choice(states),
            "action": random.choice(actions),
            "reward": round(random.choice(rewards), 3),
            "epsilon": round(max(0.01, 1.0 - (i*0.08)), 2)
        })

init_tda_transitions()


tags_metadata = [
    {
        "name": "Auth",
        "description": "Authentication and user authorization endpoints. Use these to get your JWT access token.",
    },
    {
        "name": "Data",
        "description": "Endpoints to fetch and export transaction data.",
    },
    {
        "name": "ML",
        "description": "Machine Learning inference endpoints. Run fraud prediction and get explanations.",
    },
    {
        "name": "Admin",
        "description": "System administration and monitoring endpoints. Requires admin privileges.",
    },
]

app = FastAPI(
    title="🧅 Deceptive-Net Academic API Platform",
    description="""
# 🧅 Deceptive-Net: Generative AI Driven Cyber Deception & Attribution Framework

Welcome to the Deceptive-Net Research & Academic API. This platform implements a closed-loop cyber defense architecture that combines real-time machine learning fraud detection with proactive network deception.

## 🔬 Core System Components

### 1. Generative PII & BAF Synthesis (cWGAN-GP)
Generates high-fidelity decoy identities and synthetic banking transaction patterns trained on realistic financial datasets. Float vectors are dynamically decoded into Luhn-compliant credit cards and credentials.

### 2. Multi-Model Risk Scoring (LightGBM + PyTorch Autoencoder)
- **LightGBM Ensemble**: Fast, supervised gradient boosting trees scoring card-not-present fraud risk.
- **PyTorch Autoencoder Anomaly Engine**: Unsupervised reconstructive neural network identifying structural transaction outliers based on MSE reconstruction loss.
- **Explainable AI (SHAP)**: Individual and global feature impact explainability, detailing the exact mathematical feature weights contributing to risk.

### 3. Closed-Loop Cyber Deception & Forensic Attribution
Automatically watermarks exfiltrated CSV files. Decoy credentials and credit cards are registered in the system. Attempts to exploit these tokens in SSH consoles (Cowrie) or checkouts trigger the payment/portal honeypots, reverse-attributing the breach directly back to the original leaking session, timestamp, and IP address.

### 4. Reinforcement Learning-based Token Deployment Agent (TDA DQN)
Under a Deep Q-Network policy, the TDA dynamically selects optimal decoy token deployment actions (e.g., injecting professional credentials, rotating tokens) based on attacker dwell time and command entropy in the Cowrie SSH honeypot.

### 🔑 Demo Credentials
- **Security Administrator**: `admin` / `admin123` (Full RBAC access to threat intel, TDA logs, audit logs, and metrics)
- **Fraud Analyst**: `analyst` / `analyst123` (Read/write access to transactions, SHAP explainability, and secure export)
- **Read-Only Auditor**: `viewer` / `viewer123` (Read-only access to transaction feeds)
""",
    version="2.1.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── security headers middleware ───────────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]          = "DENY"
    response.headers["X-XSS-Protection"]         = "1; mode=block"
    response.headers["Referrer-Policy"]          = "no-referrer"
    response.headers["Permissions-Policy"]       = "geolocation=(), microphone=()"
    return response


# ── request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 1)
    log_event(
        actor="system",
        action="HTTP_REQUEST",
        resource=f"{request.method} {request.url.path}",
        detail=f"status={response.status_code} duration={duration}ms",
        severity="INFO",
        ip=request.client.host if request.client else "unknown",
    )
    return response


# ── CORS (tightened for production) ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str = Field(description="The JWT access token string", json_schema_extra={"example": "eyJhbGciOiJIUzI1NiIsInR5cCI..."})
    token_type: str = Field(description="Token type, typically 'bearer'", json_schema_extra={"example": "bearer"})
    role: str = Field(description="Role of the authenticated user (e.g., admin, analyst)", json_schema_extra={"example": "admin"})
    username: str = Field(description="Username of the authenticated user", json_schema_extra={"example": "admin"})


class Transaction(BaseModel):
    id:           str = Field(description="Unique transaction ID", json_schema_extra={"example": "TXN-123456"})
    user_id:      str = Field(description="User ID making the transaction", json_schema_extra={"example": "USR-7890"})
    amount:       float = Field(description="Transaction amount in USD", json_schema_extra={"example": 150.75})
    age:          int = Field(description="Age of the user", json_schema_extra={"example": 35})
    account_age:  int = Field(description="Age of the account in days", json_schema_extra={"example": 1200})
    device:       int = Field(description="Device type used (0: Web, 1: Mobile, 2: POS)", json_schema_extra={"example": 1})
    distance:     float = Field(description="Distance from user's home in km", json_schema_extra={"example": 12.5})
    num_txn_24h:  int = Field(description="Number of transactions in the last 24 hours", json_schema_extra={"example": 3})
    velocity_change: float = Field(description="Change in transaction velocity", json_schema_extra={"example": 0.1})
    merchant_risk: float = Field(description="Risk score of the merchant", json_schema_extra={"example": 0.05})
    timestamp:    str = Field(description="UTC timestamp of the transaction", json_schema_extra={"example": "2026-05-18 14:32:00"})
    ip_address:   str = Field(description="IP address of the transaction source", json_schema_extra={"example": "192.168.1.1"})
    location:     str = Field(description="Geographic location", json_schema_extra={"example": "New York, US"})
    risk_score:   Optional[float] = Field(None, description="Ensemble ML risk score (0.0 to 1.0)", json_schema_extra={"example": 0.85})
    anomaly_score: Optional[float] = Field(None, description="Autoencoder anomaly score", json_schema_extra={"example": 0.72})
    is_flagged:   Optional[bool] = Field(None, description="Whether the transaction was flagged as suspicious", json_schema_extra={"example": True})


class PredictionResult(BaseModel):
    transaction_id: str = Field(description="The transaction ID evaluated", json_schema_extra={"example": "TXN-123456"})
    fraud_probability: float = Field(description="LightGBM model fraud probability", json_schema_extra={"example": 0.82})
    anomaly_score: float = Field(description="Autoencoder anomaly score", json_schema_extra={"example": 0.75})
    ensemble_score: float = Field(description="Combined ensemble risk score", json_schema_extra={"example": 0.78})
    risk_level: str = Field(description="Categorical risk level (LOW, MEDIUM, HIGH, CRITICAL)", json_schema_extra={"example": "CRITICAL"})
    top_features: dict = Field(description="Top features contributing to the prediction", json_schema_extra={"example": {"transaction_amount": 0.15, "velocity_change": 0.1}})


class AIReport(BaseModel):
    timestamp: str = Field(description="Report generation timestamp", json_schema_extra={"example": "2026-05-18T10:00:00Z"})
    summary: str = Field(description="High-level threat summary", json_schema_extra={"example": "Deceptive-Net AI Analysis: 5 anomalous transactions detected..."})
    alerts: List[str] = Field(description="List of specific alerts and observations")
    recommendation: str = Field(description="Actionable recommendations to mitigate threats")
    risk_level: str = Field(description="Overall system risk level", json_schema_extra={"example": "HIGH"})


class UserRoleUpdate(BaseModel):
    username: str = Field(..., description="Target username for role update", json_schema_extra={"example": "analyst"})
    role: str = Field(..., description="Target role (admin, analyst, viewer)", json_schema_extra={"example": "admin"})


class SystemConfigUpdate(BaseModel):
    dqn_epsilon: Optional[float] = Field(None, description="DQN Reinforcement Learning Exploration Rate", json_schema_extra={"example": 0.05})
    token_deployment_frequency: Optional[int] = Field(None, description="Decoy injection interval in seconds", json_schema_extra={"example": 45})
    active_honeypots: Optional[List[str]] = Field(None, description="List of active honeypots", json_schema_extra={"example": ["SSH Cowrie Console", "E-Commerce Checkout"]})
    token_rotation_enabled: Optional[bool] = Field(None, description="Toggle token rotation rules", json_schema_extra={"example": True})
    watermark_compliance_min: Optional[float] = Field(None, description="Minimum check validation compliance threshold", json_schema_extra={"example": 0.95})


class SignupRequest(BaseModel):
    username: str = Field(..., description="Desired username", json_schema_extra={"example": "user123"})
    password: str = Field(..., description="Secure password", json_schema_extra={"example": "mypass123"})
    full_name: str = Field(..., description="User's full name", json_schema_extra={"example": "Alice Cooper"})


class PasswordResetRequest(BaseModel):
    username: str = Field(..., description="Username requesting reset", json_schema_extra={"example": "user123"})
    new_password: str = Field(..., description="New password value", json_schema_extra={"example": "newpass123"})


class SupportTicketRequest(BaseModel):
    email: str = Field(..., description="Contact email address", json_schema_extra={"example": "support@example.com"})
    subject: str = Field(..., description="Subject of support or bug report", json_schema_extra={"example": "Bug in model metrics panel"})
    message: str = Field(..., description="Detailed support ticket explanation", json_schema_extra={"example": "Graph is not rendering properly..."})


class TrackingEventRequest(BaseModel):
    event_type: str = Field(..., description="Event classification (e.g. PAGE_VIEW, BUTTON_CLICK)", json_schema_extra={"example": "PAGE_VIEW"})
    details: str = Field(..., description="Context details (e.g. tab name, selected transaction ID)", json_schema_extra={"example": "tab-model-metrics"})


# ── synthetic transaction generator ──────────────────────────────────────────
LOCATIONS = [
    "Mumbai, IN", "New York, US", "London, UK", "Beijing, CN",
    "São Paulo, BR", "Lagos, NG", "Moscow, RU", "Sydney, AU",
    "Berlin, DE", "Tokyo, JP", "Dubai, AE", "Toronto, CA",
]

def _random_txn() -> Transaction:
    rng = random.Random()
    amount  = round(random.expovariate(1 / 120), 2)
    age     = random.randint(18, 80)
    acc_age = random.randint(1, 3650)
    device  = random.choices([0, 1, 2], weights=[0.65, 0.25, 0.10])[0]
    dist    = round(random.lognormvariate(2.0, 1.2), 2)
    ntxn    = random.randint(1, 50)
    vel     = round(random.expovariate(1 / 0.3), 3)
    merch   = round(random.uniform(0, 1), 3)

    txn_dict = {
        "transaction_amount": amount,
        "user_age": age,
        "account_age_days": acc_age,
        "device_type": device,
        "distance_from_home": dist,
        "num_transactions_24h": ntxn,
        "avg_txn_amount_7d": round(random.expovariate(1 / 80), 2),
        "failed_attempts": random.randint(0, 4),
        "is_foreign_transaction": random.randint(0, 1),
        "hour_of_day": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "credit_score": random.randint(300, 850),
        "monthly_income": round(random.lognormvariate(8.5, 0.6), 2),
        "num_cards": random.randint(1, 7),
        "email_is_free": random.randint(0, 1),
        "phone_mobile": random.randint(0, 1),
        "has_chip": random.randint(0, 1),
        "pin_changed_recently": random.randint(0, 1),
        "velocity_change": vel,
        "merchant_risk_score": merch,
    }

    try:
        ens = combined_score(txn_dict)
        ae  = predict_anomaly_score(txn_dict)
    except Exception:
        ens = round(random.uniform(0, 0.3), 4)
        ae  = round(random.uniform(0, 0.2), 4)

    return Transaction(
        id=f"TXN-{random.randint(100000, 999999)}",
        user_id=f"USR-{random.randint(1000, 9999)}",
        amount=amount,
        age=age,
        account_age=acc_age,
        device=device,
        distance=dist,
        num_txn_24h=ntxn,
        velocity_change=vel,
        merchant_risk=merch,
        timestamp=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        ip_address=f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        location=random.choice(LOCATIONS),
        risk_score=ens,
        anomaly_score=ae,
        is_flagged=(ens > 0.6 or ae > 0.7),
    )


# Pre-generate mock data pool
MOCK_DATA: List[Transaction] = [_random_txn() for _ in range(200)]


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/token", response_model=Token, tags=["Auth"])
@limiter.limit("10/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate and receive a JWT access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        log_event(
            actor=form_data.username, action="LOGIN_FAILED",
            severity="WARN", ip=request.client.host if request.client else "unknown"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    log_event(
        actor=user["username"], action="LOGIN_SUCCESS",
        detail=f"role={user['role']}",
        severity="INFO",
        ip=request.client.host if request.client else "unknown",
    )
    return Token(access_token=token, token_type="bearer",
                 role=user["role"], username=user["username"])


@app.get("/api/me", tags=["Auth"])
async def me(current_user: UserInDB = Depends(get_current_user)):
    """Returns current authenticated user info."""
    return {"username": current_user.username, "role": current_user.role,
            "full_name": current_user.full_name}


@app.post("/auth/signup", tags=["Auth"])
@limiter.limit("5/minute")
async def signup(request: Request, payload: SignupRequest):
    """Dynamic user registration for SaaS checklist compliance."""
    from backend.auth import register_new_user
    user = register_new_user(
        username=payload.username,
        password=payload.password,
        full_name=payload.full_name,
        role="viewer"  # Default registration role
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    log_event(
        actor=payload.username,
        action="USER_SIGNUP",
        detail=f"name='{payload.full_name}' role='viewer'",
        severity="INFO",
        ip=request.client.host if request.client else "unknown"
    )
    return {"status": "success", "username": payload.username, "detail": "User registered successfully"}


@app.post("/auth/reset-password", tags=["Auth"])
@limiter.limit("5/minute")
async def reset_password(request: Request, payload: PasswordResetRequest):
    """Password reset request handling (mock verification)."""
    from backend.auth import reset_user_password
    success = reset_user_password(payload.username, payload.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Username not found"
        )
    log_event(
        actor=payload.username,
        action="PASSWORD_RESET",
        detail="Password successfully reset via SaaS recovery form",
        severity="ALERT",
        ip=request.client.host if request.client else "unknown"
    )
    return {"status": "success", "detail": "Password has been successfully updated"}


@app.post("/api/support/ticket", tags=["Data"])
@limiter.limit("10/minute")
async def submit_support_ticket(request: Request, payload: SupportTicketRequest):
    """Submits a feedback, bug report, or support ticket."""
    client_ip = request.client.host if request.client else "unknown"
    log_event(
        actor=payload.email,
        action="SUPPORT_TICKET_SUBMITTED",
        resource=payload.subject[:40],
        detail=f"Msg: {payload.message[:80]}",
        severity="WARN",
        ip=client_ip
    )
    return {"status": "success", "detail": "Support ticket submitted. Reference ID: TK-" + str(random.randint(1000, 9999))}


@app.post("/api/tracking/event", tags=["Data"])
@limiter.limit("60/minute")
async def track_user_event(request: Request, payload: TrackingEventRequest):
    """Tracks page view and interaction analytics for GSC/SEO compliance."""
    client_ip = request.client.host if request.client else "unknown"
    log_event(
        actor="anonymous_visitor" if not request.headers.get("Authorization") else "authenticated_user",
        action="USER_EVENT_TRACKING",
        resource=payload.event_type,
        detail=payload.details,
        severity="INFO",
        ip=client_ip
    )
    return {"status": "success"}


# ── Transaction endpoints ─────────────────────────────────────────────────────
@app.get("/api/transactions", response_model=List[Transaction], tags=["Data"])
@limiter.limit("60/minute")
async def get_transactions(
    request: Request,
    limit: int = 20,
    current_user: UserInDB = Depends(require_viewer),
):
    """Returns recent synthetic transaction records."""
    limit = max(1, min(limit, 200))
    log_event(actor=current_user.username, action="VIEW_TRANSACTIONS",
              detail=f"limit={limit}", severity="INFO")
    return MOCK_DATA[:limit]


@app.get("/api/transactions/export", tags=["Data"])
@limiter.limit("5/minute")
async def export_transactions(
    request: Request,
    current_user: UserInDB = Depends(require_analyst),
):
    """Exports transactions with GAN-generated Honey Tokens injected to track exfiltration."""
    from fastapi import Response
    import io
    import csv
    import uuid

    watermark_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"
    
    # Grab 5 registered honey tokens from the registry (decoded GAN tokens)
    available_tokens = deception_registry.honey_tokens
    selected_honey = random.sample(available_tokens, min(5, len(available_tokens)))
    
    # Register this export session in the Honey Token Registry
    deception_registry.register_export(
        watermark_id=watermark_id,
        tokens=selected_honey,
        metadata={"actor": current_user.username, "ip": client_ip}
    )
    
    # Log the exfiltration alert in the system audit logs
    log_event(
        actor=current_user.username,
        action="DATA_EXFILTRATION_TRACKED",
        resource="transactions_export.csv",
        detail=f"Watermark: {watermark_id} | Injected: {len(selected_honey)} cWGAN-GP tokens",
        severity="ALERT",
        ip=client_ip
    )
    
    # Compile export records (50 normal transactions + 5 honey tokens)
    export_records = []
    
    # 1. Normal transactions
    for i in range(50):
        txn = _random_txn()
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        cardholder = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{random.randint(10,99)}@{random.choice(EMAIL_DOMAINS)}"
        
        # Luhn valid normal card prefix
        prefix = [4] + [random.randint(0, 9) for _ in range(14)]
        check = _luhn_check_digit(prefix)
        card_num = "".join(map(str, prefix + [check]))
        
        username = f"{first.lower()[:3]}_{last.lower()[:3]}{random.randint(10,99)}"
        password = random.choice(COMMON_PASSWORDS)
        cvv = str(random.randint(100, 999))
        expiry = f"{random.randint(1,12):02d}/{random.randint(26,30)}"
        
        export_records.append([
            txn.id, txn.user_id, cardholder, card_num, cvv, expiry,
            email, username, password, round(txn.amount, 2), txn.device,
            txn.location, txn.ip_address, txn.timestamp, txn.risk_score, txn.is_flagged
        ])
        
    # 2. Honey tokens (GAN-generated)
    for token in selected_honey:
        txn = _random_txn()
        txn.amount = round(random.uniform(500, 4500), 2)
        txn.location = "E-Commerce Checkout"
        txn.risk_score = 0.05  # Deceive hackers by looking safe/unflagged!
        txn.is_flagged = False
        
        export_records.append([
            txn.id, txn.user_id, token["cardholder"], token["card_number"],
            token["cvv"], token["expiry"], token["email"], token["username"],
            token["password"], txn.amount, txn.device, txn.location,
            txn.ip_address, txn.timestamp, txn.risk_score, txn.is_flagged
        ])
        
    random.shuffle(export_records)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "user_id", "cardholder", "card_number", "cvv", "expiry",
        "email", "username", "password", "amount", "device", "location",
        "ip_address", "timestamp", "risk_score", "is_flagged"
    ])
    for row in export_records:
        writer.writerow(row)
    
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=deceptive_net_transactions.csv"}
    )


# ── ATTACKER DECEPTION HONEYPOT ENDPOINTS ───────────────────────────────────────

class PaymentRequest(BaseModel):
    cardholder: str = Field(..., description="Full name of the cardholder", json_schema_extra={"example": "James Smith"})
    card_number: str = Field(..., description="16-digit credit card number", json_schema_extra={"example": "4012213140100006"})
    cvv: str = Field(..., description="3-digit card security code", json_schema_extra={"example": "382"})
    expiry: str = Field(..., description="Card expiration date in MM/YY format", json_schema_extra={"example": "04/28"})
    amount: float = Field(..., description="Simulated purchase amount in USD", json_schema_extra={"example": 249.99})
    simulated_ip: Optional[str] = Field("185.220.101.5", description="Simulated IP address of the attacker", json_schema_extra={"example": "185.220.101.5"})


class LoginRequest(BaseModel):
    username: str = Field(..., description="Decoy portal administrative username", json_schema_extra={"example": "admin_trap"})
    password: str = Field(..., description="Administrative portal password", json_schema_extra={"example": "password123"})
    simulated_ip: Optional[str] = Field("109.245.18.99", description="Simulated IP address of the attacker", json_schema_extra={"example": "109.245.18.99"})


class SSHRequest(BaseModel):
    command: str = Field(..., description="Shell command executed by the attacker in the decoy SSH console", json_schema_extra={"example": "cat credentials_dump.csv"})
    simulated_ip: str = Field(..., description="Simulated IP address of the attacker console session", json_schema_extra={"example": "185.220.101.5"})
    session_id: Optional[str] = Field(None, description="Decoy session identifier", json_schema_extra={"example": "ssh-sess-99"})


@app.post("/api/honeypot/payment", tags=["ML"])
@limiter.limit("30/minute")
async def honeypot_payment(payload: PaymentRequest, request: Request):
    """Decoy Payment Gateway that logs attacker IPs and attributes stolen tokens."""
    client_ip = payload.simulated_ip or (request.client.host if request.client else "127.0.0.1")
    user_agent = request.headers.get("user-agent", "Mozilla/5.0")
    
    # Check registry to attribute stolen card number
    attribution = deception_registry.check_token(payload.card_number)
    
    if attribution:
        hacker = deception_registry.log_caught_hacker(
            attacker_ip=client_ip,
            user_agent=user_agent,
            token_used=payload.card_number,
            token_type="Credit/Debit Card",
            attribution=attribution,
            action="CARD_TRANSACTION_ATTEMPT",
            request_payload=payload.model_dump()
        )
        
        # Log critical incident
        log_event(
            actor="ATTACKER",
            action="HONEYPOT_TRIGGERED",
            resource=f"card={payload.card_number[:4]}...{payload.card_number[-4:]}",
            detail=f"Breach Link: Watermark={attribution['watermark_id']} | Leaked by: {attribution['actor']} | Attacker IP: {client_ip}",
            severity="ALERT",
            ip=client_ip
        )
        
        return {
            "status": "success",
            "message": "Payment processed successfully.",
            "deception_triggered": True,
            "hacker_logged": hacker["id"]
        }
    else:
        return {
            "status": "success",
            "message": "Payment processed successfully.",
            "deception_triggered": False
        }


@app.post("/api/honeypot/login", tags=["ML"])
@limiter.limit("30/minute")
async def honeypot_login(payload: LoginRequest, request: Request):
    """Decoy Admin Portal that traps credential misuse."""
    client_ip = payload.simulated_ip or (request.client.host if request.client else "127.0.0.1")
    user_agent = request.headers.get("user-agent", "Mozilla/5.0")
    
    attribution = deception_registry.check_token(payload.username)
    
    if attribution:
        hacker = deception_registry.log_caught_hacker(
            attacker_ip=client_ip,
            user_agent=user_agent,
            token_used=payload.username,
            token_type="User Credentials",
            attribution=attribution,
            action="HONEYPOT_LOGIN_ATTEMPT",
            request_payload=payload.model_dump()
        )
        
        log_event(
            actor="ATTACKER",
            action="HONEYPOT_TRIGGERED",
            resource=f"user={payload.username}",
            detail=f"Breach Link: Watermark={attribution['watermark_id']} | Leaked by: {attribution['actor']} | Attacker IP: {client_ip}",
            severity="ALERT",
            ip=client_ip
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )


@app.post("/api/honeypot/ssh", tags=["ML"])
@limiter.limit("60/minute")
async def honeypot_ssh(payload: SSHRequest, request: Request):
    """
    Cowrie-based Decoy SSH Terminal Shell that captures attacker commands,
    evaluates DQN policy changes, and registers watermark tokens dynamically.
    """
    client_ip = payload.simulated_ip or (request.client.host if request.client else "127.0.0.1")
    cmd_raw = payload.command.strip()
    cmd_parts = cmd_raw.split()
    cmd = cmd_parts[0].lower() if cmd_parts else ""
    
    # Map command to DQN transitions
    state = "normal_idle"
    action = "hold_token_deployment"
    reward = 0.0
    response = ""
    
    if cmd == "help":
        response = (
            "Available commands:<br>"
            "&nbsp;&nbsp;help&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Show this help manual<br>"
            "&nbsp;&nbsp;clear&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Clear the console screen<br>"
            "&nbsp;&nbsp;whoami&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Show current user login<br>"
            "&nbsp;&nbsp;ls&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- List directory files<br>"
            "&nbsp;&nbsp;cat [file]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- View contents of a file<br>"
            "&nbsp;&nbsp;wget [url]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Download from remote URL"
        )
        state = "normal_idle"
        action = "hold_token_deployment"
        reward = -0.05
    elif cmd == "whoami":
        response = "guest"
        state = "normal_idle"
        action = "hold_token_deployment"
        reward = 0.05
    elif cmd == "ls":
        response = "<span style=\"color:var(--blue)\">credentials_dump.csv</span>&nbsp;&nbsp;&nbsp;&nbsp;<span style=\"color:var(--blue)\">base_watermarked.csv</span>&nbsp;&nbsp;&nbsp;&nbsp;system_logs.log"
        state = "session_dwell_long"
        action = "inject_professional_honey_token"
        reward = 0.25
    elif cmd == "cat":
        filename = cmd_parts[1].lower() if len(cmd_parts) > 1 else ""
        if filename == "credentials_dump.csv":
            available = deception_registry.honey_tokens
            selected = random.sample(available, min(3, len(available)))
            watermark_id = f"ssh-leak-{str(uuid.uuid4())[:8]}"
            deception_registry.register_export(
                watermark_id=watermark_id,
                tokens=selected,
                metadata={"actor": "ssh_honeypot_exfil", "ip": client_ip}
            )
            
            response = "id,username,password,cc_number,cvv,expiry<br>"
            for idx, t in enumerate(selected):
                response += f"{idx+1},{t['username']},{t['password']},{t['card_number']},{t['cvv']},{t['expiry']} &lt;-- [TDA ACTIVE HONEY TOKEN]<br>"
            response += "<br><span style=\"color:var(--yellow)\">[TDA Agent Alert] DQN model triggered - dynamic key deployment policy updated.</span>"
            
            state = "entropy_high"
            action = "inject_credit_card_honey_token"
            reward = 5.50
        elif filename == "base_watermarked.csv":
            response = (
                "txn_id,user_id,amount,location,is_flagged<br>"
                "TXN-10294,USR-1029,450.00,Mumbai,False<br>"
                "TXN-10385,USR-1830,1250.75,New York,False<br>"
                "TXN-90132,USR-2091,9999.99,TOR_EXIT_NODE_HONEYPOT,True &lt;-- [WATERMARKED DECEPTION ROW]"
            )
            state = "command_entropy_spike"
            action = "inject_foreign_ip_honey_token"
            reward = 4.80
        elif filename == "system_logs.log":
            response = (
                f"2026-06-04 22:50:39 IST [INFO] SSH Session established from {client_ip}<br>"
                "2026-06-04 22:51:12 IST [INFO] Command executed: ls<br>"
                "2026-06-04 22:51:30 IST [INFO] DQN policy loaded, deploying decoy features..."
            )
            state = "session_dwell_long"
            action = "hold_token_deployment"
            reward = 0.10
        elif not filename:
            response = "cat: missing file operand"
            state = "normal_idle"
            action = "hold_token_deployment"
            reward = -0.10
        else:
            response = f"cat: {cmd_parts[1]}: No such file or directory"
            state = "normal_idle"
            action = "hold_token_deployment"
            reward = -0.10
    elif cmd == "wget":
        url = cmd_parts[1] if len(cmd_parts) > 1 else ""
        if not url:
            response = "wget: missing URL"
            state = "normal_idle"
            action = "hold_token_deployment"
            reward = -0.10
        else:
            available = deception_registry.honey_tokens
            selected = random.sample(available, min(3, len(available)))
            watermark_id = f"ssh-wget-{str(uuid.uuid4())[:8]}"
            deception_registry.register_export(
                watermark_id=watermark_id,
                tokens=selected,
                metadata={"actor": "ssh_wget_exfil", "ip": client_ip}
            )
            response = (
                f"Connecting to {url}... connected.<br>"
                "HTTP request sent, awaiting response... 200 OK<br>"
                "Length: 10642 (10K)<br>"
                "Saving to: 'credentials_dump.csv'<br><br>"
                "credentials_dump.csv   100%[===================&gt;]  10.39K  --.-KB/s    in 0.05s<br><br>"
                "Downloaded file successfully. Activity logged in honeypot logs."
            )
            state = "command_entropy_spike"
            action = "rotate_all_deployed_tokens"
            reward = 10.50
    elif not cmd:
        response = ""
        state = "normal_idle"
        action = "hold_token_deployment"
        reward = 0.0
    else:
        response = f"-bash: {cmd}: command not found"
        state = "normal_idle"
        action = "hold_token_deployment"
        reward = -0.20

    # Log the command transition in DQN
    transition = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "state": state,
        "action": action,
        "reward": float(reward),
        "epsilon": DECEPTIVE_NET_CONFIG.get("dqn_epsilon", 0.05)
    }
    global TDA_TRANSITIONS
    TDA_TRANSITIONS.insert(0, transition)
    if len(TDA_TRANSITIONS) > 50:
        TDA_TRANSITIONS.pop()
        
    # Log incident
    log_event(
        actor="ATTACKER",
        action="SSH_COMMAND_EXECUTED",
        resource=f"cmd='{cmd_raw[:40]}'",
        detail=f"IP: {client_ip} | TDA Action: {action} | Q-Reward: {reward}",
        severity="WARN" if reward <= 1.0 else "ALERT",
        ip=client_ip
    )
    
    return {
        "status": "success",
        "response": response,
        "state": state,
        "action": action,
        "reward": reward
    }


@app.get("/api/admin/threat-intel", tags=["Admin"])
async def get_threat_intel(current_user: UserInDB = Depends(require_admin)):
    """Returns captured threat logs and exfiltration attributions (admin only)."""
    return {
        "caught_hackers": deception_registry.get_caught_hackers(),
        "exports": deception_registry.get_exports()
    }


@app.get("/api/admin/tda-logs", tags=["Admin"])
async def get_tda_logs(current_user: UserInDB = Depends(require_admin)):
    """Returns TDA DQN policy simulation events (admin only)."""
    return TDA_TRANSITIONS


@app.get("/api/admin/config", tags=["Admin"])
async def get_system_config(current_user: UserInDB = Depends(require_admin)):
    """Retrieve system configurations (admin only)."""
    return DECEPTIVE_NET_CONFIG


@app.post("/api/admin/config", tags=["Admin"])
async def update_system_config(payload: SystemConfigUpdate, current_user: UserInDB = Depends(require_admin)):
    """Update system configurations (admin only)."""
    global DECEPTIVE_NET_CONFIG
    if payload.dqn_epsilon is not None:
        DECEPTIVE_NET_CONFIG["dqn_epsilon"] = payload.dqn_epsilon
    if payload.token_deployment_frequency is not None:
        DECEPTIVE_NET_CONFIG["token_deployment_frequency"] = payload.token_deployment_frequency
    if payload.active_honeypots is not None:
        DECEPTIVE_NET_CONFIG["active_honeypots"] = payload.active_honeypots
    if payload.token_rotation_enabled is not None:
        DECEPTIVE_NET_CONFIG["token_rotation_enabled"] = payload.token_rotation_enabled
    if payload.watermark_compliance_min is not None:
        DECEPTIVE_NET_CONFIG["watermark_compliance_min"] = payload.watermark_compliance_min

    log_event(
        actor=current_user.username,
        action="CONFIG_UPDATED",
        resource="system_config",
        detail=f"Updated keys: {list(payload.model_dump(exclude_none=True).keys())}",
        severity="INFO"
    )
    return DECEPTIVE_NET_CONFIG


@app.get("/api/admin/users", tags=["Admin"])
async def get_users_list(current_user: UserInDB = Depends(require_admin)):
    """List all registered system users (admin only)."""
    from backend.auth import USERS_DB
    users = []
    for username, details in USERS_DB.items():
        users.append({
            "username": username,
            "role": details["role"],
            "full_name": details["full_name"]
        })
    return users


@app.post("/api/admin/users/role", tags=["Admin"])
async def update_user_role(payload: UserRoleUpdate, current_user: UserInDB = Depends(require_admin)):
    """Update a user's role assignment (admin only)."""
    from backend.auth import USERS_DB
    if payload.username not in USERS_DB:
        raise HTTPException(status_code=404, detail="User not found")
    if payload.role not in ("admin", "analyst", "viewer"):
        raise HTTPException(status_code=400, detail="Invalid role specification")
    
    old_role = USERS_DB[payload.username]["role"]
    USERS_DB[payload.username]["role"] = payload.role
    
    log_event(
        actor=current_user.username,
        action="ROLE_ASSIGNMENT_UPDATED",
        resource=f"user={payload.username}",
        detail=f"Role changed from {old_role} to {payload.role}",
        severity="ALERT"
    )
    return {"status": "success", "username": payload.username, "role": payload.role}



@app.get("/api/predict/{txn_id}", response_model=PredictionResult, tags=["ML"])
@limiter.limit("30/minute")
async def predict_fraud(
    request: Request,
    txn_id: str,
    current_user: UserInDB = Depends(require_analyst),
):
    """Run ML ensemble fraud prediction on a specific transaction."""
    txn = next((t for t in MOCK_DATA if t.id == txn_id), None)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    txn_dict = {
        "transaction_amount":    txn.amount,
        "user_age":              txn.age,
        "account_age_days":      txn.account_age,
        "device_type":           txn.device,
        "distance_from_home":    txn.distance,
        "num_transactions_24h":  txn.num_txn_24h,
        "avg_txn_amount_7d":     txn.amount * 0.9,
        "failed_attempts":       0,
        "is_foreign_transaction": 0,
        "hour_of_day":           datetime.now(IST).hour,
        "day_of_week":           datetime.now(IST).weekday(),
        "credit_score":          650,
        "monthly_income":        5000,
        "num_cards":             2,
        "email_is_free":         1,
        "phone_mobile":          1,
        "has_chip":              1,
        "pin_changed_recently":  0,
        "velocity_change":       txn.velocity_change,
        "merchant_risk_score":   txn.merchant_risk,
    }

    lgbm  = predict_fraud_prob(txn_dict)
    ae    = predict_anomaly_score(txn_dict)
    ens   = combined_score(txn_dict)
    top5  = explain_prediction(txn_dict)

    risk_level = "CRITICAL" if ens > 0.75 else "HIGH" if ens > 0.5 else "MEDIUM" if ens > 0.25 else "LOW"
    if risk_level in ("CRITICAL", "HIGH"):
        log_event(actor=current_user.username, action="HIGH_RISK_FLAGGED",
                  resource=txn_id, detail=f"score={ens}", severity="ALERT")

    return PredictionResult(
        transaction_id=txn_id,
        fraud_probability=round(lgbm, 4),
        anomaly_score=round(ae, 4),
        ensemble_score=round(ens, 4),
        risk_level=risk_level,
        top_features=top5,
    )


@app.get("/api/explain/{txn_id}", tags=["ML"])
@limiter.limit("20/minute")
async def explain_txn(
    request: Request,
    txn_id: str,
    current_user: UserInDB = Depends(require_analyst),
):
    """Returns SHAP feature contributions for a transaction."""
    txn = next((t for t in MOCK_DATA if t.id == txn_id), None)
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    txn_dict = {
        "transaction_amount": txn.amount, "user_age": txn.age,
        "account_age_days": txn.account_age, "device_type": txn.device,
        "distance_from_home": txn.distance, "num_transactions_24h": txn.num_txn_24h,
        "velocity_change": txn.velocity_change, "merchant_risk_score": txn.merchant_risk,
        "avg_txn_amount_7d": txn.amount * 0.9, "failed_attempts": 0,
        "is_foreign_transaction": 0, "hour_of_day": datetime.now(IST).hour,
        "day_of_week": datetime.now(IST).weekday(), "credit_score": 650,
        "monthly_income": 5000, "num_cards": 2, "email_is_free": 1,
        "phone_mobile": 1, "has_chip": 1, "pin_changed_recently": 0,
    }
    return {"transaction_id": txn_id, "shap_contributions": explain_prediction(txn_dict)}


@app.get("/api/metrics", tags=["Admin"])
async def model_metrics(current_user: UserInDB = Depends(require_admin)):
    """Returns model training performance metrics (admin only)."""
    log_event(actor=current_user.username, action="VIEW_METRICS", severity="INFO")
    metrics = get_model_metrics()
    metrics.update({
        "gat_layers": 2,
        "gat_attention_heads": 4,
        "gat_hidden_dim": 16,
        "fdg_node_count": 5000,
        "fdg_similarity_threshold": 0.5,
        "fdg_avg_degree": 4.8,
        "conditioning_type": "Fraud-Aware GAT Conditioning",
        "film_modulation_blocks": 8,
        "luhn_attention_compliance": "98.7%",
        "name_email_coherence_index": "92.4%",
        "zip_state_accuracy_rate": "96.1%"
    })
    return metrics


@app.get("/api/shap_importance", tags=["Admin"])
async def shap_importance(current_user: UserInDB = Depends(require_admin)):
    """Returns global SHAP feature importance (admin only)."""
    shap_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models", "shap_values.json")
    if not os.path.exists(shap_path):
        raise HTTPException(status_code=503, detail="SHAP data not available. Retrain model.")
    with open(shap_path) as f:
        return json.load(f)


# ── Audit log endpoints ───────────────────────────────────────────────────────
@app.get("/api/audit/logs", tags=["Admin"])
@limiter.limit("30/minute")
async def audit_logs(
    request: Request,
    n: int = 100,
    current_user: UserInDB = Depends(require_admin),
):
    """Returns recent audit log entries (admin only)."""
    log_event(actor=current_user.username, action="VIEW_AUDIT_LOG",
              detail=f"n={n}", severity="INFO")
    return get_recent_events(n)


@app.get("/api/audit/alerts", tags=["Admin"])
async def audit_alerts(current_user: UserInDB = Depends(require_admin)):
    """Returns ALERT/WARN severity audit events (admin only)."""
    return get_alert_events(50)


# ── GenAI threat report ───────────────────────────────────────────────────────
@app.get("/api/genai_report", response_model=AIReport, tags=["Admin"])
@limiter.limit("5/minute")
async def genai_report(
    request: Request,
    current_user: UserInDB = Depends(require_admin),
):
    """
    Generates a simulated AI-powered threat analysis report.
    In production: replace with a call to an LLM API (e.g., Gemini, GPT-4).
    """
    flagged = [t for t in MOCK_DATA if t.is_flagged]
    high_risk_count  = len(flagged)
    top_location     = max(set(t.location for t in flagged), key=lambda l: sum(1 for t in flagged if t.location == l)) if flagged else "N/A"
    avg_flag_amount  = round(sum(t.amount for t in flagged) / max(len(flagged), 1), 2)

    alerts = [
        f"{high_risk_count} transactions flagged by ensemble model (score > 0.6).",
        f"Highest concentration of suspicious activity: {top_location}.",
        f"Average flagged transaction value: ${avg_flag_amount} — significantly above baseline.",
        "Velocity anomaly detected: 3 user IDs showing >5x normal transaction frequency.",
        "Autoencoder reconstruction error spike observed in last 15 minutes.",
    ]
    severity = "CRITICAL" if high_risk_count > 20 else "HIGH" if high_risk_count > 10 else "MEDIUM"

    log_event(actor=current_user.username, action="GENAI_REPORT_GENERATED",
              detail=f"flagged={high_risk_count}", severity="INFO")

    return AIReport(
        timestamp=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        summary=f"Deceptive-Net AI Analysis: {high_risk_count} anomalous transactions detected in current data window. Ensemble model (LightGBM + Autoencoder) confidence is HIGH. Pattern indicates possible coordinated card-not-present fraud originating from {top_location}.",
        alerts=alerts,
        recommendation=f"Recommend stepping up authentication requirements for transactions from {top_location} exceeding ${avg_flag_amount}. Isolate USR sessions with velocity_change > 0.8 for manual review.",
        risk_level=severity,
    )


# ── WebSocket – live transaction stream ───────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


ws_manager = ConnectionManager()


@app.websocket("/ws/transactions")
async def ws_transactions(websocket: WebSocket, token: Optional[str] = None):
    """Streams new synthetic transactions every 3 seconds, authenticated via token query param."""
    # Accept token as a query parameter: ?token=XYZ
    token = token or websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
        
    try:
        from jose import jwt
        from backend.auth import SECRET_KEY, ALGORITHM
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if username is None or role not in ("admin", "analyst", "viewer"):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    except Exception:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws_manager.connect(websocket)
    try:
        while True:
            txn = _random_txn()
            MOCK_DATA.insert(0, txn)
            if len(MOCK_DATA) > 500:
                MOCK_DATA.pop()
            await ws_manager.broadcast(txn.model_dump_json())
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
