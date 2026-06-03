#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Deceptive-Net: Web UI - Admin Panel & User Dashboard
================================================================================

Minimal, clean interface for:
1. Admin Panel: Model metrics, user monitoring, activity logs
2. User Dashboard: Limited fraud detection interface
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Deceptive-Net",
    description="Fraud Detection & Honey Token Generation System",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class UserActivity(BaseModel):
    """User activity log entry"""
    timestamp: str
    user_id: str
    action: str  # "login", "query", "download", etc.
    ip_address: str
    status: str  # "success", "suspicious", "blocked"
    details: Optional[Dict] = None


class ModelMetrics(BaseModel):
    """Current model performance metrics"""
    dds_score: float
    luhn_validity: float
    name_email_similarity: float
    throughput: float
    inference_time_ms: float
    total_tokens_generated: int
    uptime_hours: float


class HoneyToken(BaseModel):
    """Generated honey token"""
    token_id: str
    created_at: str
    baf_features: List[float]
    pii_features: List[float]
    validity_score: float


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA & STATE
# ─────────────────────────────────────────────────────────────────────────────

# Simulated state
mock_metrics = {
    "dds_score": 0.67,
    "luhn_validity": 95.8,
    "name_email_similarity": 90.7,
    "throughput": 14500.0,
    "inference_time_ms": 0.74,
    "total_tokens_generated": 45000,
    "uptime_hours": 2.5,
    "model_status": "running",
}

mock_activities = [
    {
        "timestamp": "2026-05-16 23:45:12",
        "user_id": "admin_001",
        "action": "login",
        "ip_address": "192.168.1.100",
        "status": "success",
        "details": {"method": "password"}
    },
    {
        "timestamp": "2026-05-16 23:46:03",
        "user_id": "user_042",
        "action": "query_tokens",
        "ip_address": "10.0.0.50",
        "status": "success",
        "details": {"count": 100, "format": "csv"}
    },
    {
        "timestamp": "2026-05-16 23:47:55",
        "user_id": "unknown",
        "action": "unauthorized_access",
        "ip_address": "203.0.113.45",
        "status": "blocked",
        "details": {"attempts": 5, "method": "api_key"}
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# ADMIN PANEL ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Admin dashboard homepage"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deceptive-Net - Admin Panel</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            header {
                background: #1a1a1a;
                color: white;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            header h1 { font-size: 24px; }
            header .status {
                background: #4caf50;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card h3 {
                color: #666;
                font-size: 14px;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .metric-card .value {
                font-size: 32px;
                font-weight: bold;
                color: #d62728;
            }
            .metric-card .unit {
                color: #999;
                font-size: 14px;
                margin-left: 8px;
            }
            .activity-section {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .activity-section h2 {
                margin-bottom: 20px;
                font-size: 18px;
            }
            .activity-list {
                list-style: none;
            }
            .activity-item {
                padding: 12px;
                border-left: 3px solid #ddd;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 14px;
            }
            .activity-item.success { border-color: #4caf50; }
            .activity-item.suspicious { border-color: #ff9800; }
            .activity-item.blocked { border-color: #f44336; }
            .badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 600;
            }
            .badge.success { background: #e8f5e9; color: #2e7d32; }
            .badge.suspicious { background: #fff3e0; color: #e65100; }
            .badge.blocked { background: #ffebee; color: #c62828; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1>🛡️ Deceptive-Net Admin Panel</h1>
                    <p style="color: #ccc; margin-top: 5px; font-size: 14px;">Fraud Detection & Honey Token System</p>
                </div>
                <div class="status">System: Online</div>
            </header>
            
            <div class="metrics-grid" id="metrics"></div>
            
            <div class="activity-section">
                <h2>Recent Activity</h2>
                <ul class="activity-list" id="activities"></ul>
            </div>
        </div>
        
        <script>
            async function loadMetrics() {
                const response = await fetch('/api/admin/metrics');
                const data = await response.json();
                const metricsDiv = document.getElementById('metrics');
                
                const metrics = [
                    { label: 'DDS Score', value: data.dds_score.toFixed(3), unit: '(lower better)' },
                    { label: 'Luhn Validity', value: data.luhn_validity.toFixed(1), unit: '%' },
                    { label: 'Coherence', value: data.name_email_similarity.toFixed(1), unit: '%' },
                    { label: 'Throughput', value: (data.throughput/1000).toFixed(1), unit: 'k tokens/s' },
                    { label: 'Inference Time', value: data.inference_time_ms.toFixed(2), unit: 'ms' },
                    { label: 'Tokens Generated', value: data.total_tokens_generated, unit: '' },
                ];
                
                metricsDiv.innerHTML = metrics.map(m => `
                    <div class="metric-card">
                        <h3>${m.label}</h3>
                        <div class="value">${m.value}<span class="unit">${m.unit}</span></div>
                    </div>
                `).join('');
            }
            
            async function loadActivities() {
                const response = await fetch('/api/admin/activities');
                const activities = await response.json();
                const activitiesList = document.getElementById('activities');
                
                activitiesList.innerHTML = activities.map(a => `
                    <li class="activity-item ${a.status}">
                        <div>
                            <strong>${a.user_id}</strong> · ${a.action}
                            <br>
                            <span style="color: #999; font-size: 12px;">${a.timestamp} · ${a.ip_address}</span>
                        </div>
                        <span class="badge ${a.status}">${a.status.toUpperCase()}</span>
                    </li>
                `).join('');
            }
            
            loadMetrics();
            loadActivities();
            setInterval(loadMetrics, 5000);  // Refresh every 5 seconds
            setInterval(loadActivities, 3000);
        </script>
    </body>
    </html>
    """


@app.get("/api/admin/metrics", response_model=ModelMetrics)
async def get_metrics():
    """Get current model metrics"""
    return ModelMetrics(**mock_metrics)


@app.get("/api/admin/activities", response_model=List[UserActivity])
async def get_activities(limit: int = 20):
    """Get user activity log"""
    return [UserActivity(**a) for a in mock_activities[:limit]]


@app.post("/api/admin/activities")
async def log_activity(activity: UserActivity):
    """Log user activity"""
    logger.info(f"Activity: {activity.user_id} - {activity.action} - {activity.status}")
    mock_activities.insert(0, activity.dict())
    return {"status": "logged"}


# ─────────────────────────────────────────────────────────────────────────────
# USER DASHBOARD ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard():
    """User dashboard homepage"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deceptive-Net - User Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .dashboard {
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                max-width: 600px;
                width: 100%;
                padding: 40px;
            }
            h1 {
                margin-bottom: 30px;
                color: #333;
                text-align: center;
            }
            .content-box {
                margin-bottom: 30px;
            }
            .content-box h2 {
                font-size: 16px;
                color: #666;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .token-card {
                background: #f9f9f9;
                border-left: 4px solid #667eea;
                padding: 15px;
                border-radius: 4px;
                margin-bottom: 10px;
                font-size: 14px;
                font-family: 'Courier New', monospace;
            }
            .token-card code {
                display: block;
                word-break: break-all;
                color: #555;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                cursor: pointer;
                font-weight: 600;
                width: 100%;
                margin-top: 20px;
                transition: background 0.3s;
            }
            button:hover {
                background: #764ba2;
            }
            .notice {
                background: #f0f7ff;
                border: 1px solid #d0e8ff;
                padding: 15px;
                border-radius: 6px;
                color: #333;
                font-size: 14px;
                line-height: 1.6;
            }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <h1>🔐 Your Fraud Detection Status</h1>
            
            <div class="content-box">
                <h2>System Status</h2>
                <div class="notice">
                    ✓ Your account is secure. No suspicious activities detected.
                    <br><br>
                    Last check: <strong>Just now</strong>
                </div>
            </div>
            
            <div class="content-box">
                <h2>Generated Tokens</h2>
                <p id="token-count" style="color: #666; margin-bottom: 15px;">Loading...</p>
                <button onclick="generateTokens()">Generate New Tokens</button>
            </div>
        </div>
        
        <script>
            async function loadStatus() {
                try {
                    const response = await fetch('/api/user/status');
                    const data = await response.json();
                    document.getElementById('token-count').innerHTML = 
                        `You have generated <strong>${data.total_tokens}</strong> tokens.`;
                } catch (e) {
                    console.log("Demo mode");
                }
            }
            
            async function generateTokens() {
                alert("Token generation coming soon!");
            }
            
            loadStatus();
        </script>
    </body>
    </html>
    """


@app.get("/api/user/status")
async def get_user_status():
    """Get user's current status"""
    return {
        "is_secure": True,
        "last_check": "2026-05-16T23:45:00",
        "suspicious_activities": 0,
        "total_tokens": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/generate-tokens")
async def generate_tokens_api(count: int = 100):
    """Generate honey tokens (placeholder)"""
    if count > 10000:
        raise HTTPException(status_code=400, detail="Maximum 10000 tokens per request")
    
    tokens = []
    for i in range(count):
        tokens.append({
            "token_id": f"ht_{i:06d}",
            "created_at": datetime.now().isoformat(),
            "baf_features": np.random.rand(87).tolist(),
            "pii_features": np.random.rand(64).tolist(),
            "validity_score": 0.95,
        })
    
    return {"count": len(tokens), "tokens": tokens}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Deceptive-Net Web UI...")
    logger.info("Admin Panel:  http://localhost:5000/admin")
    logger.info("User Dashboard: http://localhost:5000/dashboard")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
