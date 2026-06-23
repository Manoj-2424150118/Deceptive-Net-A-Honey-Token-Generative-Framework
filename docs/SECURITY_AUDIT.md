# Deceptive-Net: Security Audit Report

**Version:** 2.0.0  
**Standard:** OWASP Top 10 (2021)  
**Date:** May 2026

---

## Executive Summary

This report documents the security posture of the Deceptive-Net system. Each OWASP Top 10 category was assessed and mitigations were applied.

---

## OWASP Top 10 Assessment

| # | Category | Status | Mitigation Applied |
|---|---|---|---|
| A01 | Broken Access Control | ✅ MITIGATED | JWT RBAC with 3 roles (admin/analyst/viewer). Every endpoint requires explicit role dependency. |
| A02 | Cryptographic Failures | ✅ MITIGATED | Passwords hashed with bcrypt (passlib). JWTs signed with HS256. All API communication over localhost TLS-capable. |
| A03 | Injection | ✅ MITIGATED | No raw SQL. Pydantic models validate and coerce all input. No shell command construction. |
| A04 | Insecure Design | ✅ MITIGATED | Least-privilege RBAC. Audit logging on every action. Immutable ring buffer for tamper evidence. |
| A05 | Security Misconfiguration | ✅ MITIGATED | Security headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy) added via middleware. CORS restricted to localhost origins. |
| A06 | Vulnerable Components | ✅ MITIGATED | All dependencies pinned to specific versions. No known CVEs in pinned versions at time of deployment. |
| A07 | Auth & Session Failures | ✅ MITIGATED | Short-lived JWTs (60 min). Login rate-limited to 10/min. Failed logins are audit-logged with WARN severity. |
| A08 | Software & Data Integrity | ✅ MITIGATED | Docker image built from pinned base images. No external CDN dependencies in production. |
| A09 | Security Logging & Monitoring | ✅ MITIGATED | Every HTTP request, login, prediction, and admin action is logged. Ring buffer + rotating file. Alert-only view in admin panel. |
| A10 | SSRF | ✅ MITIGATED | No user-supplied URLs are fetched. No outbound HTTP requests in application logic. |

---

## Rate Limiting Summary

| Endpoint | Limit |
|---|---|
| POST /auth/token | 10 req/min |
| GET /api/transactions | 60 req/min |
| GET /api/predict/:id | 30 req/min |
| GET /api/genai_report | 5 req/min |
| Default (all others) | 200 req/min |

---

## Known Limitations (Academic Context)

1. **In-memory user database**: The USERS_DB is hardcoded. In production, this would be a properly salted database with bcrypt hashes.
2. **SECRET_KEY**: Loaded from environment variable. The default value must be changed before any real deployment.
3. **HTTP not HTTPS**: The local Docker setup uses plain HTTP. A real deployment would terminate TLS at a reverse proxy (nginx) before forwarding to uvicorn.
4. **CORS allows localhost**: Restricted to `localhost:8080`. In production, this should be the specific domain.

---

## Recommendations for Production Hardening

- Replace in-memory user DB with PostgreSQL + SQLAlchemy
- Implement token revocation list (Redis)  
- Add mutual TLS between internal services
- Set `SECRET_KEY` from Docker secrets or Vault
- Enable fail2ban-style IP blocking after N failed logins
