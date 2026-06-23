"""
Deceptive-Net – Auth Module
============================
JWT-based Role-Based Access Control (RBAC).
Roles: admin | analyst | viewer

Dependencies: python-jose, passlib[bcrypt]
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

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── config ────────────────────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("JWT_SECRET_KEY", "deceptive-net-dev-secret-change-in-production")
ALGORITHM   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# ── static user database (in production: use a proper DB) ─────────────────────
# Hashes pre-computed: bcrypt(password, rounds=12)
# admin123    → $2b$12$...
# analyst123  → $2b$12$...
# viewer123   → $2b$12$...
# To regenerate: python -c "from passlib.context import CryptContext; c=CryptContext(schemes=['bcrypt']); print(c.hash('yourpassword'))"
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "full_name": "System Administrator",
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": "analyst",
        "full_name": "Fraud Analyst",
    },
    "viewer": {
        "username": "viewer",
        "hashed_password": pwd_context.hash("viewer123"),
        "role": "viewer",
        "full_name": "Read-Only Viewer",
    },
}


# ── models ────────────────────────────────────────────────────────────────────
class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class UserInDB(BaseModel):
    username: str
    role: str
    full_name: str


# ── helpers ───────────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = USERS_DB.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def register_new_user(username: str, password: str, full_name: str, role: str = "viewer") -> Optional[dict]:
    if username in USERS_DB:
        return None
    USERS_DB[username] = {
        "username": username,
        "hashed_password": pwd_context.hash(password),
        "role": role,
        "full_name": full_name,
    }
    return USERS_DB[username]


def reset_user_password(username: str, new_password: str) -> bool:
    if username not in USERS_DB:
        return False
    USERS_DB[username]["hashed_password"] = pwd_context.hash(new_password)
    return True


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role     = payload.get("role")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception

    user = USERS_DB.get(token_data.username)
    if user is None:
        raise credentials_exception
    return UserInDB(**{k: user[k] for k in ("username", "role", "full_name")})


def require_role(*roles):
    """Dependency factory — enforces minimum role access."""
    async def role_checker(current_user: UserInDB = Depends(get_current_user)):
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {roles}",
            )
        return current_user
    return role_checker


# ── role shorthands ───────────────────────────────────────────────────────────
require_admin   = require_role("admin")
require_analyst = require_role("admin", "analyst")
require_viewer  = require_role("admin", "analyst", "viewer")
