import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session

from database import get_db  
from auth.utils import create_access_token, verify_token, get_password_hash, verify_password, create_refresh_token, get_refresh_token_expiry
from auth.schemas import UserCreate, UserLogin, TokenRefreshRequest, TokenResponse
from auth.models import User, RefreshToken
from auth.dependencies import get_current_user 
from supabase import create_client
from datetime import datetime


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter()

@router.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):

    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")


    if user.is_admin:
        raise HTTPException(status_code=400, detail="Cannot register as admin. Admins must be assigned by the system.")
    
    user_role = user.user_role or "user"
    if user_role not in ["user"]:  
        raise HTTPException(status_code=400, detail="Role must be 'user' during registration.")
    
    hashed_password = get_password_hash(user.password)
    
    new_user = User(
        email=user.email,
        password_hash=hashed_password,
        is_admin=False,  
        user_role=user_role,
        first_name=user.first_name,
        last_name=user.last_name
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User created successfully"}


@router.post("/login")
async def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user is None or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    access_token = create_access_token(data={
        "sub": db_user.email,
        "name": db_user.first_name,
        "is_admin": db_user.is_admin,
        "user_role": db_user.user_role
    })
    refresh_token_str = create_refresh_token()
    expires_at = get_refresh_token_expiry()

    refresh_token = RefreshToken(
        user_id=db_user.id,
        token=refresh_token_str,
        expires_at=expires_at
    )
    db.add(refresh_token)
    db.commit()
    

    return {
        "access_token": access_token,
        "refresh_token": refresh_token_str,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(request: TokenRefreshRequest, db: Session = Depends(get_db)):
    token_str = request.refresh_token
    db_token = db.query(RefreshToken).filter(
        RefreshToken.token == token_str,
        RefreshToken.revoked == False,
        RefreshToken.expires_at > datetime.utcnow()
    ).first()

    if not db_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")

    user = db.query(User).filter(User.id == db_token.user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    db_token.revoked = True
    db.commit()

    access_token = create_access_token(data={
        "sub": str(user.email),
        "name": user.first_name,
        "is_admin": user.is_admin,
        "user_role": user.user_role
    })

    refresh_token_str = create_refresh_token()
    expires_at = get_refresh_token_expiry()

    new_refresh_token = RefreshToken(
        user_id=user.id,
        token=refresh_token_str,
        expires_at=expires_at
    )
    db.add(new_refresh_token)
    db.commit()

    return {
        "access_token": access_token,
        "refresh_token": refresh_token_str,
        "token_type": "bearer"
    }


# @router.get("/admin-only")
# def only_admin(current_user: dict = Depends(get_current_user)):
#     if current_user["role"] != "admin":
#         raise HTTPException(status_code=403, detail="Not authorized")
#     return {"message": f"Welcome, admin {current_user['email']}!"}
