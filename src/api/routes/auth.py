"""Authentication endpoints"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from ...database.supabase_client import get_supabase_client
from ...observability.tracing import trace_function
from ...security.auth_service import auth_service
from ...security.encryption import encryption_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


# Request/Response Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    phone_number: Optional[str] = None
    date_of_birth: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    customer_id: str
    email: str
    first_name: str
    last_name: str
    kyc_status: str


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@trace_function("auth_register")
async def register_user(user_data: UserRegister):
    """
    Register a new user/customer
    
    Creates customer account with hashed password
    """
    try:
        supabase = get_supabase_client()
        
        # Check if user already exists
        existing = await supabase.get_customer_by_email(user_data.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        hashed_password = encryption_service.hash_password(user_data.password)
        
        # Create customer
        customer_data = {
            "email": user_data.email,
            "hashed_password": hashed_password,
            "full_name": f"{user_data.first_name} {user_data.last_name}",
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "phone_number": user_data.phone_number,
            "date_of_birth": user_data.date_of_birth,
            "kyc_status": "pending",
            "risk_score": 50,  # Default medium risk
            "is_active": True
        }
        
        customer = await supabase.create_customer(customer_data)
        
        logger.info(f"User registered: {customer['customer_id']}")
        
        return UserResponse(
            customer_id=customer["customer_id"],
            email=customer["email"],
            first_name=customer["first_name"],
            last_name=customer["last_name"],
            kyc_status=customer["kyc_status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/token", response_model=TokenResponse)
@trace_function("auth_login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login
    
    Returns access and refresh tokens
    """
    try:
        supabase = get_supabase_client()
        
        # Get user by email
        user = await supabase.get_customer_by_email(form_data.username)
        
        # Authenticate
        authenticated_user = auth_service.authenticate_user(
            email=form_data.username,
            password=form_data.password,
            user_data=user
        )
        
        if not authenticated_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if account is active
        if not authenticated_user.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Create tokens
        token_data = {
            "sub": authenticated_user["customer_id"],
            "email": authenticated_user["email"],
            "first_name": authenticated_user["first_name"],
            "last_name": authenticated_user["last_name"]
        }
        
        access_token = auth_service.create_access_token(data=token_data)
        refresh_token = auth_service.create_refresh_token(data=token_data)
        
        logger.info(f"User logged in: {authenticated_user['customer_id']}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
@trace_function("auth_refresh")
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token
    """
    try:
        # Verify refresh token
        payload = auth_service.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Create new tokens
        token_data = {
            "sub": payload["sub"],
            "email": payload["email"],
            "first_name": payload.get("first_name"),
            "last_name": payload.get("last_name")
        }
        
        new_access_token = auth_service.create_access_token(data=token_data)
        new_refresh_token = auth_service.create_refresh_token(data=token_data)
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.get("/me", response_model=UserResponse)
@trace_function("auth_me")
async def get_current_user_info(customer_id: str = Depends(auth_service.get_current_user_id)):
    """
    Get current authenticated user information
    """
    try:
        supabase = get_supabase_client()
        
        customer = await supabase.get_customer_by_id(customer_id)
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            customer_id=customer["customer_id"],
            email=customer["email"],
            first_name=customer["first_name"],
            last_name=customer["last_name"],
            kyc_status=customer["kyc_status"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current user failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )


@router.post("/logout")
@trace_function("auth_logout")
async def logout(customer_id: str = Depends(auth_service.get_current_user_id)):
    """
    Logout user (client should discard tokens)
    """
    logger.info(f"User logged out: {customer_id}")
    return {"message": "Successfully logged out"}
