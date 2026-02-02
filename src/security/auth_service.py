"""Authentication service - OAuth2/JWT implementation"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
import logging

from ..utils.config import settings
from ..security.encryption import encryption_service

logger = logging.getLogger(__name__)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


class AuthService:
    """Handle authentication and JWT token management"""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_EXPIRATION // 60
    
    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token
        
        Args:
            data: Data to encode in token (user_id, email, etc.)
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create refresh token (longer expiration)"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=30)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return payload
            
        except JWTError as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_current_user_id(self, token: str = Depends(oauth2_scheme)) -> str:
        """
        Get current user ID from token
        
        Args:
            token: JWT token from Authorization header
            
        Returns:
            User ID string
        """
        payload = self.verify_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_id
    
    def authenticate_user(
        self,
        email: str,
        password: str,
        user_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with email and password
        
        Args:
            email: User email
            password: Plain text password
            user_data: User data from database (must include hashed_password)
            
        Returns:
            User data if authenticated, None otherwise
        """
        if not user_data:
            return None
        
        # Verify password
        hashed_password = user_data.get("hashed_password")
        if not hashed_password:
            return None
        
        if not encryption_service.verify_password(password, hashed_password):
            return None
        
        return user_data
    
    def create_api_key(self, customer_id: str, name: str) -> str:
        """
        Create API key for programmatic access
        
        Args:
            customer_id: Customer ID
            name: API key name/description
            
        Returns:
            API key string
        """
        data = {
            "sub": customer_id,
            "type": "api_key",
            "name": name,
        }
        
        # API keys don't expire
        token = jwt.encode(
            data,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return f"bank_{''.join([c for c in token[:40]])}"
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """
        Verify API key and return customer ID
        
        Args:
            api_key: API key string
            
        Returns:
            Customer ID if valid, None otherwise
        """
        try:
            # Extract JWT from API key
            if not api_key.startswith("bank_"):
                return None
            
            token = api_key[5:]  # Remove "bank_" prefix
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            if payload.get("type") != "api_key":
                return None
            
            return payload.get("sub")
            
        except JWTError:
            return None


# Global instance
auth_service = AuthService()


# Dependency for routes
async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """FastAPI dependency to get current user"""
    return auth_service.get_current_user_id(token)
