"""Encryption service for sensitive data"""

from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import logging

from ..utils.config import settings

logger = logging.getLogger(__name__)


class EncryptionService:
    """Handle encryption/decryption of sensitive data (PII, passwords, etc.)"""
    
    def __init__(self):
        self.fernet = self._initialize_fernet()
    
    def _initialize_fernet(self) -> Fernet:
        """Initialize Fernet cipher with key from settings"""
        try:
            # Use encryption key from settings or generate one
            key = settings.ENCRYPTION_KEY
            
            # If key is not proper base64, derive it
            if len(key) != 44 or not key.endswith('='):
                # Derive proper key from provided key
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'banking_llm_salt_2026',  # In production, use unique salt per env
                    iterations=100000,
                    backend=default_backend()
                )
                derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
                return Fernet(derived_key)
            
            return Fernet(key.encode())
            
        except Exception as e:
            logger.warning(f"Failed to initialize encryption key: {e}. Generating new key.")
            # Generate a new key if initialization fails
            key = Fernet.generate_key()
            return Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data
        
        Args:
            data: String or bytes to encrypt
            
        Returns:
            Encrypted bytes
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            encrypted = self.fernet.encrypt(data)
            return encrypted
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """
        Decrypt data
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            Decrypted string
        """
        try:
            decrypted = self.fernet.decrypt(encrypted_data)
            return decrypted.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_ssn(self, ssn: str) -> bytes:
        """Encrypt Social Security Number"""
        # Remove any formatting
        ssn_clean = ssn.replace('-', '').replace(' ', '')
        return self.encrypt(ssn_clean)
    
    def decrypt_ssn(self, encrypted_ssn: bytes) -> str:
        """Decrypt SSN and format it"""
        ssn = self.decrypt(encrypted_ssn)
        # Format as XXX-XX-XXXX
        return f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:]}"
    
    def encrypt_address(self, address: dict) -> bytes:
        """Encrypt address data"""
        import json
        address_str = json.dumps(address)
        return self.encrypt(address_str)
    
    def decrypt_address(self, encrypted_address: bytes) -> dict:
        """Decrypt address data"""
        import json
        address_str = self.decrypt(encrypted_address)
        return json.loads(address_str)
    
    def hash_password(self, password: str) -> str:
        """
        Hash password for storage (one-way)
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        from passlib.context import CryptContext
        
        # Bcrypt has 72 byte limit, truncate if needed
        if len(password.encode('utf-8')) > 72:
            password = password[:72]
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Stored hash
            
        Returns:
            True if password matches
        """
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key"""
        key = Fernet.generate_key()
        return key.decode('utf-8')


# Global instance
encryption_service = EncryptionService()
