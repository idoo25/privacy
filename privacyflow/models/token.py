"""
Privacy Token Generator - generates anonymous, non-reversible person tokens.

Security features:
- Daily rotating salt (breaks cross-day tracking)
- SHA-256 hashing (irreversible)
- Feature quantization (reduces precision)
"""

import hashlib
import os
from datetime import date
from typing import Optional

import numpy as np


class PrivacyTokenGenerator:
    """
    Generates anonymous, non-reversible person tokens.
    
    Security features:
    - Daily rotating salt (breaks cross-day tracking)
    - SHA-256 hashing (irreversible)
    - Feature quantization (reduces precision)
    """
    
    def __init__(self):
        self._daily_salt: Optional[bytes] = None
        self._salt_date: Optional[date] = None
    
    @property
    def daily_salt(self) -> bytes:
        """Get or generate daily salt."""
        today = date.today()
        if self._salt_date != today:
            self._daily_salt = os.urandom(32)
            self._salt_date = today
        return self._daily_salt
    
    def rotate_salt(self) -> None:
        """Force salt rotation (used during daily purge)."""
        self._daily_salt = os.urandom(32)
        self._salt_date = date.today()
    
    def generate_token(self, 
                       face_embedding: np.ndarray,
                       body_proportions: np.ndarray,
                       appearance_features: np.ndarray) -> str:
        """
        Generate anonymous person token from features.
        
        Args:
            face_embedding: 64-dim face features
            body_proportions: 8-dim body ratios
            appearance_features: 32-dim appearance vector
        
        Returns:
            str: 16-character anonymous token (truncated hash)
        """
        # Quantize features to reduce precision
        face_quant = self._quantize(face_embedding, bins=16)
        body_quant = self._quantize(body_proportions, bins=8)
        appearance_quant = self._quantize(appearance_features, bins=16)
        
        # Concatenate all features
        combined = np.concatenate([face_quant, body_quant, appearance_quant])
        
        # Hash with daily salt
        feature_bytes = combined.tobytes()
        salted = self.daily_salt + feature_bytes
        hash_digest = hashlib.sha256(salted).hexdigest()
        
        # Return truncated hash (sufficient for daily uniqueness)
        return hash_digest[:16]
    
    def _quantize(self, features: np.ndarray, bins: int) -> np.ndarray:
        """Reduce feature precision through quantization."""
        # Ensure features are in [0, 1] range
        clipped = np.clip(features, 0, 1)
        return np.digitize(clipped, np.linspace(0, 1, bins))
