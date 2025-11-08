"""Cryptographic signing and verification."""

import base64
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

from .validation import ContentManifest, SignatureMetadata


class SigningError(Exception):
    """Raised when signing operations fail."""
    pass


class VerificationError(Exception):
    """Raised when signature verification fails."""
    pass


class KeyManager:
    """Manages Ed25519 key pairs for signing."""
    
    def __init__(self, private_key: Optional[Ed25519PrivateKey] = None):
        """
        Initialize key manager.
        
        Args:
            private_key: Optional existing private key; generates new if None
        """
        self._private_key = private_key or Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
    
    @property
    def private_key(self) -> Ed25519PrivateKey:
        """Get private key."""
        return self._private_key
    
    @property
    def public_key(self) -> Ed25519PublicKey:
        """Get public key."""
        return self._public_key
    
    def export_private_key(self, password: Optional[bytes] = None) -> bytes:
        """
        Export private key in PEM format.
        
        Args:
            password: Optional password for encryption
            
        Returns:
            PEM-encoded private key
        """
        if password:
            encryption = serialization.BestAvailableEncryption(password)
        else:
            encryption = serialization.NoEncryption()
        
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )
    
    def export_public_key(self) -> bytes:
        """
        Export public key in PEM format.
        
        Returns:
            PEM-encoded public key
        """
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    
    def export_public_key_base64(self) -> str:
        """
        Export public key as base64-encoded raw bytes.
        
        Returns:
            Base64-encoded public key
        """
        raw_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw_bytes).decode("utf-8")
    
    @classmethod
    def from_private_key_file(
        cls, 
        path: Path, 
        password: Optional[bytes] = None
    ) -> "KeyManager":
        """
        Load key manager from private key file.
        
        Args:
            path: Path to PEM-encoded private key
            password: Optional password for encrypted keys
            
        Returns:
            KeyManager instance
        """
        with open(path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=password,
            )
        
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError("Key file does not contain Ed25519 private key")
        
        return cls(private_key=private_key)
    
    @classmethod
    def from_public_key_file(cls, path: Path) -> Ed25519PublicKey:
        """
        Load public key from file.
        
        Args:
            path: Path to PEM-encoded public key
            
        Returns:
            Ed25519PublicKey instance
        """
        with open(path, "rb") as f:
            public_key = serialization.load_pem_public_key(f.read())
        
        if not isinstance(public_key, Ed25519PublicKey):
            raise ValueError("Key file does not contain Ed25519 public key")
        
        return public_key
    
    @classmethod
    def from_public_key_base64(cls, b64_key: str) -> Ed25519PublicKey:
        """
        Load public key from base64-encoded raw bytes.
        
        Args:
            b64_key: Base64-encoded public key
            
        Returns:
            Ed25519PublicKey instance
        """
        raw_bytes = base64.b64decode(b64_key)
        return Ed25519PublicKey.from_public_bytes(raw_bytes)
    
    def save_keys(
        self,
        private_path: Path,
        public_path: Path,
        password: Optional[bytes] = None,
    ) -> None:
        """
        Save key pair to files.
        
        Args:
            private_path: Path for private key
            public_path: Path for public key
            password: Optional password for private key encryption
        """
        private_path.write_bytes(self.export_private_key(password))
        private_path.chmod(0o600)  # Restrict permissions
        
        public_path.write_bytes(self.export_public_key())


class Signer:
    """Signs manifests and files."""
    
    def __init__(self, key_manager: KeyManager):
        """
        Initialize signer.
        
        Args:
            key_manager: KeyManager with private key
        """
        self.key_manager = key_manager
    
    def sign_data(self, data: bytes) -> bytes:
        """
        Sign arbitrary data.
        
        Args:
            data: Data to sign
            
        Returns:
            Raw signature bytes
        """
        try:
            return self.key_manager.private_key.sign(data)
        except Exception as e:
            raise SigningError(f"Failed to sign data: {e}") from e
    
    def sign_manifest(self, manifest: ContentManifest) -> ContentManifest:
        """
        Sign a manifest and update its signature field.
        
        Args:
            manifest: Manifest to sign
            
        Returns:
            New manifest with signature
        """
        # Get canonical JSON (excludes signature field)
        canonical_json = manifest.to_canonical_json()
        data_to_sign = canonical_json.encode("utf-8")
        
        # Sign
        signature_bytes = self.sign_data(data_to_sign)
        
        # Encode signature and public key as base64
        signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")
        public_key_b64 = self.key_manager.export_public_key_base64()
        
        # Create new signature metadata
        sig_metadata = SignatureMetadata(
            algorithm="ed25519",
            public_key=public_key_b64,
            signature=signature_b64,
        )
        
        # Return new manifest with signature
        return manifest.model_copy(update={"signature": sig_metadata})
    
    def sign_file(self, path: Path) -> Tuple[bytes, str]:
        """
        Create detached signature for a file.
        
        Args:
            path: Path to file
            
        Returns:
            Tuple of (signature_bytes, base64_encoded_signature)
        """
        with open(path, "rb") as f:
            data = f.read()
        
        signature_bytes = self.sign_data(data)
        signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")
        
        return signature_bytes, signature_b64


class Verifier:
    """Verifies signatures on manifests and files."""
    
    @staticmethod
    def verify_data(
        data: bytes,
        signature: bytes,
        public_key: Ed25519PublicKey,
    ) -> bool:
        """
        Verify signature on data.
        
        Args:
            data: Original data
            signature: Signature bytes
            public_key: Public key for verification
            
        Returns:
            True if signature is valid
            
        Raises:
            VerificationError: If signature is invalid
        """
        try:
            public_key.verify(signature, data)
            return True
        except Exception as e:
            raise VerificationError(f"Signature verification failed: {e}") from e
    
    @staticmethod
    def verify_manifest(manifest: ContentManifest) -> bool:
        """
        Verify manifest signature.
        
        Args:
            manifest: Manifest to verify
            
        Returns:
            True if signature is valid
            
        Raises:
            VerificationError: If signature is invalid
        """
        # Decode public key and signature
        try:
            public_key = KeyManager.from_public_key_base64(
                manifest.signature.public_key
            )
            signature_bytes = base64.b64decode(manifest.signature.signature)
        except Exception as e:
            raise VerificationError(f"Failed to decode signature data: {e}") from e
        
        # Get canonical JSON (excludes signature)
        canonical_json = manifest.to_canonical_json()
        data = canonical_json.encode("utf-8")
        
        return Verifier.verify_data(data, signature_bytes, public_key)
    
    @staticmethod
    def verify_file(
        path: Path,
        signature_b64: str,
        public_key: Ed25519PublicKey,
    ) -> bool:
        """
        Verify detached file signature.
        
        Args:
            path: Path to file
            signature_b64: Base64-encoded signature
            public_key: Public key for verification
            
        Returns:
            True if signature is valid
            
        Raises:
            VerificationError: If signature is invalid
        """
        with open(path, "rb") as f:
            data = f.read()
        
        signature_bytes = base64.b64decode(signature_b64)
        
        return Verifier.verify_data(data, signature_bytes, public_key)


def generate_keypair(
    output_dir: Path,
    key_name: str = "signing_key",
    password: Optional[bytes] = None,
) -> Tuple[Path, Path]:
    """
    Generate and save a new Ed25519 key pair.
    
    Args:
        output_dir: Directory to save keys
        key_name: Base name for key files
        password: Optional password for private key
        
    Returns:
        Tuple of (private_key_path, public_key_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    km = KeyManager()
    
    private_path = output_dir / f"{key_name}.pem"
    public_path = output_dir / f"{key_name}.pub"
    
    km.save_keys(private_path, public_path, password)
    
    return private_path, public_path
