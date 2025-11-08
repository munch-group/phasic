"""Tests for signing module."""

import tempfile
from pathlib import Path

import pytest

from hybrid_p2p.signing import (
    KeyManager,
    Signer,
    Verifier,
    SigningError,
    VerificationError,
    generate_keypair,
)
from hybrid_p2p.validation import (
    ContentManifest,
    FileEntry,
    SignatureMetadata,
    ProvenanceInfo,
)


@pytest.fixture
def key_manager() -> KeyManager:
    """Create a key manager with fresh keys."""
    return KeyManager()


@pytest.fixture
def sample_manifest() -> ContentManifest:
    """Create a sample manifest."""
    return ContentManifest(
        content_id="test-id",
        version="1.0.0",
        name="test-package",
        files=[
            FileEntry(
                path="test.txt",
                size=100,
                sha256="a" * 64,
                mime_type="text/plain",
            )
        ],
        total_size=100,
        signature=SignatureMetadata(
            public_key="placeholder",
            signature="placeholder",
        ),
        provenance=ProvenanceInfo(uploader_id="alice"),
    )


class TestKeyManager:
    """Tests for KeyManager."""
    
    def test_generate_keys(self):
        """Test key generation."""
        km = KeyManager()
        assert km.private_key is not None
        assert km.public_key is not None
    
    def test_export_private_key(self, key_manager: KeyManager):
        """Test exporting private key."""
        pem = key_manager.export_private_key()
        assert b"BEGIN PRIVATE KEY" in pem
        assert b"END PRIVATE KEY" in pem
    
    def test_export_private_key_encrypted(self, key_manager: KeyManager):
        """Test exporting encrypted private key."""
        password = b"test-password"
        pem = key_manager.export_private_key(password=password)
        assert b"BEGIN ENCRYPTED PRIVATE KEY" in pem
    
    def test_export_public_key(self, key_manager: KeyManager):
        """Test exporting public key."""
        pem = key_manager.export_public_key()
        assert b"BEGIN PUBLIC KEY" in pem
        assert b"END PUBLIC KEY" in pem
    
    def test_export_public_key_base64(self, key_manager: KeyManager):
        """Test exporting public key as base64."""
        b64 = key_manager.export_public_key_base64()
        assert len(b64) > 0
        # Ed25519 public keys are 32 bytes -> 44 base64 chars
        assert len(b64) == 44
    
    def test_save_and_load_keys(self, key_manager: KeyManager, tmp_path: Path):
        """Test saving and loading keys."""
        private_path = tmp_path / "private.pem"
        public_path = tmp_path / "public.pem"
        
        # Save
        key_manager.save_keys(private_path, public_path)
        
        assert private_path.exists()
        assert public_path.exists()
        
        # Load
        loaded_km = KeyManager.from_private_key_file(private_path)
        
        # Should be able to sign with loaded key
        test_data = b"test data"
        sig1 = key_manager.private_key.sign(test_data)
        sig2 = loaded_km.private_key.sign(test_data)
        
        # Signatures will differ, but both should verify
        key_manager.public_key.verify(sig1, test_data)
        loaded_km.public_key.verify(sig2, test_data)
    
    def test_save_and_load_encrypted_keys(
        self,
        key_manager: KeyManager,
        tmp_path: Path
    ):
        """Test saving and loading encrypted keys."""
        private_path = tmp_path / "private.pem"
        public_path = tmp_path / "public.pem"
        password = b"test-password"
        
        # Save with encryption
        key_manager.save_keys(private_path, public_path, password=password)
        
        # Load with password
        loaded_km = KeyManager.from_private_key_file(private_path, password=password)
        
        # Verify keys work
        test_data = b"test data"
        sig = key_manager.private_key.sign(test_data)
        loaded_km.public_key.verify(sig, test_data)
    
    def test_load_public_key_from_base64(self, key_manager: KeyManager):
        """Test loading public key from base64."""
        b64 = key_manager.export_public_key_base64()
        loaded_pk = KeyManager.from_public_key_base64(b64)
        
        # Should be able to verify with loaded key
        test_data = b"test data"
        sig = key_manager.private_key.sign(test_data)
        loaded_pk.verify(sig, test_data)


class TestSigner:
    """Tests for Signer."""
    
    def test_sign_data(self, key_manager: KeyManager):
        """Test signing arbitrary data."""
        signer = Signer(key_manager)
        data = b"test data"
        signature = signer.sign_data(data)
        
        assert len(signature) == 64  # Ed25519 signatures are 64 bytes
        
        # Verify signature
        key_manager.public_key.verify(signature, data)
    
    def test_sign_manifest(self, key_manager: KeyManager, sample_manifest: ContentManifest):
        """Test signing a manifest."""
        signer = Signer(key_manager)
        signed_manifest = signer.sign_manifest(sample_manifest)
        
        # Should have signature
        assert signed_manifest.signature.algorithm == "ed25519"
        assert len(signed_manifest.signature.signature) > 0
        assert len(signed_manifest.signature.public_key) > 0
        
        # Original should be unchanged
        assert sample_manifest.signature.signature == "placeholder"
    
    def test_sign_file(self, key_manager: KeyManager, tmp_path: Path):
        """Test signing a file."""
        signer = Signer(key_manager)
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")
        
        sig_bytes, sig_b64 = signer.sign_file(test_file)
        
        assert len(sig_bytes) == 64
        assert len(sig_b64) > 0


class TestVerifier:
    """Tests for Verifier."""
    
    def test_verify_data(self, key_manager: KeyManager):
        """Test verifying data signature."""
        data = b"test data"
        signature = key_manager.private_key.sign(data)
        
        # Should verify
        assert Verifier.verify_data(data, signature, key_manager.public_key)
    
    def test_verify_invalid_signature(self, key_manager: KeyManager):
        """Test that invalid signature fails."""
        data = b"test data"
        wrong_sig = b"x" * 64
        
        with pytest.raises(VerificationError):
            Verifier.verify_data(data, wrong_sig, key_manager.public_key)
    
    def test_verify_manifest(self, key_manager: KeyManager, sample_manifest: ContentManifest):
        """Test verifying manifest signature."""
        signer = Signer(key_manager)
        signed_manifest = signer.sign_manifest(sample_manifest)
        
        # Should verify
        assert Verifier.verify_manifest(signed_manifest)
    
    def test_verify_tampered_manifest(
        self,
        key_manager: KeyManager,
        sample_manifest: ContentManifest
    ):
        """Test that tampered manifest fails verification."""
        signer = Signer(key_manager)
        signed_manifest = signer.sign_manifest(sample_manifest)
        
        # Tamper with content
        signed_manifest.name = "tampered"
        
        # Should fail
        with pytest.raises(VerificationError):
            Verifier.verify_manifest(signed_manifest)
    
    def test_verify_file(self, key_manager: KeyManager, tmp_path: Path):
        """Test verifying file signature."""
        signer = Signer(key_manager)
        
        # Create and sign file
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")
        
        _, sig_b64 = signer.sign_file(test_file)
        
        # Verify
        assert Verifier.verify_file(test_file, sig_b64, key_manager.public_key)
    
    def test_verify_wrong_file(self, key_manager: KeyManager, tmp_path: Path):
        """Test that wrong file fails verification."""
        signer = Signer(key_manager)
        
        # Create and sign file
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")
        
        _, sig_b64 = signer.sign_file(test_file)
        
        # Modify file
        test_file.write_bytes(b"modified content")
        
        # Should fail
        with pytest.raises(VerificationError):
            Verifier.verify_file(test_file, sig_b64, key_manager.public_key)


class TestGenerateKeypair:
    """Tests for generate_keypair helper."""
    
    def test_generate_keypair(self, tmp_path: Path):
        """Test generating keypair."""
        private_path, public_path = generate_keypair(tmp_path)
        
        assert private_path.exists()
        assert public_path.exists()
        assert private_path.name == "signing_key.pem"
        assert public_path.name == "signing_key.pub"
        
        # Should be loadable
        km = KeyManager.from_private_key_file(private_path)
        assert km.private_key is not None
    
    def test_generate_keypair_custom_name(self, tmp_path: Path):
        """Test generating keypair with custom name."""
        private_path, public_path = generate_keypair(
            tmp_path,
            key_name="custom_key",
        )
        
        assert private_path.name == "custom_key.pem"
        assert public_path.name == "custom_key.pub"
    
    def test_generate_keypair_encrypted(self, tmp_path: Path):
        """Test generating encrypted keypair."""
        password = b"test-password"
        private_path, public_path = generate_keypair(
            tmp_path,
            password=password,
        )
        
        # Should require password to load
        km = KeyManager.from_private_key_file(private_path, password=password)
        assert km.private_key is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
