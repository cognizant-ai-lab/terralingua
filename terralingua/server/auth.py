"""Fernet helpers for encrypting API keys at rest."""
import os

from cryptography.fernet import Fernet, InvalidToken

_fernet_instance: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet_instance
    if _fernet_instance is None:
        key = os.environ.get("FERNET_SECRET_KEY", "")
        if not key:
            raise ValueError(
                "FERNET_SECRET_KEY is not set. "
                "Generate one with: "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" "
                "and add it to your .env file."
            )
        _fernet_instance = Fernet(key.encode())
    return _fernet_instance


def encrypt_api_key(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_api_key(encrypted: str) -> str:
    try:
        return _get_fernet().decrypt(encrypted.encode()).decode()
    except InvalidToken as e:
        raise ValueError("Failed to decrypt API key — FERNET_SECRET_KEY may have changed.") from e
