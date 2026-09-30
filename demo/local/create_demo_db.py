#!/usr/bin/env python3
"""
Idempotent setup script: seeds the local demo user into the database that
``DATABASE_URL`` points at.

Run once during container startup (no-op if the user already exists):
    python demo/local/create_demo_db.py [--api-key sk-ant-...]

DATABASE_URL must be set; the docker entrypoint populates it before invoking
this script. Outside Docker, point it at a local Postgres for parity.
"""
import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi_users.password import PasswordHelper
from sqlalchemy import func, select

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env", override=True)

from terralingua.server.auth import encrypt_api_key
from terralingua.server.models import User
from terralingua.server.user_db import async_session_maker, create_db_and_tables


async def create(email: str, username: str, password: str, api_key: str | None) -> None:
    await create_db_and_tables()

    encrypted_key = encrypt_api_key(api_key) if api_key else None
    hashed = PasswordHelper().hash(password)

    async with async_session_maker() as session:
        existing = await session.scalar(
            select(User).where(func.lower(User.email) == email.lower())
        )
        if existing is not None:
            print(f"[create_demo_db] Demo user already exists (id={existing.id}). Skipping.")
            return

        user = User(
            id=uuid.uuid4(),
            email=email,
            username=username,
            hashed_password=hashed,
            is_active=True,
            is_verified=True,
            is_superuser=False,
            encrypted_api_key=encrypted_key,
            created_at=datetime.now(timezone.utc),
        )
        session.add(user)
        await session.commit()

    print(f"[create_demo_db] Created demo user: email={email}, username={username}, api_key={'set' if api_key else 'none'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Seed the local demo user into Postgres (idempotent).")
    p.add_argument("--email",    default=os.environ.get("DEMO_USER_EMAIL", "demo@demo.com"))
    p.add_argument("--username", default=os.environ.get("DEMO_USER_NAME",  "demo"))
    p.add_argument("--password", default=os.environ.get("DEMO_USER_PASS",  "demopass123"))
    p.add_argument("--api-key",  default=None, help="Anthropic API key to encrypt and store for the demo user")
    args = p.parse_args()
    asyncio.run(create(args.email, args.username, args.password, args.api_key))


if __name__ == "__main__":
    main()
