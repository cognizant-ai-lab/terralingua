"""SQLAlchemy async engine and session for fastapi-users.

The user database is PostgreSQL, configured via the ``DATABASE_URL`` env var.
Example: ``postgresql+asyncpg://ogw:secret@localhost:5432/ogw_users``.

A leading ``postgresql://`` is normalized to ``postgresql+asyncpg://`` so the
common env-var form (matching the libpq URL most people are used to) works
without further escaping.
"""
import os
from functools import lru_cache

from fastapi import Depends
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyAccessTokenDatabase
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from terralingua.server.models import AccessToken, Base


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Point it at a Postgres instance, e.g. "
            "DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname"
        )
    # Allow libpq-style URLs without the async driver suffix.
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://"):]
    return url


@lru_cache(maxsize=None)
def _engine():
    """The engine is created on first use, so importing this module needs no DATABASE_URL."""
    return create_async_engine(_db_url())


@lru_cache(maxsize=None)
def _session_factory():
    return async_sessionmaker(_engine(), expire_on_commit=False)


def async_session_maker() -> AsyncSession:
    """Open a session on the shared engine."""
    return _session_factory()()


async def create_db_and_tables() -> None:
    # Serialize concurrent DDL across uvicorn workers — every worker runs
    # this in its lifespan startup, and without the lock they race inside
    # Postgres on type creation (pg_type_typname_nsp_index UniqueViolation
    # for the loser). The advisory lock is transactional, so it's released
    # automatically when the begin()-managed transaction commits.
    async with _engine().begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(727272)"))
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session


async def get_access_token_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyAccessTokenDatabase(session, AccessToken)
