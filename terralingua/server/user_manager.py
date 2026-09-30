"""fastapi-users wiring: UserManager, auth backend, dependency helpers."""
import logging
import os
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib
from fastapi import Depends
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, CookieTransport
from fastapi_users.authentication.strategy import DatabaseStrategy
from fastapi_users.exceptions import InvalidPasswordException, UserAlreadyExists
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession

from terralingua.server.models import User
from terralingua.server.user_db import get_access_token_db, get_async_session

logger = logging.getLogger(__name__)


async def _send_email(to: str, subject: str, plain: str, html: str) -> None:
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    from_addr = os.environ.get("SMTP_FROM", smtp_user)
    if not smtp_user or not smtp_password or not smtp_host:
        logger.warning("SMTP_USER / SMTP_PASSWORD / SMTP_HOST not set — skipping email to %s", to)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        await aiosmtplib.send(
            msg,
            hostname=smtp_host,
            port=smtp_port,
            start_tls=True,
            username=smtp_user,
            password=smtp_password,
        )
    except Exception:
        logger.exception("Failed to send email to %s", to)


async def _send_verification_email(email: str, token: str) -> None:
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    url = f"{base_url}/verify-email?token={token}"
    await _send_email(
        to=email,
        subject="Verify your OpenGridWorld account",
        plain=f"Please verify your email address by visiting:\n{url}\n\nThis link expires in 24 hours.",
        html=f"""<p>Thanks for registering!</p>
<p><a href="{url}" style="padding:8px 16px;background:#4f46e5;color:#fff;border-radius:4px;text-decoration:none">Verify my email</a></p>
<p>Or copy this link: {url}</p>
<p>The link expires in 24 hours.</p>""",
    )


async def _send_reset_password_email(email: str, token: str) -> None:
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    url = f"{base_url}/login?reset_token={token}"
    await _send_email(
        to=email,
        subject="Reset your OpenGridWorld password",
        plain=f"You requested a password reset. Visit this link to set a new password:\n{url}\n\nThis link expires in 1 hour. If you didn't request this, ignore this email.",
        html=f"""<p>You requested a password reset.</p>
<p><a href="{url}" style="padding:8px 16px;background:#4f46e5;color:#fff;border-radius:4px;text-decoration:none">Reset my password</a></p>
<p>Or copy this link: {url}</p>
<p>This link expires in 1 hour. If you didn't request this, ignore this email.</p>""",
    )


def _get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET", "")
    if not secret:
        raise ValueError(
            "JWT_SECRET is not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
            "and add it to your .env file."
        )
    return secret


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    @property
    def reset_password_token_secret(self) -> str:
        return _get_jwt_secret()

    @property
    def verification_token_secret(self) -> str:
        return _get_jwt_secret()

    async def validate_password(self, password: str, user=None) -> None:
        if len(password) < 8:
            raise InvalidPasswordException("Password must be at least 8 characters.")
        if password.isdigit():
            raise InvalidPasswordException("Password must contain at least one letter.")

    async def on_after_register(self, user: User, request=None) -> None:
        await self.request_verify(user, request)

    async def on_after_request_verify(self, user: User, token: str, request=None) -> None:
        await _send_verification_email(user.email, token)

    async def on_after_forgot_password(self, user: User, token: str, request=None) -> None:
        await _send_reset_password_email(user.email, token)

    async def create(self, user_create, safe: bool = False, request=None):
        result = await self.user_db.session.execute(
            sa_select(User).where(User.username == user_create.username)
        )
        if result.scalar_one_or_none() is not None:
            raise UserAlreadyExists()
        return await super().create(user_create, safe=safe, request=request)


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


async def _get_database_strategy(
    access_token_db=Depends(get_access_token_db),
) -> DatabaseStrategy:
    return DatabaseStrategy(access_token_db, lifetime_seconds=86400)


def cookie_secure_default() -> bool:
    """Whether to set the ``Secure`` flag on auth cookies.

    Defaults to True so production deployments (where TLS is terminated at an
    upstream proxy and uvicorn never sees the cert) are safe by default. Set
    ``COOKIE_SECURE=false`` for local HTTP development, or pass ``--dev`` to
    terralingua-dashboard.
    """
    return os.environ.get("COOKIE_SECURE", "true").lower() != "false"


def _get_auth_backend() -> AuthenticationBackend:
    transport = CookieTransport(
        cookie_name="ogw_user_session",
        cookie_max_age=86400,
        cookie_httponly=True,
        cookie_samesite="lax",
        cookie_secure=cookie_secure_default(),
    )
    return AuthenticationBackend(name="cookie", transport=transport, get_strategy=_get_database_strategy)


auth_backend = _get_auth_backend()
fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])
current_user = fastapi_users.current_user(active=True)
optional_current_user = fastapi_users.current_user(active=True, optional=True)
current_verified_user = fastapi_users.current_user(active=True, verified=True)
superuser = fastapi_users.current_user(active=True, superuser=True)
