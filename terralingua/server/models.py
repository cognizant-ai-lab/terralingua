"""SQLAlchemy User model and Pydantic schemas for fastapi-users."""
import uuid
from datetime import datetime

from fastapi_users import schemas
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyBaseAccessTokenTableUUID
from pydantic import Field, computed_field
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    encrypted_api_key: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=func.now()
    )


class UserRead(schemas.BaseUser[uuid.UUID]):
    username: str
    encrypted_api_key: str | None = Field(default=None, exclude=True)

    @computed_field
    @property
    def has_api_key(self) -> bool:
        return self.encrypted_api_key is not None

    model_config = {"from_attributes": True}


class UserCreate(schemas.BaseUserCreate):
    username: str


class UserUpdate(schemas.BaseUserUpdate):
    username: str | None = None
