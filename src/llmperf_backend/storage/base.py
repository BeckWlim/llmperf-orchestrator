"""Shared SQLAlchemy metadata for independently owned storage modules."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
