"""Shared application state.

A small container that owns the long-lived, GUI-agnostic services so that every
tab consumes the *same* Spot connection and capture root. Future reconstruction
tabs can read ``ctx.spot`` / ``ctx.sessions`` without re-creating anything.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.capture.capture_session import SessionManager
from src.capture.spot_service import SpotService
from src.config import Config


@dataclass
class AppContext:
    config: Config
    spot: SpotService
    sessions: SessionManager

    @classmethod
    def create(cls, config: Config) -> "AppContext":
        return cls(
            config=config,
            spot=SpotService(config),
            sessions=SessionManager(config.output_dir),
        )
