from __future__ import annotations


class AnjukeException(Exception):
    """Base exception for crawler runtime errors."""


class ParseError(AnjukeException):
    """Raised when parser cannot extract required fields."""


class BanError(AnjukeException):
    """Raised when request is blocked by anti-bot."""


class NetworkError(AnjukeException):
    """Raised when network interaction fails."""


class PageCrashError(AnjukeException):
    """Raised when browser page crashes (Out of Memory etc.)."""
