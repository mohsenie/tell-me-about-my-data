"""Tiny ANSI color helper. No dependency. Auto-disables when not a TTY or when
NO_COLOR is set, so piped/redirected output stays clean.
"""
from __future__ import annotations

import os
import sys

_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_CODES = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "red": "\033[31m",
    "gray": "\033[90m",
}


def c(text: str, *styles: str) -> str:
    if not _ENABLED or not styles:
        return text
    prefix = "".join(_CODES.get(s, "") for s in styles)
    return f"{prefix}{text}{_CODES['reset']}"


# Semantic roles (so callers don't hardcode colors)
def user(text: str) -> str:       return c(text, "cyan", "bold")
def assistant(text: str) -> str:  return c(text, "green")
def system(text: str) -> str:     return c(text, "gray")
def heading(text: str) -> str:    return c(text, "yellow", "bold")
def warn(text: str) -> str:       return c(text, "yellow")
def error(text: str) -> str:      return c(text, "red")
def info(text: str) -> str:       return c(text, "blue")
