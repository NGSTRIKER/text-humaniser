"""Lightweight local review interface (Stage 10).

Run:  python -m humanizer.web            (opens http://127.0.0.1:8765)

Standard library only (``http.server``); binds to localhost by default.
"""

from humanizer.web.server import ReviewApp, make_server, serve

__all__ = ["ReviewApp", "make_server", "serve"]
