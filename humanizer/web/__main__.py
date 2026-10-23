"""``python -m humanizer.web [--host 127.0.0.1] [--port 8765] [--feedback data/feedback.jsonl]``"""

from __future__ import annotations

import argparse

from humanizer.config import RewriterConfig
from humanizer.logging_config import setup_logging
from humanizer.web.server import serve


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local review interface for the rewriter.")
    p.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: localhost only).")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--feedback", default="data/feedback.jsonl", help="Where feedback is stored (JSONL).")
    p.add_argument("--no-browser", action="store_true", help="Do not open a browser tab.")
    args = p.parse_args(argv)
    config = RewriterConfig()
    setup_logging(level=config.log_level, log_file=config.log_file)
    serve(config, host=args.host, port=args.port, feedback_path=args.feedback, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
