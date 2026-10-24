"""JSON API + static files for the review interface.

Endpoints (all JSON):

    GET  /api/health                         backend status
    POST /api/rewrite                        {text, intensity, mode, seed, protected[], tone, strictness}
    GET  /api/session/<id>                   current session view
    POST /api/session/<id>/edit              {edit_id, decision: accept|reject|reset}
    POST /api/session/<id>/undo  | /redo
    POST /api/session/<id>/feedback          {verdict, final_text, rating, comment}
    GET  /api/feedback/summary               acceptance rates per transform type

Only one rewrite runs at a time (one local model in memory, plan §2).
"""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

from humanizer.config import Intensity, RewriterConfig
from humanizer.feedback import FeedbackStore, summarize
from humanizer.logging_config import get_logger
from humanizer.schemas import RewriteRequest
from humanizer.session import ReviewSession, SessionStore

log = get_logger("web")

STATIC = Path(__file__).parent / "static"
MAX_BODY = 1_000_000          # bytes
MAX_TEXT = 50_000             # characters per rewrite
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


PipelineFactory = Callable[[RewriterConfig], Any]


def _default_factory(cfg: RewriterConfig):
    from humanizer.rewriter import RewriterPipeline
    return RewriterPipeline(cfg)


class ReviewApp:
    """Request handling logic, separate from HTTP plumbing (easy to test)."""

    def __init__(self, config: Optional[RewriterConfig] = None, *, feedback_path: str | Path = "data/feedback.jsonl",
                 pipeline_factory: PipelineFactory = _default_factory, health: Optional[Callable[[], dict]] = None) -> None:
        self.config = config or RewriterConfig()
        self.factory = pipeline_factory
        self.sessions = SessionStore()
        self.feedback = FeedbackStore(feedback_path)
        self._health = health
        self._run_lock = threading.Lock()
        self._validator = None

    # ── helpers ──────────────────────────────────────────────────
    @property
    def validator(self):
        if self._validator is None and self.config.validation_enabled:
            try:
                from humanizer.validation import Validator
                self._validator = Validator(self.config)
            except Exception as exc:  # pragma: no cover
                log.warning("Validator unavailable: %s", exc)
        return self._validator

    def health(self) -> dict[str, Any]:
        if self._health:
            return self._health()
        from humanizer import analyzer
        ollama = False
        models: list[str] = []
        try:
            from humanizer.llm_backend import OllamaBackend
            b = OllamaBackend(self.config)
            ollama = b.health_check()
            resp = b._client.get("/api/tags")
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
            b._client.close()
        except Exception:
            pass
        return {"ollama": ollama, "model": self.config.model_name, "models": models, "analyzer": analyzer.is_available()}

    # ── actions ──────────────────────────────────────────────────
    def rewrite(self, body: dict[str, Any]) -> dict[str, Any]:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, "Please enter some text.")
        if len(text) > MAX_TEXT:
            raise ApiError(413, f"Text is too long (max {MAX_TEXT} characters).")
        mode = body.get("mode", "hybrid")
        if mode not in ("rules", "hybrid", "llm"):
            raise ApiError(400, "mode must be rules, hybrid or llm")
        try:
            intensity = Intensity(body.get("intensity") or self.config.intensity.value)
        except ValueError:
            raise ApiError(400, "intensity must be light, moderate or heavy")
        seed = body.get("seed")
        if seed in ("", None):
            seed = None
        elif not isinstance(seed, int):
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                raise ApiError(400, "seed must be a whole number")
        protected = [p.strip() for p in body.get("protected", []) if isinstance(p, str) and p.strip()][:50]
        overrides: dict[str, Any] = {"intensity": intensity, "rules_only": mode == "rules",
                                     "rules_enabled": mode != "llm", "seed": seed}
        if body.get("style") in ("neutral", "casual", "professional", "academic"):
            overrides["style"] = body["style"]
        if body.get("model") and isinstance(body["model"], str) and body["model"].strip():
            overrides["model_name"] = body["model"].strip()
        if body.get("tone"):
            overrides["tone"] = str(body["tone"])[:200]
        if body.get("strictness") in ("low", "medium", "high"):
            overrides["semantic_strictness"] = body["strictness"]
        if body.get("vocabulary_variation") in ("low", "medium", "high"):
            overrides["vocabulary_variation"] = body["vocabulary_variation"]
        if body.get("restructuring") in ("low", "medium", "high"):
            overrides["restructuring"] = body["restructuring"]
        if "sentence_reordering" in body:
            overrides["sentence_reordering"] = bool(body["sentence_reordering"])
        elif intensity == Intensity.HEAVY:
            overrides["sentence_reordering"] = True
        cfg = self.config.model_copy(update=overrides)

        with self._run_lock:
            pipeline = self.factory(cfg)
            try:
                result = pipeline.rewrite_text(RewriteRequest(text=text, intensity=intensity.value,
                                                              protected_terms=protected, seed=seed))
            finally:
                close = getattr(getattr(pipeline, "backend", None), "close", None)
                if callable(close):
                    close()
        settings = {"intensity": intensity.value, "mode": mode, "seed": seed, "tone": cfg.tone,
                    "protected": protected, "strictness": cfg.semantic_strictness}
        session = self.sessions.add(ReviewSession.from_result(result, settings, self.validator))
        return session.to_dict()

    def session(self, sid: str) -> ReviewSession:
        try:
            return self.sessions.get(sid)
        except KeyError:
            raise ApiError(404, "This review session has expired. Please rewrite again.")

    def edit(self, sid: str, body: dict[str, Any]) -> dict[str, Any]:
        s = self.session(sid)
        decision = body.get("decision")
        if decision not in ("accept", "reject", "reset"):
            raise ApiError(400, "decision must be accept, reject or reset")
        try:
            s.decide(str(body.get("edit_id")), decision)
        except KeyError as exc:
            raise ApiError(404, str(exc))
        except ValueError as exc:
            raise ApiError(409, str(exc))
        return s.to_dict()

    def submit_feedback(self, sid: str, body: dict[str, Any]) -> dict[str, Any]:
        s = self.session(sid)
        rating = body.get("rating")
        try:
            record = s.feedback(self.feedback, verdict=body.get("verdict", ""),
                                final_text=body.get("final_text") or None,
                                rating=int(rating) if rating not in (None, "") else None,
                                comment=str(body.get("comment", "")))
        except (TypeError, ValueError) as exc:
            raise ApiError(400, str(exc))
        return {"saved": True, "id": record.id, "final_validation": record.validation}

    # ── routing ──────────────────────────────────────────────────
    def handle(self, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if method == "GET" and path == "/api/health":
            return self.health()
        if method == "GET" and path == "/api/feedback/summary":
            return summarize(self.feedback.all())
        if method == "POST" and path == "/api/rewrite":
            return self.rewrite(body)
        m = re.fullmatch(r"/api/session/([0-9a-f]{12})(?:/(edit|undo|redo|feedback))?", path)
        if m:
            sid, action = m.groups()
            if method == "GET" and action is None:
                return self.session(sid).to_dict()
            if method == "POST" and action == "edit":
                return self.edit(sid, body)
            if method == "POST" and action in ("undo", "redo"):
                s = self.session(sid)
                (s.undo if action == "undo" else s.redo)()
                return s.to_dict()
            if method == "POST" and action == "feedback":
                return self.submit_feedback(sid, body)
        raise ApiError(404, "not found")


def _handler_for(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RewriterReview/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # route through our logger
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: bytes, ctype: str) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                log.debug("Client disconnected before response was sent")

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _api(self, method: str) -> None:
            body: dict[str, Any] = {}
            if method == "POST":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                if length > MAX_BODY:
                    return self._json(413, {"error": "request too large"})
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    return self._json(400, {"error": "invalid JSON"})
            try:
                self._json(200, app.handle(method, self.path.split("?")[0], body))
            except (BrokenPipeError, ConnectionResetError):
                log.debug("Client connection dropped")
            except ApiError as exc:
                self._json(exc.status, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - last resort
                log.exception("Request failed")
                self._json(500, {"error": f"internal error: {exc}"})

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path.startswith("/api/"):
                return self._api("GET")
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            file = (STATIC / name).resolve()
            if STATIC.resolve() not in file.parents or not file.is_file():
                return self._json(404, {"error": "not found"})
            self._send(200, file.read_bytes(), CONTENT_TYPES.get(file.suffix, "application/octet-stream"))

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/api/"):
                return self._json(404, {"error": "not found"})
            self._api("POST")

    return Handler


def make_server(app: ReviewApp, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler_for(app))


def serve(config: Optional[RewriterConfig] = None, *, host: str = "127.0.0.1", port: int = 8765,
          feedback_path: str | Path = "data/feedback.jsonl", open_browser: bool = True) -> None:
    app = ReviewApp(config, feedback_path=feedback_path)
    httpd = make_server(app, host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    if host not in ("127.0.0.1", "localhost"):
        log.warning("Listening on %s — anyone on your network can use it.", host)
    log.info("Review interface running at %s  (Ctrl+C to stop)", url)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        httpd.server_close()
