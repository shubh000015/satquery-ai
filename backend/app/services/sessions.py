"""Session store backing the workspace history sidebar.

Sessions are held in memory and mirrored to .data/sessions as JSON so a dev
reload does not wipe the chat history the UI is showing.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.schemas.agent import QueryResult
from app.schemas.imagery import InputMode

_MAX_RESULTS_PER_SESSION = 50


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _title_from_query(query: str) -> str:
    words = query.split()
    title = " ".join(words[:5])
    return f"{title}..." if len(words) > 5 else title or "New analysis"


class Session:
    def __init__(self, session_id: str, mode: InputMode = "single", title: str = "New analysis") -> None:
        self.id = session_id
        self.title = title
        self.mode: InputMode = mode
        self.asset_ids: list[str] = []
        self.results: list[QueryResult] = []
        self.created_at = _now()
        self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "mode": self.mode,
            "assetIds": self.asset_ids,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "results": [r.model_dump(by_alias=True) for r in self.results],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Session":
        session = cls(raw["id"], raw.get("mode", "single"), raw.get("title", "New analysis"))
        session.asset_ids = list(raw.get("assetIds", []))
        session.created_at = raw.get("createdAt", _now())
        session.updated_at = raw.get("updatedAt", session.created_at)
        session.results = [QueryResult.model_validate(r) for r in raw.get("results", [])]
        return session


class SessionStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._load_existing()

    def _path(self, session_id: str):
        return self.settings.session_dir / f"{session_id}.json"

    def _load_existing(self) -> None:
        for path in self.settings.session_dir.glob("*.json"):
            try:
                self._sessions[path.stem] = Session.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue

    def _persist(self, session: Session) -> None:
        try:
            self._path(session.id).write_text(
                json.dumps(session.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass

    def create(self, mode: InputMode = "single", title: str | None = None) -> Session:
        session = Session(uuid.uuid4().hex[:12], mode, title or "New analysis")
        with self._lock:
            self._sessions[session.id] = session
            self._persist(session)
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None, mode: InputMode = "single") -> Session:
        if session_id:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing
        return self.create(mode)

    def list(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)

    def attach_assets(self, session_id: str, asset_ids: list[str], mode: InputMode) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            for asset_id in asset_ids:
                if asset_id not in session.asset_ids:
                    session.asset_ids.append(asset_id)
            session.mode = mode
            session.updated_at = _now()
            self._persist(session)
            return session

    def add_result(self, session_id: str, result: QueryResult) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.title == "New analysis" and result.query:
                session.title = _title_from_query(result.query)
            session.results.append(result)
            del session.results[:-_MAX_RESULTS_PER_SESSION]
            session.mode = result.mode
            session.updated_at = _now()
            self._persist(session)
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
            self._path(session_id).unlink(missing_ok=True)
            return existed


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
