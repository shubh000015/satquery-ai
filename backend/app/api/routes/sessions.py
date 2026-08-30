from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.schemas.api import SessionDetail, SessionSummary
from app.services.report import result_to_markdown, session_to_markdown
from app.services.sessions import Session, get_session_store
from app.services.storage import get_asset_store

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _summary(session: Session) -> SessionSummary:
    return SessionSummary(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        mode=session.mode,
        asset_count=len(session.asset_ids),
        query_count=len(session.results),
    )


def _require(session_id: str) -> Session:
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return session


@router.get("", response_model=list[SessionSummary])
def list_sessions() -> list[SessionSummary]:
    return [_summary(session) for session in get_session_store().list()]


@router.post("", response_model=SessionSummary)
def create_session() -> SessionSummary:
    return _summary(get_session_store().create())


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str) -> SessionDetail:
    session = _require(session_id)
    store = get_asset_store()
    assets = [asset for asset in (store.get(aid) for aid in session.asset_ids) if asset is not None]
    return SessionDetail(
        **_summary(session).model_dump(),
        assets=assets,
        results=session.results,
    )


@router.delete("/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    store = get_session_store()
    session = store.get(session_id)
    if session is not None:
        asset_store = get_asset_store()
        for asset_id in session.asset_ids:
            asset_store.delete(asset_id)
    return {"deleted": store.delete(session_id)}


@router.get("/{session_id}/report", response_class=PlainTextResponse)
def get_report(
    session_id: str,
    query_id: str | None = Query(default=None, alias="queryId"),
) -> PlainTextResponse:
    """Markdown report with answer, metrics, evidence and the execution trace."""
    session = _require(session_id)
    if not session.results:
        raise HTTPException(status_code=404, detail="No results in this session yet.")

    if query_id:
        match = next((r for r in session.results if r.query_id == query_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Query {query_id} not found in this session.")
        body = result_to_markdown(match, heading_level=1)
        filename = f"satquery-{query_id}.md"
    else:
        body = session_to_markdown(session.title, session.results)
        filename = f"satquery-session-{session.id}.md"

    return PlainTextResponse(
        body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
