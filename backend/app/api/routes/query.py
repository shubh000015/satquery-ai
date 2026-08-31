import json
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.controller import get_controller
from app.core.errors import AgentError
from app.schemas.agent import QueryResult
from app.schemas.api import QueryRequest

router = APIRouter(prefix="/query", tags=["query"])


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@router.post("", response_model=QueryResult)
def run_query(payload: QueryRequest) -> QueryResult:
    """Full pipeline in one response: validate, route, execute, integrate."""
    return get_controller().run(
        payload.query,
        payload.asset_ids,
        payload.session_id,
        payload.mode,
    )


@router.post("/stream")
def stream_query(payload: QueryRequest) -> StreamingResponse:
    """Same pipeline as an SSE stream so the UI can animate the trace live.

    Events: `trace` (skeleton), `step` (status change), `result`, `error`.
    """
    controller = get_controller()

    def generate() -> Iterator[str]:
        try:
            for event in controller.run_stream(
                payload.query,
                payload.asset_ids,
                payload.session_id,
                payload.mode,
            ):
                kind = event["type"]
                if kind == "result":
                    yield _sse("result", event["result"].model_dump_json(by_alias=True))
                elif kind == "trace":
                    yield _sse(
                        "trace",
                        json.dumps({"queryId": event["queryId"], "steps": event["steps"]}),
                    )
                else:
                    yield _sse("step", json.dumps(event["step"]))
        except AgentError as exc:
            yield _sse(
                "error",
                json.dumps({"detail": exc.detail, "code": exc.code, "issues": exc.issues}),
            )
        except Exception as exc:  # never leave the client hanging on an open stream
            yield _sse("error", json.dumps({"detail": str(exc), "code": "internal-error", "issues": []}))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
