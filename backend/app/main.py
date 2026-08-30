from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import assets, health, query, sessions
from app.core.config import get_settings
from app.core.errors import AgentError
from app.schemas.api import ErrorResponse

DESCRIPTION = """
Agentic remote-sensing backend for **SatQuery AI** (SIH26167).

Upload one scene or a pair, ask a question in plain language, and the controller
validates the inputs, routes the query to a task, selects specialist tools from
the registry, runs them, and returns a grounded answer with visual evidence,
confidence and an auditable execution trace.

Fine-tuned RS models are not wired yet: specialists currently run a deterministic
heuristic baseline and report that in `inferenceBackend`. Configure the endpoint
settings to swap in real weights without changing the API.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.app_name} API",
        version=settings.version,
        description=DESCRIPTION,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AgentError)
    async def agent_error_handler(_request: Request, exc: AgentError) -> JSONResponse:
        payload = ErrorResponse(detail=exc.detail, code=exc.code, issues=exc.issues)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(by_alias=True))

    api = APIRouter(prefix="/api")
    api.include_router(health.router)
    api.include_router(assets.router)
    api.include_router(query.router)
    api.include_router(sessions.router)
    app.include_router(api)

    @app.get("/", include_in_schema=False)
    def index() -> dict[str, str]:
        return {
            "app": settings.app_name,
            "problemStatement": settings.problem_statement,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
