from pydantic import Field

from app.schemas.agent import QueryResult
from app.schemas.common import CamelModel
from app.schemas.imagery import Asset, InputMode, ValidationReport


class HealthResponse(CamelModel):
    status: str
    app: str
    version: str
    problem_statement: str
    rasterio: bool
    weights_wired: int


class UploadResponse(CamelModel):
    session_id: str
    assets: list[Asset]
    mode: InputMode
    validation: ValidationReport
    suggested: list[str] = Field(default_factory=list)


class QueryRequest(CamelModel):
    query: str
    asset_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    mode: InputMode | None = None


class ValidateRequest(CamelModel):
    asset_ids: list[str]
    mode: InputMode | None = None


class SessionSummary(CamelModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    mode: InputMode
    asset_count: int
    query_count: int


class SessionDetail(SessionSummary):
    assets: list[Asset] = Field(default_factory=list)
    results: list[QueryResult] = Field(default_factory=list)


class ErrorResponse(CamelModel):
    detail: str
    code: str = "error"
    issues: list[str] = Field(default_factory=list)
