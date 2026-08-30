from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel

Modality = Literal["optical", "sar", "multispectral"]
AssetRole = Literal["primary", "secondary"]
InputMode = Literal["single", "cross-modal", "bi-temporal"]
Severity = Literal["error", "warning", "info"]

# SIH26167 input scope: GeoTIFF/TIFF for geospatial imagery, PNG/JPEG accepted
# only for the prescribed public benchmark datasets.
GEOSPATIAL_EXTENSIONS = {".tif", ".tiff"}
BENCHMARK_ONLY_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ACCEPTED_EXTENSIONS = GEOSPATIAL_EXTENSIONS | BENCHMARK_ONLY_EXTENSIONS


class RasterMeta(CamelModel):
    """What we could actually read off the file, as opposed to what was claimed."""

    width: int
    height: int
    bands: int
    dtype: str
    georeferenced: bool = False
    crs: str | None = None
    gsd_meters: float | None = None
    bounds: list[float] | None = None
    driver: str | None = None
    nodata: float | None = None


class Asset(CamelModel):
    """Shaped for the `Asset` type the workspace already renders."""

    id: str
    role: AssetRole
    modality: Modality
    name: str
    src: str
    sensor: str
    date: str
    gsd: str
    location: str
    coords: str
    format: str
    crs: str | None = None

    # Backend-only extras; harmless to the TypeScript consumer.
    session_id: str | None = None
    size_bytes: int = 0
    benchmark_only_format: bool = False
    benchmark_dataset: str | None = None
    modality_source: str = "inferred"
    meta: RasterMeta | None = None


class ValidationIssue(CamelModel):
    code: str
    severity: Severity
    message: str
    hint: str | None = None
    asset_id: str | None = None


class ValidationReport(CamelModel):
    """The `Input validator` box of the architecture diagram."""

    ok: bool
    mode: InputMode
    asset_count: int
    issues: list[ValidationIssue] = Field(default_factory=list)
    co_registered: bool | None = None
    overlap_percent: float | None = None
    summary: str = ""

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]
