from fastapi import APIRouter

from app.agent import registry
from app.core.config import get_settings
from app.schemas.api import HealthResponse
from app.schemas.registry import RegistryResponse
from app.services.raster import RASTERIO_AVAILABLE
from app.tools import tool_specs
from app.tools.endpoint_client import probe_ml

router = APIRouter(tags=["meta"])


def _configured_ml(settings) -> str | None:
    return (
        settings.ml_endpoint
        or settings.vlm_endpoint
        or settings.grounding_endpoint
        or settings.change_endpoint
        or settings.fusion_endpoint
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    ml = _configured_ml(settings)
    reachable = None if not ml else probe_ml(ml) is not None
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.version,
        problem_statement=settings.problem_statement,
        rasterio=RASTERIO_AVAILABLE,
        weights_wired=registry.weights_wired(settings),
        ml_endpoint=ml,
        ml_reachable=reachable,
    )


@router.get("/registry", response_model=RegistryResponse)
def get_registry() -> RegistryResponse:
    """The model plan and the specialist tools, with whether weights are wired."""
    settings = get_settings()
    specs = registry.model_specs(settings)
    return RegistryResponse(
        models=specs,
        tools=tool_specs(),
        weights_wired=registry.weights_wired(settings),
        total=len(specs),
    )
