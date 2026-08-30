from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.agent import suggestions, validator
from app.core.config import get_settings
from app.core.errors import AgentError
from app.schemas.api import UploadResponse, ValidateRequest
from app.schemas.imagery import Asset, InputMode, ValidationReport
from app.services.sessions import get_session_store
from app.services.storage import get_asset_store

router = APIRouter(prefix="/assets", tags=["assets"])

_VALID_MODES = {"single", "cross-modal", "bi-temporal"}


@router.post("", response_model=UploadResponse)
async def upload_assets(
    files: Annotated[list[UploadFile], File(description="One scene, or a pair")],
    session_id: Annotated[str | None, Form()] = None,
    benchmark_dataset: Annotated[str | None, Form()] = None,
    mode: Annotated[str | None, Form()] = None,
) -> UploadResponse:
    """Accept 1–2 scenes, derive their metadata, and report compatibility."""
    settings = get_settings()
    store = get_asset_store()
    sessions = get_session_store()

    if not files:
        raise HTTPException(status_code=422, detail="No files supplied.")
    if len(files) > settings.max_assets_per_query:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{len(files)} files supplied. The defined input scope is a single image or a pair "
                f"(max {settings.max_assets_per_query})."
            ),
        )
    if mode is not None and mode not in _VALID_MODES:
        raise HTTPException(status_code=422, detail=f"Unknown mode '{mode}'.")

    session = sessions.get_or_create(session_id)
    stored: list[Asset] = []
    try:
        for index, upload in enumerate(files):
            payload = await upload.read()
            stored.append(
                store.save(
                    upload.filename or f"scene-{index}",
                    payload,
                    role="primary" if index == 0 else "secondary",
                    session_id=session.id,
                    benchmark_dataset=benchmark_dataset,
                )
            )
    except AgentError as exc:
        for asset in stored:  # do not leave half an upload behind
            store.delete(asset.id)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    mode_hint: InputMode | None = mode  # type: ignore[assignment]
    report = validator.validate(stored, mode_hint, settings)
    sessions.attach_assets(session.id, [a.id for a in stored], report.mode)

    return UploadResponse(
        session_id=session.id,
        assets=stored,
        mode=report.mode,
        validation=report,
        suggested=suggestions.suggest(report.mode, stored),
    )


@router.post("/validate", response_model=ValidationReport)
def validate_assets(payload: ValidateRequest) -> ValidationReport:
    """Dry-run compatibility check without spending time on specialists."""
    store = get_asset_store()
    assets: list[Asset] = []
    for asset_id in payload.asset_ids:
        asset = store.get(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found.")
        assets.append(asset)
    return validator.validate(assets, payload.mode, get_settings())


@router.get("/{asset_id}", response_model=Asset)
def get_asset(asset_id: str) -> Asset:
    asset = get_asset_store().get(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found.")
    return asset


@router.get("/{asset_id}/preview")
def get_preview(asset_id: str) -> FileResponse:
    """8-bit PNG rendering, because browsers cannot display GeoTIFF."""
    store = get_asset_store()
    preview = store.preview_path(asset_id)
    if not preview.exists():
        source = store.path(asset_id)
        if source is None:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found.")
        from app.services.raster import write_preview

        try:
            write_preview(source, preview)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Preview failed: {exc}") from exc
    return FileResponse(preview, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/{asset_id}/raw")
def get_raw(asset_id: str) -> FileResponse:
    """Original bytes as uploaded, for download or external tooling."""
    store = get_asset_store()
    path = store.path(asset_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found.")
    asset = store.get(asset_id)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=asset.name if asset else path.name,
    )


@router.delete("/{asset_id}")
def delete_asset(asset_id: str) -> dict[str, bool]:
    return {"deleted": get_asset_store().delete(asset_id)}
