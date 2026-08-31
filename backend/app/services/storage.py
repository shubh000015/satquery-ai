"""On-disk asset store.

Uploads land in .data/assets as <asset_id><ext> with a JSON sidecar holding the
derived Asset record, plus a PNG preview because browsers cannot render GeoTIFF.
"""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.errors import UnsupportedFormatError
from app.schemas.imagery import (
    ACCEPTED_EXTENSIONS,
    BENCHMARK_ONLY_EXTENSIONS,
    Asset,
    AssetRole,
)
from app.services import modality as modality_service
from app.services.raster import Scene, load_scene, probe, write_preview

_SCENE_CACHE_SIZE = 8


class AssetStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self._scenes: OrderedDict[str, Scene] = OrderedDict()

    def path(self, asset_id: str) -> Path | None:
        for candidate in self.settings.asset_dir.glob(f"{asset_id}.*"):
            if candidate.suffix != ".json" and not candidate.name.endswith(".preview.png"):
                return candidate
        return None

    def preview_path(self, asset_id: str) -> Path:
        return self.settings.asset_dir / f"{asset_id}.preview.png"

    def _sidecar(self, asset_id: str) -> Path:
        return self.settings.asset_dir / f"{asset_id}.json"

    def save(
        self,
        filename: str,
        data: bytes,
        *,
        role: AssetRole = "primary",
        session_id: str | None = None,
        benchmark_dataset: str | None = None,
    ) -> Asset:
        suffix = Path(filename).suffix.lower()
        if suffix not in ACCEPTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"{filename}: '{suffix or 'no extension'}' is not an accepted format. "
                "Use GeoTIFF/TIFF, or PNG/JPEG for prescribed benchmark datasets."
            )
        if len(data) > self.settings.max_upload_bytes:
            raise UnsupportedFormatError(
                f"{filename} is {len(data) / 1e6:.1f} MB, over the "
                f"{self.settings.max_upload_bytes / 1e6:.0f} MB limit."
            )

        asset_id = uuid.uuid4().hex[:12]
        target = self.settings.asset_dir / f"{asset_id}{suffix}"
        target.write_bytes(data)

        try:
            meta = probe(target)
            scene = load_scene(target, max_edge=self.settings.analysis_max_edge)
        except Exception as exc:  # unreadable raster: do not keep a broken asset
            target.unlink(missing_ok=True)
            raise UnsupportedFormatError(f"{filename} could not be read as a raster: {exc}") from exc

        modality, source, _notes = modality_service.infer_modality(filename, scene)
        location, coords = modality_service.describe_position(meta)
        declared_format = "GeoTIFF" if suffix in {".tif", ".tiff"} and meta.georeferenced else suffix.lstrip(".").upper()

        asset = Asset(
            id=asset_id,
            role=role,
            modality=modality,
            name=filename,
            src=f"/api/assets/{asset_id}/preview",
            sensor=modality_service.guess_sensor(filename, modality),
            date=modality_service.guess_date(filename, target),
            gsd=modality_service.format_gsd(meta),
            location=location,
            coords=coords,
            format=declared_format,
            crs=meta.crs,
            session_id=session_id,
            size_bytes=len(data),
            benchmark_only_format=suffix in BENCHMARK_ONLY_EXTENSIONS,
            benchmark_dataset=benchmark_dataset,
            modality_source=source,
            meta=meta,
        )

        self._sidecar(asset_id).write_text(asset.model_dump_json(by_alias=True), encoding="utf-8")
        try:
            write_preview(target, self.preview_path(asset_id))
        except Exception:
            pass  # preview is a convenience, not a hard requirement

        self._cache_scene(asset_id, scene)
        return asset

    def get(self, asset_id: str) -> Asset | None:
        sidecar = self._sidecar(asset_id)
        if not sidecar.exists():
            return None
        try:
            return Asset.model_validate(json.loads(sidecar.read_text(encoding="utf-8")))
        except Exception:
            return None

    def require(self, asset_id: str) -> Asset:
        asset = self.get(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        return asset

    def update_role(self, asset_id: str, role: AssetRole) -> Asset:
        asset = self.require(asset_id)
        updated = asset.model_copy(update={"role": role})
        self._sidecar(asset_id).write_text(updated.model_dump_json(by_alias=True), encoding="utf-8")
        return updated

    def scene(self, asset_id: str) -> Scene:
        cached = self._scenes.get(asset_id)
        if cached is not None:
            self._scenes.move_to_end(asset_id)
            return cached
        path = self.path(asset_id)
        if path is None:
            raise KeyError(asset_id)
        scene = load_scene(path, max_edge=self.settings.analysis_max_edge)
        self._cache_scene(asset_id, scene)
        return scene

    def delete(self, asset_id: str) -> bool:
        removed = False
        for candidate in self.settings.asset_dir.glob(f"{asset_id}*"):
            candidate.unlink(missing_ok=True)
            removed = True
        self._scenes.pop(asset_id, None)
        return removed

    def _cache_scene(self, asset_id: str, scene: Scene) -> None:
        self._scenes[asset_id] = scene
        self._scenes.move_to_end(asset_id)
        while len(self._scenes) > _SCENE_CACHE_SIZE:
            self._scenes.popitem(last=False)


_store: AssetStore | None = None


def get_asset_store() -> AssetStore:
    global _store
    if _store is None:
        _store = AssetStore()
    return _store
