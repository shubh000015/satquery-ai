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
from app.core.errors import UnsupportedFormatError, UploadRejectedError
from app.schemas.imagery import (
    ACCEPTED_EXTENSIONS,
    BENCHMARK_ONLY_EXTENSIONS,
    Asset,
    AssetRole,
)
from app.services import modality as modality_service
from app.services.bands_ingest import Expect, parse_band_token, plan_scenes
from app.services.raster import Scene, load_scene, probe, stack_scenes, write_preview

_SCENE_CACHE_SIZE = 8


def _order_members(
    members: list[tuple[str, Path, Scene]], names: list[str]
) -> list[tuple[str, Path, Scene]]:
    by_name: dict[str, tuple[str, Path, Scene]] = {}
    unnamed: list[tuple[str, Path, Scene]] = []
    for item in members:
        token = parse_band_token(item[0])
        if token and token not in by_name:
            by_name[token] = item
        else:
            unnamed.append(item)
    ordered: list[tuple[str, Path, Scene]] = []
    used: set[str] = set()
    for band in names:
        if band in by_name:
            ordered.append(by_name[band])
            used.add(band)
        elif unnamed:
            ordered.append(unnamed.pop(0))
    ordered.extend(item for key, item in by_name.items() if key not in used)
    ordered.extend(unnamed)
    return ordered


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
            band_names=list(scene.band_names or []),
            source_files=[filename],
        )

        self._sidecar(asset_id).write_text(asset.model_dump_json(by_alias=True), encoding="utf-8")
        try:
            write_preview(target, self.preview_path(asset_id))
        except Exception:
            pass  # preview is a convenience, not a hard requirement

        self._cache_scene(asset_id, scene)
        return asset

    def save_uploads(
        self,
        files: list[tuple[str, bytes]],
        *,
        session_id: str | None = None,
        benchmark_dataset: str | None = None,
        expect: Expect = None,
    ) -> tuple[list[Asset], list[str]]:
        """Save one or more files, stacking band planes into at most two scenes."""
        if not files:
            raise UploadRejectedError("No files supplied.")
        if len(files) > self.settings.max_upload_files:
            raise UploadRejectedError(
                f"{len(files)} files supplied. A scene is 12 optical bands, 2 SAR bands, "
                f"or both (max {self.settings.max_upload_files} files)."
            )

        temps: list[Path] = []
        loaded: list[tuple[str, Path, Scene]] = []
        assets: list[Asset] = []
        try:
            for filename, data in files:
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
                tmp = self.settings.asset_dir / f"_ingest_{uuid.uuid4().hex[:10]}{suffix}"
                tmp.write_bytes(data)
                temps.append(tmp)
                try:
                    scene = load_scene(tmp, max_edge=self.settings.analysis_max_edge)
                except Exception as exc:
                    raise UnsupportedFormatError(
                        f"{filename} could not be read as a raster: {exc}"
                    ) from exc
                loaded.append((filename, tmp, scene))

            plan = plan_scenes(
                [name for name, _, _ in loaded],
                band_counts=[scene.bands for _, _, scene in loaded],
                expect=expect,
            )
            scene_count = sum(bool(scene.indices) for scene in plan.scenes)
            if scene_count > self.settings.max_assets_per_query:
                raise UploadRejectedError(
                    f"{scene_count} scenes supplied; the defined input scope is a single "
                    f"image or a pair (max {self.settings.max_assets_per_query}). "
                    "Optical band TIFFs of one date stack into one scene; SAR VV+VH stack into one scene."
                )

            for scene_plan in plan.scenes:
                bands = [parse_band_token(loaded[i][0]) for i in scene_plan.indices]
                duplicates = sorted({band for band in bands if band and bands.count(band) > 1})
                if duplicates:
                    raise UploadRejectedError(
                        f"Duplicate band(s) in {scene_plan.label}: {', '.join(duplicates)}."
                    )

            for scene_plan in plan.scenes:
                if not scene_plan.indices:
                    continue
                members = [loaded[i] for i in scene_plan.indices]
                role: AssetRole = "primary" if not assets else "secondary"
                if len(members) == 1 and not scene_plan.stacked:
                    filename, path, _scene = members[0]
                    asset = self.save(
                        filename,
                        path.read_bytes(),
                        role=role,
                        session_id=session_id,
                        benchmark_dataset=benchmark_dataset,
                    )
                    if scene_plan.names and not asset.band_names:
                        asset = asset.model_copy(update={"band_names": scene_plan.names})
                        self._sidecar(asset.id).write_text(
                            asset.model_dump_json(by_alias=True), encoding="utf-8"
                        )
                else:
                    ordered = _order_members(members, scene_plan.names)
                    asset = self._save_stack(scene_plan, ordered, role=role, session_id=session_id)
                assets.append(asset)
            return assets, plan.notes
        except Exception:
            for asset in assets:
                self.delete(asset.id)
            raise
        finally:
            for tmp in temps:
                tmp.unlink(missing_ok=True)

    def _save_stack(
        self,
        scene_plan,
        members: list[tuple[str, Path, Scene]],
        *,
        role: AssetRole,
        session_id: str | None,
    ) -> Asset:
        asset_id = uuid.uuid4().hex[:12]
        target = self.settings.asset_dir / f"{asset_id}.npz"
        stacked = stack_scenes([(name, scene) for name, _path, scene in members], scene_plan.names, target)
        first_name = members[0][0]
        modality, source, _notes = modality_service.infer_modality(
            first_name if scene_plan.kind != "sar" else "scene_sar_vv.tif",
            stacked,
        )
        if scene_plan.kind == "sar":
            modality, source = "sar", "band-stack"
        elif scene_plan.kind == "optical":
            modality, source = "multispectral" if len(scene_plan.names) >= 4 else "optical", "band-stack"
        location, coords = modality_service.describe_position(stacked.meta)
        asset = Asset(
            id=asset_id,
            role=role,
            modality=modality,
            name=scene_plan.label,
            src=f"/api/assets/{asset_id}/preview",
            sensor=modality_service.guess_sensor(first_name, modality),
            date=modality_service.guess_date(first_name, members[0][1]),
            gsd=modality_service.format_gsd(stacked.meta),
            location=location,
            coords=coords,
            format="Band stack",
            crs=stacked.meta.crs,
            session_id=session_id,
            size_bytes=target.stat().st_size,
            benchmark_only_format=False,
            modality_source=source,
            meta=stacked.meta,
            band_names=list(scene_plan.names),
            source_files=[name for name, _, _ in members],
        )
        self._sidecar(asset_id).write_text(asset.model_dump_json(by_alias=True), encoding="utf-8")
        try:
            write_preview(target, self.preview_path(asset_id))
        except Exception:
            pass
        self._cache_scene(asset_id, stacked)
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
