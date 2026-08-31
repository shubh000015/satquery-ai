from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden from backend/.env."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_prefix="SATQUERY_",
        extra="ignore",
    )

    app_name: str = "SatQuery AI"
    problem_statement: str = "SIH26167"
    version: str = "0.1.0"

    data_dir: Path = BACKEND_ROOT / ".data"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Rasters are decimated to this longest edge before analysis. Keeps the
    # heuristic baseline responsive on full Cartosat/RISAT tiles.
    analysis_max_edge: int = 512
    max_upload_bytes: int = 200 * 1024 * 1024
    max_assets_per_query: int = 2

    # SIH26167 allows PNG/JPEG only for the prescribed benchmark datasets. Strict
    # mode rejects them outright; the default downgrades it to a warning so the
    # team can still demo with JPEG chips.
    strict_format_policy: bool = False

    # Fine-tuned weights are not shipped yet. Point these at local checkpoints or
    # an inference server and the matching tool switches off the heuristic path.
    vlm_endpoint: str | None = None
    grounding_endpoint: str | None = None
    change_endpoint: str | None = None
    fusion_endpoint: str | None = None

    @property
    def asset_dir(self) -> Path:
        return self.data_dir / "assets"

    @property
    def session_dir(self) -> Path:
        return self.data_dir / "sessions"

    def ensure_dirs(self) -> None:
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
