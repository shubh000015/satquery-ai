from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def clean_endpoint(value: object) -> str | None:
    """Strip quotes, inline comments and trailing slashes from an endpoint URL.

    The committed .env once had indented keys, localhost lines after the
    tunnel URL, and `# comments` on the same line — any of those silently
    pointed the backend at the wrong host or at nothing.
    """
    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    if not text or text.lower() in {"none", "null", "-"}:
        return None
    if "#" in text:
        text = text.split("#", 1)[0].strip()
    return text.rstrip("/") or None


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden from backend/.env."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_prefix="SATQUERY_",
        extra="ignore",
        env_ignore_empty=False,
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

    # One URL for every specialist (Kaggle + cloudflared). Per-task fields
    # override this when set.
    ml_endpoint: str | None = None
    vlm_endpoint: str | None = None
    grounding_endpoint: str | None = None
    change_endpoint: str | None = None
    fusion_endpoint: str | None = None

    @field_validator(
        "ml_endpoint",
        "vlm_endpoint",
        "grounding_endpoint",
        "change_endpoint",
        "fusion_endpoint",
        mode="before",
    )
    @classmethod
    def _clean_endpoints(cls, value: object) -> str | None:
        return clean_endpoint(value)

    def resolved_endpoint(self, name: str) -> str | None:
        """Per-task URL, or the shared `ml_endpoint` if that task was left blank."""
        specific = clean_endpoint(getattr(self, name, None))
        return specific or clean_endpoint(self.ml_endpoint)

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
