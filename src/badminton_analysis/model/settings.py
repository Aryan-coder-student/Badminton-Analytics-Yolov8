"""Central configuration for model backbones."""

from functools import lru_cache

# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Model configuration loaded from defaults, environment, or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BADMINTON_",
        extra="ignore",
    )

    clip_model_name: str = "openai/clip-vit-base-patch32"
    dinov2_model_name: str = "facebook/dinov2-base"
    resnet_weights_name: str = "DEFAULT"
    detection_model_name: str = "yolo11n.pt"
    detection_confidence: float = 0.25
    segmentation_model_name: str = "yolo11n-seg.pt"
    segmentation_confidence: float = 0.25
    segmentation_device: str | None = None
    tracking_config_name: str = "bytetrack.yaml"


@lru_cache
def get_settings() -> Settings:
    """Return the shared model settings instance."""
    return Settings()
