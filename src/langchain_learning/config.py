"""Read model settings from environment variables and a local .env file."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_PROVIDERS = {"google_genai", "nvidia"}


@dataclass(frozen=True)
class ModelSettings:
    """Settings needed to initialize a chat model."""

    provider: str
    model: str


def get_model_settings() -> ModelSettings:
    """Return validated provider/model settings from the environment."""
    provider = os.getenv("LC_PROVIDER", "google_genai").strip().lower()
    model = os.getenv("LC_MODEL", "gemini-2.5-flash").strip()

    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Unsupported LC_PROVIDER {provider!r}. Choose one of: {supported}."
        )

    if not model:
        raise ValueError("LC_MODEL must not be empty.")

    return ModelSettings(provider=provider, model=model)
