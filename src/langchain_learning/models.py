"""Provider-agnostic LangChain chat model initialization."""

from langchain.chat_models import init_chat_model

from langchain_learning.config import get_model_settings


def get_chat_model():
    """Initialize the chat model selected by LC_PROVIDER and LC_MODEL.

    Required credentials are read from environment variables by the provider
    integration. For Gemini use GOOGLE_API_KEY; for NVIDIA use NVIDIA_API_KEY.
    """
    settings = get_model_settings()
    return init_chat_model(
        settings.model,
        model_provider=settings.provider,
        temperature=0,
    )
