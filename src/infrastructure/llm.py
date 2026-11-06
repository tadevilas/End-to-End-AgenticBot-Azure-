"""Creates the LangChain chat model that talks to Azure AI Foundry."""

from langchain_openai import ChatOpenAI

from src.core.config import Settings


def create_chat_model(settings: Settings, temperature: float = 0.2, max_tokens: int = 1024) -> ChatOpenAI:
    """Answer-writing model (Phi-4-mini)."""
    return _create(settings, settings.azure_ai_foundry_chat_model_name, temperature, max_tokens)


def create_planner_model(settings: Settings) -> ChatOpenAI:
    """Routing model (Llama 3.3 70B): deterministic and short, so it stays cheap."""
    return _create(settings, settings.azure_ai_foundry_planner_model_name, temperature=0, max_tokens=40)


def _create(settings: Settings, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.azure_ai_foundry_chat_endpoint,
        api_key=settings.azure_ai_foundry_chat_key.get_secret_value(),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
        timeout=30,  # the client default is 600s, which makes a stalled call look like a hang
    )
