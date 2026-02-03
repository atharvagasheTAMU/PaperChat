import os
from dotenv import load_dotenv

load_dotenv()


def _normalized_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def _require_ollama():
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except Exception as exc:
        raise ImportError(
            "Ollama support requires the 'langchain-ollama' package. "
            "Install it with: pip install langchain-ollama"
        ) from exc
    return ChatOllama, OllamaEmbeddings


def get_llm(temperature: float | None = None, tools=None):
    provider = _normalized_provider()
    llm_kwargs = {}
    if temperature is not None:
        llm_kwargs["temperature"] = temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        llm = ChatOpenAI(model=model, **llm_kwargs)
    elif provider == "ollama":
        ChatOllama, _ = _require_ollama()

        model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        llm = ChatOllama(model=model, **llm_kwargs)
    else:
        raise ValueError(
            "Unsupported LLM_PROVIDER. Use 'openai' or 'ollama'."
        )

    return llm.bind_tools(tools) if tools else llm


def get_embeddings():
    provider = _normalized_provider()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    if provider == "ollama":
        _, OllamaEmbeddings = _require_ollama()

        model = os.getenv("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model)

    raise ValueError("Unsupported LLM_PROVIDER. Use 'openai' or 'ollama'.")

