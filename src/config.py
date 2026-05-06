import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. Load your existing environment variables
load_dotenv()

def get_llm(temperature: float = 0.0):
    """
    Retrieves the LLM instance using your current .env configuration.
    """
    active_env = os.getenv("ACTIVE_ENV", "local").lower()

    # 1. Standard Cloud Path
    if active_env == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # 2. Corporate / Enterprise Proxy Path
    # Designed for OpenAI-compatible internal gateways (e.g., LiteLLM, vLLM proxies)
    elif active_env == "internal":
        return ChatOpenAI(
            model=os.getenv("INTERNAL_MODEL"),
            base_url=os.getenv("INTERNAL_BASE_URL"),
            api_key=os.getenv("INTERNAL_API_KEY"),
            temperature=temperature
        )
    
    # 3. Local Hardware Path (Default Fallback)
    # Compatible with Ollama, vLLM local server, or LM Studio
    else:
        return ChatOpenAI(
            model=os.getenv("LOCAL_MODEL", "local-model"),
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("LOCAL_API_KEY", "placeholder"),
            temperature=temperature
        )