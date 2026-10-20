import os
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv(find_dotenv(usecwd=True))


class GeminiChat(ChatGoogleGenerativeAI):
    """
    Wrapper around ChatGoogleGenerativeAI that normalizes response content
    from multipart lists (e.g. [{'type': 'text', 'text': '...'}]) into
    a standard string. This preserves full compatibility with downstream
    code expecting string content (such as `.strip()`, string formatting, etc.).
    """

    def _normalize_message(self, message):
        if hasattr(message, "content") and isinstance(message.content, list):
            text_parts = []
            for part in message.content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            message.content = "".join(text_parts)
        return message

    def invoke(self, *args, **kwargs):
        res = super().invoke(*args, **kwargs)
        return self._normalize_message(res)


def pick_llm(level: str = "medium", model_name: str | None = None, temperature: float = 0):
    """
    Picks the appropriate Google Gemini model based on task complexity or explicit model name.

    Gemini Model Tiers:
        - "low" / "lite": gemini-flash-lite-latest (fast, lightweight for classification, curation, and formatting)
        - "medium" / "flash": gemini-flash-lite-latest (balanced performance for SQL generation and tool calls)
        - "high" / "pro": gemini-3.8-flash (advanced reasoning and complex analytical tasks)

    Args:
        level (str): The complexity level of the task ("low", "medium", "high").
        model_name (str, optional): Explicit Gemini model name (e.g., "gemini-3.8-flash").
        temperature (float): Sampling temperature (default 0).

    Returns:
        GeminiChat: Configured Google Gemini model instance.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY=your_key to your .env file."
        )

    # If explicit model_name is provided, use it directly
    if model_name:
        selected_model = model_name
    else:
        lvl = level.lower().strip()
        if lvl in ["low", "lite", "flash-lite", "gemini-lite"]:
            selected_model = "gemini-flash-lite-latest"
        elif lvl in ["medium", "flash", "gemini-flash"]:
            selected_model = "gemini-flash-lite-latest"
        elif lvl in ["high", "pro", "gemini-pro"]:
            selected_model = "gemini-3.8-flash"
        elif lvl.startswith("gemini-"):
            selected_model = lvl
        else:
            selected_model = "gemini-flash-lite-latest"

    try:
        return GeminiChat(
            model=selected_model,
            api_key=api_key,
            temperature=temperature
        )
    except Exception as e:
        # Graceful fallback to reliable flash-lite if high tier hits quota
        if selected_model != "gemini-flash-lite-latest":
            print(f"[Warning] Failed to initialize {selected_model}: {e}. Falling back to gemini-flash-lite-latest.")
            return GeminiChat(
                model="gemini-flash-lite-latest",
                api_key=api_key,
                temperature=temperature
            )
        raise e


if __name__ == "__main__":
    llm_obj = pick_llm("low")
    res = llm_obj.invoke("What is the capital of France?")
    print("Response:", res.content)