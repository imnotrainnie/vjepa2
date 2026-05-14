from typing import Optional


def clean_text(text: Optional[str], fallback: str = "") -> str:
    if text is None:
        return fallback
    return " ".join(str(text).strip().split())
