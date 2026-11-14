from pathlib import Path
import streamlit.components.v1 as components

_voice = components.declare_component(
    "browser_voice_assistant",
    path=str(Path(__file__).resolve().parent.parent / "voice_component"),
)

def browser_voice(answer: str = "", key: str | None = None):
    return _voice(answer=answer, key=key, default=None)
