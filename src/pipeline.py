"""Voice agent component builder."""
import os
from datetime import datetime
from pathlib import Path

from src.services.asr_whisper import LaptopWhisperASR
from src.services.tts_piper import PiperTTS
from src.services.rag_chroma import ChromaRAG
from src.services.llm_groq import GroqLLM
from src.services.llm_tokenrouter import TokenRouterLLM


def load_system_prompt() -> str:
    prompt = "You are a helpful, concise voice assistant. Keep answers under 2 sentences."
    try:
        import yaml
        project_root = Path(__file__).resolve().parent.parent
        with open(project_root / "config" / "prompts.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            prompt = cfg.get("system_prompt", prompt)
        prompt = prompt.format(date=datetime.now().isoformat(), context="")
    except Exception:
        pass
    return prompt


def build_pipeline(
    llm_mode: str = "tokenrouter",
    enable_rag: bool = True,
    enable_tts: bool = True,
    enable_asr: bool = True,
):
    """Build standalone components for console/audio modes."""

    # ASR
    asr = None
    if enable_asr:
        asr = LaptopWhisperASR(
            model_size=os.getenv("WHISPER_MODEL", "tiny"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        )

    # RAG
    project_root = Path(__file__).resolve().parent.parent
    chroma_dir = os.getenv("CHROMA_DIR", str(project_root / "chroma_db"))
    rag = (
        ChromaRAG(collection_name="voice_kb", persist_dir=chroma_dir, top_k=3)
        if enable_rag
        else None
    )

    # LLM
    system_prompt = load_system_prompt()

    if llm_mode == "tokenrouter":
        llm = TokenRouterLLM(
            model="z-ai/glm-5.3-free",
            system_prompt=system_prompt,
            max_tokens=256,
        )
    elif llm_mode == "groq":
        llm = GroqLLM(
            model="llama-3.1-70b-versatile",
            system_prompt=system_prompt,
            max_tokens=256,
        )
    else:
        llm = None

    # TTS
    tts = None
    if enable_tts:
        model_path = os.getenv(
            "PIPER_MODEL",
            str(project_root / "models" / "en_US-lessac-medium.onnx")
        )

        if os.path.exists(model_path):
            tts = PiperTTS(
                model_path=model_path,
                sample_rate=22050
            )
        else:
            print("[WARN] Piper model not found. TTS disabled. Run setup.sh")

    return None, asr, rag, llm, tts
