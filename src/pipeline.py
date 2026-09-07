"""Voice agent component builder."""
import os
import time
from datetime import datetime
from pathlib import Path

from src.services.llm_groq import GroqLLM
from src.services.llm_tokenrouter import TokenRouterLLM


def _log(message: str):
    print(f"[STARTUP] {message}", flush=True)


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
    started_at = time.perf_counter()
    _log(
        f"build_pipeline start: rag={enable_rag}, tts={enable_tts}, "
        f"asr={enable_asr}, llm={llm_mode}"
    )

    # ASR
    asr = None
    if enable_asr:
        asr_provider = os.getenv("ASR_PROVIDER", "local").lower()
        _log(f"ASR loading start: provider={asr_provider}")
        if asr_provider == "groq":
            from src.services.asr_groq import GroqWhisperASR

            asr = GroqWhisperASR()
        else:
            from src.services.asr_whisper import LaptopWhisperASR

            model_size = os.getenv("WHISPER_MODEL", "tiny")
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            asr = LaptopWhisperASR(model_size=model_size, compute_type=compute_type)
        _log("ASR loading complete")

    # RAG
    project_root = Path(__file__).resolve().parent.parent
    chroma_dir = os.getenv("CHROMA_DIR", str(project_root / "chroma_db"))
    _log(f"RAG loading start: path={chroma_dir}") if enable_rag else _log("RAG disabled")
    rag = None
    if enable_rag:
        from src.services.rag_chroma import ChromaRAG

        rag = ChromaRAG(collection_name="voice_kb", persist_dir=chroma_dir, top_k=3)
    if rag:
        _log("RAG loading complete")

    # LLM
    _log(f"LLM loading start: mode={llm_mode}")
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
    _log("LLM loading complete")

    # TTS
    tts = None
    if enable_tts:
        tts_provider = os.getenv("TTS_PROVIDER", "none").lower()
        _log(f"TTS loading start: provider={tts_provider}")
        if tts_provider == "edge":
            from src.services.tts_edge import EdgeTTS

            tts = EdgeTTS(voice=os.getenv("TTS_VOICE", "en-US-GuyNeural"))
            _log("TTS loading complete")
        else:
            _log("TTS disabled: set TTS_PROVIDER=edge to enable network TTS")
    else:
        _log("TTS disabled")

    _log(f"build_pipeline complete in {time.perf_counter() - started_at:.2f}s")

    return None, asr, rag, llm, tts
