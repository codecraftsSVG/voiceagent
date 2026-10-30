import asyncio
import hashlib
import io
import logging
import os
import time
import wave

import numpy as np
import streamlit as st

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import build_pipeline
from src.services.resume_context import ResumeContext
from src.services.semantic_cache import SemanticCache
from src.services.session_history import SessionHistory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [voice-agent] %(message)s",
)
logger = logging.getLogger("voice-agent.frontend")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vamshidhar's AI Voice Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }
    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .status-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }
    .status-green {
        color: #16803c;
        font-weight: 600;
    }
    .status-red {
        color: #c62828;
        font-weight: 600;
    }
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background: rgba(128, 128, 128, 0.08);
        text-align: center;
    }
    .resume-hint {
        background: rgba(22, 128, 60, 0.08);
        border-left: 4px solid #16803c;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def run_async(coro):
    """Execute an async function safely from Streamlit."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert raw PCM bytes returned by Piper into WAV bytes."""
    if not pcm_bytes:
        return b""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return buffer.getvalue()


def transcribe_audio(audio_bytes, whisper_model):
    """Convert browser microphone audio into text using Whisper."""
    if not audio_bytes:
        return ""

    try:
        import soundfile as sf

        audio_buffer = io.BytesIO(audio_bytes)
        audio, sample_rate = sf.read(audio_buffer, dtype="float32")

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Whisper accepts float32 audio.
        segments, _ = whisper_model.transcribe(
            audio,
            language="en",
            vad_filter=True,
        )

        text = " ".join(segment.text for segment in segments).strip()
        return text

    except Exception as exc:
        st.error(f"Speech recognition failed: {exc}")
        return ""


def ask_llm(user_text, rag, llm):
    """Execute RAG + LLM pipeline."""
    context_key = hashlib.sha256(
        f"{getattr(llm, 'model', 'unknown')}:{bool(rag)}".encode()
    ).hexdigest()
    st.session_state.last_cache_context_key = context_key
    cached = semantic_cache.lookup(user_text, context_key)
    st.session_state.last_cache_hit = cached
    if cached:
        return cached["response"]

    prompt = user_text

    # RAG
    if rag:
        try:
            contexts = rag.query(user_text)
            if contexts:
                prompt = rag.build_prompt(user_text, contexts)
        except Exception as exc:
            st.warning(f"RAG unavailable: {exc}")

    # LLM
    if not llm:
        raise RuntimeError("LLM is not initialized.")

    if hasattr(llm, "chat_stream"):
        response = run_async(llm.chat_stream(prompt))
    elif hasattr(llm, "chat"):
        response = run_async(llm.chat(prompt))
    else:
        raise RuntimeError("LLM does not expose chat/chat_stream.")

    semantic_cache.put(user_text, context_key, response)
    return response


# ============================================================
# LOAD BACKEND
# ============================================================

@st.cache_resource(show_spinner="Loading AI models...")
def load_backend():
    logger.info("load_backend start")
    llm_mode = os.getenv("LLM_MODE", "groq" if os.getenv("GROQ_API_KEY") else "tokenrouter")
    try:
        pipeline, asr, rag, llm, tts = build_pipeline(
            llm_mode=llm_mode,
            enable_rag=False,
            enable_tts=True,
            enable_asr=False,
        )
        logger.info("load_backend complete")
        return {
            "pipeline": pipeline,
            "asr": asr,
            "rag": rag,
            "llm": llm,
            "tts": tts,
        }
    except Exception:
        logger.exception("load_backend failed")
        raise


@st.cache_resource(show_spinner="Loading speech recognition...")
def load_asr():
    """Load one Whisper instance only when the user uses voice input."""
    logger.info("load_asr start")
    try:
        _, asr, _, _, _ = build_pipeline(
            llm_mode="tokenrouter",
            enable_rag=False,
            enable_tts=False,
            enable_asr=True,
        )
        logger.info("load_asr complete")
        return asr
    except Exception:
        logger.exception("load_asr failed")
        raise


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_rag():
    """Load the low-memory resume retriever when a question needs context."""
    logger.info("load_rag start")
    try:
        rag = ResumeContext(top_k=4)
        logger.info("load_rag complete")
        return rag
    except Exception:
        logger.exception("load_rag failed")
        raise


# ============================================================
# SESSION STATE
# ============================================================

history_store = SessionHistory()
semantic_cache = SemanticCache(str(history_store.path), threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92")))

if "active_session_id" not in st.session_state:
    sessions = history_store.list_sessions()
    if not sessions:
        sessions = [history_store.create()]
    st.session_state.active_session_id = sessions[0]["id"]
    st.session_state.messages = list(sessions[0].get("messages", []))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None


def persist_message(message):
    """Persist text metadata while retaining audio only for the current run."""
    history_store.append(st.session_state.active_session_id, message)


def reset_llm_context():
    current_llm = globals().get("llm")
    if hasattr(current_llm, "history"):
        current_llm.history = []


def sync_llm_context(messages):
    current_llm = globals().get("llm")
    if hasattr(current_llm, "history"):
        current_llm.history = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message.get("role") in {"user", "assistant"} and message.get("content")
        ]

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """<div class="app-title">🎙️ Vamshidhar's AI Voice Assistant</div>""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Ask about my resume, experience, skills, and projects — by voice or text"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="resume-hint">
        <b>💡 Try asking:</b><br>
        "How many years of experience do you have?" &nbsp;|&nbsp;
        "Tell me about the Agent Builder platform" &nbsp;|&nbsp;
        "What is your GraphRAG experience?" &nbsp;|&nbsp;
        "What technologies do you know?"
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    llm_mode_cfg = os.getenv("LLM_MODE", "groq" if os.getenv("GROQ_API_KEY") else "tokenrouter")
    if llm_mode_cfg == "groq":
        model_display = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        st.write("**LLM:** Groq / OpenRouter Proxy")
    else:
        model_display = os.getenv("TOKENROUTER_MODEL", "z-ai/glm-5.3-flash")
        st.write("**LLM:** TokenRouter")
    st.write(f"**Model:** {model_display}")
    st.write("**ASR:** Groq Whisper (on voice use)")
    st.write("**TTS:** Edge male network voice")
    st.write("**Vector DB:** Disabled on free tier")

    st.divider()
    st.subheader("System Status")

    api_key_detected = os.getenv("GROQ_API_KEY") if llm_mode_cfg == "groq" else os.getenv("TOKENROUTER_API_KEY")
    if api_key_detected:
        st.markdown(
            '<span class="status-green">🟢 LLM API key detected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-red">🔴 LLM API key missing</span>',
            unsafe_allow_html=True,
        )
        st.caption("Set GROQ_API_KEY or TOKENROUTER_API_KEY.")

    # Backend loading
    try:
        backend = load_backend()
        rag = backend["rag"]
        llm = backend["llm"]
        tts = backend["tts"]

        st.markdown(
            '<span class="status-green">🟢 Backend ready</span>',
            unsafe_allow_html=True,
        )

        st.write("📚 Resume context: lightweight mode")

        if tts:
            st.markdown(
                '<span class="status-green">🔊 Piper TTS ready</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-red">🔴 Piper unavailable</span>',
                unsafe_allow_html=True,
            )
            st.caption("Set TTS_PROVIDER=edge to enable network TTS.")

    except Exception as exc:
        backend = None
        st.markdown(
            '<span class="status-red">🔴 Backend initialization failed</span>',
            unsafe_allow_html=True,
        )
        st.error(str(exc))

    st.divider()

    st.subheader("Conversation history")
    sessions = history_store.list_sessions()
    session_by_id = {session["id"]: session for session in sessions}
    active_session = session_by_id.get(st.session_state.active_session_id)
    if active_session is None:
        active_session = history_store.create()
        st.session_state.active_session_id = active_session["id"]
        st.session_state.messages = []
        sessions = history_store.list_sessions()
        session_by_id = {session["id"]: session for session in sessions}

    session_ids = [session["id"] for session in sessions]
    selected_id = st.selectbox(
        "Open conversation",
        session_ids,
        index=session_ids.index(st.session_state.active_session_id),
        format_func=lambda session_id: session_by_id[session_id]["title"],
        label_visibility="collapsed",
    )
    if selected_id != st.session_state.active_session_id:
        st.session_state.active_session_id = selected_id
        st.session_state.messages = list(session_by_id[selected_id].get("messages", []))
        st.session_state.last_audio = None
        st.session_state.last_latency = None
        sync_llm_context(st.session_state.messages)
        st.rerun()

    if st.button("New conversation", use_container_width=True):
        new_session = history_store.create()
        st.session_state.active_session_id = new_session["id"]
        st.session_state.messages = []
        st.session_state.last_audio = None
        st.session_state.last_latency = None
        reset_llm_context()
        st.rerun()

    if st.button("Clear conversation", use_container_width=True):
        history_store.delete(st.session_state.active_session_id)
        new_session = history_store.create()
        st.session_state.active_session_id = new_session["id"]
        st.session_state.messages = []
        st.session_state.last_audio = None
        st.session_state.last_latency = None
        reset_llm_context()
        st.rerun()

# ============================================================
# BACKEND CHECK
# ============================================================

if backend is None:
    st.error("The AI backend could not be initialized.")
    st.info("Check your TokenRouter API key and deployment environment variables.")
    st.stop()

rag = backend["rag"]
llm = backend["llm"]
tts = backend["tts"]

if not st.session_state.get("llm_context_loaded"):
    sync_llm_context(st.session_state.messages)
    st.session_state.llm_context_loaded = True


def get_rag():
    """Load RAG on demand without blocking the initial page render."""
    if "rag_loaded" not in st.session_state:
        try:
            st.session_state.rag_loaded = load_rag()
        except Exception as exc:
            st.warning(f"Knowledge base unavailable: {exc}")
            st.session_state.rag_loaded = None
    return st.session_state.rag_loaded


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("audio"):
            st.audio(message["audio"], format=message.get("audio_format", "audio/wav"))
        if message.get("latency"):
            st.caption(f"Response time: {message['latency']:.2f}s")


# ============================================================
# VOICE INPUT
# ============================================================

    st.subheader("👨 Voice Input")
audio_input = st.audio_input("Record a question")

if audio_input is not None:
    audio_bytes = audio_input.getvalue()
    audio_hash = hash(audio_bytes)

    if st.session_state.get("processed_audio_hash") != audio_hash:
        st.session_state.processed_audio_hash = audio_hash

        with st.spinner("🎧 Understanding your voice..."):
            text = ""
            asr = backend.get("asr") or load_asr()

            try:
                if asr and hasattr(asr, "transcribe"):
                    text = run_async(asr.transcribe(audio_bytes))
                else:
                    st.error("ASR is not available.")

            except Exception as exc:
                st.error(f"Voice processing failed: {exc}")
                text = ""

        if text:
            st.info(f"**You said:** {text}")
            user_message = {"role": "user", "content": text}
            st.session_state.messages.append(user_message)
            persist_message(user_message)

            # LLM
            start = time.perf_counter()
            with st.spinner("🤖 Thinking..."):
                try:
                    response = ask_llm(text, get_rag(), llm)
                except Exception as exc:
                    st.error(f"LLM error: {exc}")
                    response = None

            latency = time.perf_counter() - start

            if response:
                st.session_state.last_latency = latency

                # TTS
                cached = st.session_state.get("last_cache_hit")
                audio_wav = cached.get("audio") if cached and cached.get("audio") else None
                audio_format = cached.get("audio_format") if cached and cached.get("audio_format") else None
                if tts and audio_wav is None:
                    with st.spinner("🔊 Generating voice..."):
                        try:
                                audio_data = run_async(tts.synthesize(response))
                                if audio_data:
                                    if getattr(tts, "audio_format", "").startswith("audio/"):
                                        audio_wav = audio_data
                                    else:
                                        audio_wav = pcm_to_wav(audio_data, tts.sample_rate)
                        except Exception as exc:
                            st.warning(f"TTS failed: {exc}")

                if audio_wav and not cached:
                    audio_format = getattr(tts, "audio_format", "audio/wav")
                    semantic_cache.update_audio(
                        text, st.session_state.last_cache_context_key, audio_wav, audio_format
                    )

                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "audio": audio_wav,
                    "audio_format": audio_format or (getattr(tts, "audio_format", "audio/wav") if audio_wav else None),
                    "latency": latency,
                }
                st.session_state.messages.append(assistant_message)
                persist_message(assistant_message)
                st.rerun()
        else:
            st.warning("I couldn't detect any speech. Please try again.")


# ============================================================
# TEXT INPUT
# ============================================================

st.divider()
st.subheader("💬 Text Input")

text_input = st.chat_input("Type your message...")

if text_input:
    text_input = text_input.strip()
    if text_input:
        user_message = {"role": "user", "content": text_input}
        st.session_state.messages.append(user_message)
        persist_message(user_message)

        start = time.perf_counter()
        with st.spinner("🤖 Thinking..."):
            try:
                response = ask_llm(text_input, get_rag(), llm)
            except Exception as exc:
                st.error(f"LLM error: {exc}")
                response = None

        latency = time.perf_counter() - start

        if response:
            cached = st.session_state.get("last_cache_hit")
            audio_wav = cached.get("audio") if cached and cached.get("audio") else None
            audio_format = cached.get("audio_format") if cached and cached.get("audio_format") else None
            if tts and audio_wav is None:
                with st.spinner("🔊 Generating voice..."):
                    try:
                        audio_data = run_async(tts.synthesize(response))
                        if audio_data:
                            if getattr(tts, "audio_format", "").startswith("audio/"):
                                audio_wav = audio_data
                            else:
                                audio_wav = pcm_to_wav(audio_data, tts.sample_rate)
                    except Exception as exc:
                        st.warning(f"TTS failed: {exc}")

            if audio_wav and not cached:
                audio_format = getattr(tts, "audio_format", "audio/wav")
                semantic_cache.update_audio(
                    text_input, st.session_state.last_cache_context_key, audio_wav, audio_format
                )

            assistant_message = {
                "role": "assistant",
                "content": response,
                "audio": audio_wav,
                "audio_format": audio_format or (getattr(tts, "audio_format", "audio/wav") if audio_wav else None),
                "latency": latency,
            }
            st.session_state.messages.append(assistant_message)
            persist_message(assistant_message)
            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Messages", len(st.session_state.messages))

with col2:
    if st.session_state.last_latency:
        st.metric("Last Response", f"{st.session_state.last_latency:.2f}s")
    else:
        st.metric("Last Response", "-")

with col3:
    st.metric("LLM", "TokenRouter")
