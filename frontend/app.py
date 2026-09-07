import asyncio
import io
import logging
import os
import queue
import time
import wave

import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import build_pipeline
from src.services.rag_chroma import ChromaRAG


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


def collect_webrtc_audio(webrtc_context):
    """Drain browser audio frames into the current Streamlit session."""
    if not webrtc_context or not webrtc_context.state.playing:
        return

    try:
        frames = webrtc_context.audio_receiver.get_frames(timeout=0.2)
    except queue.Empty:
        return

    for frame in frames:
        audio = frame.to_ndarray()
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        st.session_state.webrtc_frames.append(
            (audio.astype(np.float32), frame.sample_rate or 48000)
        )


def webrtc_frames_to_wav():
    """Convert collected WebRTC frames into the WAV format ASR expects."""
    frames = st.session_state.get("webrtc_frames", [])
    if not frames:
        return b""

    sample_rate = frames[0][1]
    audio = np.concatenate([audio for audio, _ in frames])
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm_to_wav(pcm.tobytes(), sample_rate)


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

    return response


# ============================================================
# LOAD BACKEND
# ============================================================

@st.cache_resource(show_spinner="Loading AI models...")
def load_backend():
    logger.info("load_backend start")
    llm_mode = "tokenrouter"
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
    """Load the embedding model only when a question needs RAG."""
    logger.info("load_rag start")
    try:
        chroma_dir = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
        rag = ChromaRAG(collection_name="voice_kb", persist_dir=chroma_dir, top_k=3)
        logger.info("load_rag complete")
        return rag
    except Exception:
        logger.exception("load_rag failed")
        raise


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

if "webrtc_frames" not in st.session_state:
    st.session_state.webrtc_frames = []


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

    st.write("**LLM:** TokenRouter")
    st.write("**Model:** z-ai/glm-5.3-free")
    st.write("**ASR:** faster-whisper tiny (on voice use)")
    st.write("**TTS:** Piper")
    st.write("**Vector DB:** ChromaDB")

    st.divider()
    st.subheader("System Status")

    tokenrouter_key = os.getenv("TOKENROUTER_API_KEY")
    if tokenrouter_key:
        st.markdown(
            '<span class="status-green">🟢 TokenRouter API key detected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-red">🔴 TokenRouter API key missing</span>',
            unsafe_allow_html=True,
        )
        st.caption("Set TOKENROUTER_API_KEY before starting Streamlit.")

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

        st.write("📚 Knowledge base: loads on first question")

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
            st.caption("Download a Piper voice model to enable TTS.")

    except Exception as exc:
        backend = None
        st.markdown(
            '<span class="status-red">🔴 Backend initialization failed</span>',
            unsafe_allow_html=True,
        )
        st.error(str(exc))

    st.divider()

    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_audio = None
        st.session_state.last_latency = None
        st.rerun()


# ============================================================
# BACKEND CHECK
# ============================================================

if backend is None:
    st.error("The AI backend could not be initialized.")
    st.info("Check your TokenRouter API key, Piper installation, and model files.")
    st.stop()

rag = backend["rag"]
llm = backend["llm"]
tts = backend["tts"]


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
            st.audio(message["audio"], format="audio/wav")
        if message.get("latency"):
            st.caption(f"Response time: {message['latency']:.2f}s")


# ============================================================
# VOICE INPUT
# ============================================================

st.subheader("🎤 Voice Input")

webrtc_ctx = webrtc_streamer(
    key="voice-input",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"audio": True, "video": False},
    audio_receiver_size=256,
)
collect_webrtc_audio(webrtc_ctx)

webrtc_audio = None
if not webrtc_ctx.state.playing and st.session_state.webrtc_frames:
    if st.button("Process browser recording", type="primary"):
        webrtc_audio = webrtc_frames_to_wav()
        st.session_state.webrtc_frames = []

if webrtc_audio:
    audio_input = None
    audio_bytes = webrtc_audio
else:
    audio_input = st.audio_input("Or record with Streamlit")

if webrtc_audio or audio_input is not None:
    audio_bytes = webrtc_audio or audio_input.getvalue()
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
            st.session_state.messages.append({"role": "user", "content": text})

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
                audio_wav = None
                if tts:
                    with st.spinner("🔊 Generating voice..."):
                        try:
                            pcm = run_async(tts.synthesize(response))
                            if pcm:
                                audio_wav = pcm_to_wav(pcm, tts.sample_rate)
                        except Exception as exc:
                            st.warning(f"TTS failed: {exc}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "audio": audio_wav,
                    "latency": latency,
                })
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
        st.session_state.messages.append({"role": "user", "content": text_input})

        start = time.perf_counter()
        with st.spinner("🤖 Thinking..."):
            try:
                response = ask_llm(text_input, get_rag(), llm)
            except Exception as exc:
                st.error(f"LLM error: {exc}")
                response = None

        latency = time.perf_counter() - start

        if response:
            audio_wav = None
            if tts:
                with st.spinner("🔊 Generating voice..."):
                    try:
                        pcm = run_async(tts.synthesize(response))
                        if pcm:
                            audio_wav = pcm_to_wav(pcm, tts.sample_rate)
                    except Exception as exc:
                        st.warning(f"TTS failed: {exc}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "audio": audio_wav,
                "latency": latency,
            })
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
