"""Lightweight ASR using faster-whisper.
Optimized for CPU-based voice agents.
"""
import asyncio
import io
import numpy as np
from faster_whisper import WhisperModel

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class LaptopWhisperASR(FrameProcessor):
    """
    CPU-friendly ASR using faster-whisper.

    Recommended:
        tiny -> fastest, lower accuracy
        base -> better accuracy, still reasonable for CPU laptops
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ):
        super().__init__()

        print(
            f"[ASR] Loading faster-whisper '{model_size}' "
            f"on {device} ({compute_type})..."
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        self.language = language
        self.sample_rate = 16000

        self._audio_buffer: list[np.ndarray] = []
        self._is_speaking = False

    # ============================================================
    # STANDALONE METHOD for web / console use (non-Pipecat)
    # ============================================================
    async def transcribe(self, audio_input) -> str:
        """
        Transcribe audio from various input formats:
        - bytes (raw PCM or WAV from browser)
        - np.ndarray (float32 audio)
        Returns transcribed text.
        """
        # Handle bytes input (from browser microphone)
        if isinstance(audio_input, (bytes, bytearray)):
            audio_np = self._bytes_to_numpy(audio_input)
        elif isinstance(audio_input, np.ndarray):
            audio_np = audio_input.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio_input)}")

        # Ensure mono
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)

        # Resample to 16kHz if needed (simple interpolation)
        # Assume common rates: 44100, 48000 from browser
        # We can't know exact rate from raw bytes, so assume 44100 for bytes
        # For numpy arrays, caller should ideally provide 16kHz

        return await self._transcribe(audio_np)

    def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw audio bytes to float32 numpy array."""
        # Try to detect WAV header
        if audio_bytes[:4] == b'RIFF':
            import wave
            with io.BytesIO(audio_bytes) as wav_io:
                with wave.open(wav_io, "rb") as wf:
                    n_frames = wf.getnframes()
                    data = wf.readframes(n_frames)
                    sample_width = wf.getsampwidth()
                    channels = wf.getnchannels()
                    rate = wf.getframerate()

                    if sample_width == 2:
                        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sample_width == 4:
                        arr = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

                    if channels > 1:
                        arr = arr.reshape(-1, channels)
                        arr = np.mean(arr, axis=1)

                    # Resample to 16kHz if needed
                    if rate != self.sample_rate:
                        arr = self._resample(arr, rate, self.sample_rate)
                    return arr
        else:
            # Assume raw PCM int16 at 16kHz mono
            arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return arr

    def _resample(self, audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        """Simple linear resampling."""
        if orig_rate == target_rate:
            return audio
        duration = len(audio) / orig_rate
        new_length = int(duration * target_rate)
        old_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        new_indices = np.linspace(0, len(audio) - 1, num=new_length)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    # ============================================================
    # PIPECAT METHODS
    # ============================================================
    async def process_frame(
        self,
        frame,
        direction: FrameDirection,
    ):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_speaking = True
            self._audio_buffer = []

        elif isinstance(frame, AudioRawFrame):
            if self._is_speaking:
                audio_np = np.frombuffer(
                    frame.audio,
                    dtype=np.int16,
                ).astype(np.float32) / 32768.0

                if frame.sample_rate != self.sample_rate:
                    audio_np = self._resample(audio_np, frame.sample_rate, self.sample_rate)

                self._audio_buffer.append(audio_np)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_speaking = False

            if self._audio_buffer:
                full_audio = np.concatenate(self._audio_buffer)
                text = await self._transcribe(full_audio)

                if text and text.strip():
                    await self.push_frame(TextFrame(text.strip()))

                self._audio_buffer = []

        await self.push_frame(frame, direction)

    async def _transcribe(self, audio_np: np.ndarray) -> str:
        loop = asyncio.get_event_loop()

        def transcribe():
            segments, info = self.model.transcribe(
                audio_np,
                language="en",
                task="transcribe",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
                initial_prompt=(
                    "This is a conversation with a voice assistant. "
                    "The user may ask questions about weather, "
                    "technology, companies, people, locations, "
                    "and general topics."
                ),
                temperature=0.0,
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            )

            return text, info

        text, info = await loop.run_in_executor(None, transcribe)

        print(
            f"[ASR] Transcribed "
            f"({info.language}, "
            f"{info.language_probability:.2f}): "
            f"{text}"
        )

        return text
