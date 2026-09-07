"""Low-memory network TTS through Microsoft Edge's public speech endpoint."""
import asyncio
import os
import tempfile

import edge_tts


class EdgeTTS:
    """Generate MP3 audio without bundling a local voice model."""

    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice
        self.audio_format = "audio/mpeg"
        print(f"[TTS] Edge TTS ready: {voice}", flush=True)

    async def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            return b""

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
                temp_path = audio_file.name
            communicator = edge_tts.Communicate(text, self.voice)
            await communicator.save(temp_path)
            with open(temp_path, "rb") as audio_file:
                return audio_file.read()
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
