"""Low-memory speech recognition through Groq's Whisper API."""
import asyncio
import os
import tempfile

from openai import OpenAI


class GroqWhisperASR:
    """Transcribe browser audio without loading a local Whisper model."""

    def __init__(self, model: str = "whisper-large-v3-turbo"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for Groq ASR")
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model
        print(f"[ASR] Groq Whisper ready: {model}", flush=True)

    async def transcribe(self, audio_input) -> str:
        if not audio_input:
            return ""
        if not isinstance(audio_input, (bytes, bytearray)):
            raise TypeError(f"Unsupported audio type: {type(audio_input)}")

        def request_transcription():
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                    audio_file.write(audio_input)
                    temp_path = audio_file.name
                with open(temp_path, "rb") as audio_file:
                    result = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language="en",
                        response_format="json",
                    )
                return result.text.strip()
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        return await asyncio.to_thread(request_transcription)
