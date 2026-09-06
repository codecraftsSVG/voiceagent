
"""Ultra-lightweight TTS using Piper - standalone, no Pipecat deps."""

import asyncio
import os
import tempfile
import wave
import numpy as np


class PiperTTS:
    """
    Piper TTS wrapper for Windows.

    Piper generates a temporary WAV file, which is then
    converted to raw PCM bytes for sounddevice playback.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        speaker_id: int = None,
        sample_rate: int = 22050,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.speaker_id = speaker_id
        self.sample_rate = sample_rate

        print(f"[TTS] Piper ready: {model_path}")

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech.

        Returns raw PCM int16 bytes.
        """

        if not text or not text.strip():
            return b""

        # Create a temporary WAV filename.
        # Do NOT keep the temporary file open while Piper writes to it.
        fd, wav_path = tempfile.mkstemp(
            suffix=".wav",
            prefix="piper_",
        )

        os.close(fd)

        cmd = [
            "piper",
            "-m",
            self.model_path,
            "--output_file",
            wav_path,
        ]

        if self.config_path:
            cmd.extend([
                "-c",
                self.config_path,
            ])

        if self.speaker_id is not None:
            cmd.extend([
                "--speaker",
                str(self.speaker_id),
            ])

        try:

            # Run Piper
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate(
                text.encode("utf-8")
            )

            # Check Piper result
            if proc.returncode != 0:

                error_message = stderr.decode(
                    "utf-8",
                    errors="replace",
                )

                print(
                    f"[TTS] Piper error:\n"
                    f"{error_message}"
                )

                return b""

            # Make sure WAV was actually created
            if not os.path.exists(wav_path):
                print(
                    "[TTS] Piper did not create WAV file."
                )
                return b""

            # Read WAV file
            with wave.open(wav_path, "rb") as wf:

                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()

                data = wf.readframes(n_frames)

            # Update actual sample rate from Piper
            self.sample_rate = sample_rate

            # Piper normally produces 16-bit PCM
            if sample_width != 2:

                print(
                    f"[TTS] Unsupported sample width: "
                    f"{sample_width}"
                )

                return b""

            # Convert stereo -> mono
            if n_channels == 2:

                arr = np.frombuffer(
                    data,
                    dtype=np.int16,
                ).reshape(-1, 2)

                mono = arr.mean(
                    axis=1
                ).astype(np.int16)

                return mono.tobytes()

            # Already mono
            return data

        except FileNotFoundError:

            print(
                "[TTS] ERROR: 'piper' command not found."
            )

            print(
                "Make sure Piper is installed inside "
                "the current virtual environment."
            )

            return b""

        except Exception as e:

            print(
                f"[TTS] Synthesis error: {e}"
            )

            return b""

        finally:

            # Always delete temporary WAV
            try:

                if os.path.exists(wav_path):
                    os.remove(wav_path)

            except Exception:
                pass

