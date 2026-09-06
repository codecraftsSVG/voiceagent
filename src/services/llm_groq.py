"""Groq API wrapper - standalone, no Pipecat deps."""
import os
from typing import List, Dict

from openai import AsyncOpenAI


class GroqLLM:
    """Groq provides free API access to Llama 3.1 70B/8B, Mixtral, etc."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "llama-3.1-70b-versatile",
        system_prompt: str = "You are a helpful voice assistant. Be concise.",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Get one at https://console.groq.com/keys")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: List[Dict[str, str]] = []
        print(f"[LLM] Groq ready: {model}")

    async def chat(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history[-12:]

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = resp.choices[0].message.content
            self.history.append({"role": "assistant", "content": content})
            print(f"[LLM] Response: {content[:120]}...")
            return content
        except Exception as e:
            print(f"[LLM] Groq error: {e}")
            return "I'm having trouble connecting right now. Please try again."
