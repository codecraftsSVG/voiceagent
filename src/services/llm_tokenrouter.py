"""TokenRouter API wrapper for z-ai/glm-5.3-free and other models.
Free tier available at https://api.tokenrouter.com
"""
import os
from typing import List, Dict

from openai import AsyncOpenAI


class TokenRouterLLM:
    """
    Uses TokenRouter's OpenAI-compatible API.
    Supports streaming for low-latency voice responses.

    Set env var: export TOKENROUTER_API_KEY="sk-..."
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.tokenrouter.com/v1",
        model: str = "z-ai/glm-5.3-free",
        system_prompt: str = "You are a helpful voice assistant. Keep responses under 2 sentences.",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        self.api_key = api_key or os.getenv("TOKENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TOKENROUTER_API_KEY not set. "
                "Get one at https://tokenrouter.com or set env var."
            )
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: List[Dict[str, str]] = []
        print(f"[LLM] TokenRouter ready: {model}")

    async def chat(self, user_text: str) -> str:
        """Non-streaming chat."""
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
            print(f"[LLM] TokenRouter error: {e}")
            return "I'm having trouble connecting right now. Please try again."

    async def chat_stream(self, user_text: str) -> str:
        """Streaming chat for lower perceived latency."""
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history[-12:]

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            content_parts = []
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content_parts.append(delta.content)

            full_content = "".join(content_parts)
            self.history.append({"role": "assistant", "content": full_content})
            print(f"[LLM] Response: {full_content[:120]}...")
            return full_content

        except Exception as e:
            print(f"[LLM] TokenRouter error: {e}")
            return "I'm having trouble connecting right now. Please try again."
