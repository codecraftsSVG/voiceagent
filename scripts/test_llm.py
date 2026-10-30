import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

env_path = Path("voice-agent.env")
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

import asyncio
from src.services.llm_groq import GroqLLM

async def main():
    print("Testing Groq LLM...")
    llm = GroqLLM(model="openai/gpt-oss-120b")
    response = await llm.chat("Hello! Reply with exactly: GROQ LOCAL TEST OK.")
    print(f"Response received: {response}")

if __name__ == "__main__":
    asyncio.run(main())
