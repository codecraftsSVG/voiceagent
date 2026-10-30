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
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    models = await client.models.list()
    print("Available Groq models:")
    for m in models.data:
        print(" -", m.id)

if __name__ == "__main__":
    asyncio.run(main())
