FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    WHISPER_MODEL=tiny \
    WHISPER_COMPUTE_TYPE=int8 \
    PORT=8501

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install the Piper CLI used by src/services/tts_piper.py.
RUN wget -q -O /tmp/piper_amd64.tar.gz \
    https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    && tar -xzf /tmp/piper_amd64.tar.gz -C /usr/local/bin --strip-components=1 \
    && rm /tmp/piper_amd64.tar.gz

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p models chroma_db resumes && \
    wget -q -O models/en_US-lessac-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx && \
    wget -q -O models/en_US-lessac-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# Download the Chroma embedding model during the image build, not at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Download the Whisper model during the image build, not on first page load.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"

# Build the fixed demo knowledge base into the image so startup is immediate.
RUN python scripts/ingest_resumes.py --clear

EXPOSE 8501

CMD ["sh", "-c", "streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]