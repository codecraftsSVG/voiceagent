---
title: Vamshidhar's AI Voice Assistant
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
---

# 🎙️ Vamshidhar's AI Voice Assistant

A real-time voice agent that answers questions about **Vamshidhar Goud's resume** using RAG (Retrieval-Augmented Generation).

Built for **8GB RAM laptops** — runs ASR and TTS locally, LLM via TokenRouter API.

```
Your Voice/Text ──► Whisper ASR ──► ChromaDB RAG ──► GLM-5.3 (TokenRouter) ──► Piper TTS ──► Voice Response
                                          │
                                    Resume Chunks
                              (experience, skills, projects)
```

---

## 📁 Project Structure

```
voice-agent-laptop/
├── src/
│   ├── pipeline.py                 # Component builder
│   └── services/
│       ├── asr_whisper.py          # faster-whisper base (CPU)
│       ├── llm_tokenrouter.py      # TokenRouter / GLM-5.3-free
│       ├── llm_groq.py             # Groq fallback
│       ├── tts_piper.py            # Piper TTS
│       └── rag_chroma.py           # ChromaDB vector store
├── frontend/
│   └── app.py                      # Streamlit web UI
├── scripts/
│   ├── ingest_resumes.py           # Ingest resume into ChromaDB
│   └── test_rag.py                 # Quick RAG sanity check
├── config/
│   └── prompts.yaml                # Resume-specific system prompt
├── resumes/                        # Drop your .docx / .pdf here
├── models/                         # Piper voice models
├── chroma_db/                      # Persistent vector DB (auto-created)
└── requirements.txt
```

---

## 🚀 Quick Start (Windows)

### 1. Install Dependencies

```cmd
cd voice-agent-laptop
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API Key

```cmd
set TOKENROUTER_API_KEY=sk-your-key-here
```

Or in PowerShell:
```powershell
$env:TOKENROUTER_API_KEY="sk-your-key-here"
```

### 3. Ingest Resume into ChromaDB

**Option A:** Drop your resume files into `./resumes/` (as `.docx` or `.pdf`), then:

```cmd
python scripts/ingest_resumes.py --dir ./resumes
```

**Option B:** If you don't have the files handy, the script auto-uses embedded resume text:

```cmd
python scripts/ingest_resumes.py
```

You should see:
```
[RAG] Indexed 25 documents
Collection 'voice_kb' now has 25 documents
```

### 4. Test RAG (Optional)

```cmd
python scripts/test_rag.py
```

This prints the top retrieved chunks for common resume questions.

### 5. Launch Streamlit Frontend

```cmd
streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501`

---

## 🎤 How to Use

### Voice Mode
1. Start the browser microphone recorder in the UI and grant microphone permission
2. Ask a question like:
   - *"How many years of experience do you have?"*
   - *"Tell me about the Agent Builder platform"*
   - *"What is your GraphRAG experience?"*
   - *"What technologies do you know?"*
3. The agent transcribes → searches resume → answers → speaks back

## Hugging Face Spaces Deployment

This repository includes a Docker image for a public Hugging Face Space. Create a new Space with the **Docker** SDK, upload or push this repository, and add `TOKENROUTER_API_KEY` under **Settings > Variables and secrets**.

The image installs `ffmpeg`, `espeak-ng`, and `libsndfile`, downloads the Piper voice model during the image build, and rebuilds the small embedded Chroma collection on every container start. Resume files can be added under `resumes/`; if none are present, the embedded resume chunks are used.

The browser microphone uses WebRTC. Visitors must grant microphone permission in their browser. The Space uses `WHISPER_MODEL=tiny` by default to fit the free CPU tier; set it to `base` only if the additional latency and memory use are acceptable.

Required Space secret:

```text
TOKENROUTER_API_KEY=your-tokenrouter-key
```

Optional Space variables:

```text
WHISPER_MODEL=tiny
WHISPER_COMPUTE_TYPE=int8
```

The free Space runtime is ephemeral, so local Chroma data is intentionally rebuilt rather than persisted between restarts.

## Render Deployment

The repository also includes `render.yaml` for Render Blueprint deployment. Push the repository to GitHub, then in Render choose **New > Blueprint**, select the repository, and apply the blueprint. Set `TOKENROUTER_API_KEY` when Render prompts for the secret.

Render's free web service has limited memory and may not reliably run Whisper, Chroma embeddings, and Streamlit together. Use a paid instance with at least 2 GB RAM for a dependable deployment. The service uses Whisper `tiny` by default and sleeps when idle on the free plan.

### Text Mode
1. Type in the chat box at the bottom
2. Same pipeline, just no microphone needed

---

## 🧪 Test Questions

| Question | Expected Answer Contains |
|----------|-------------------------|
| "How many years of experience?" | 6+ years |
| "Tell me about Agent Builder" | 120s → 1-2s, 98% latency reduction |
| "What was the latency improvement?" | 98% or 97% |
| "What technologies?" | Python, LlamaIndex, CrewAI, Vertex AI, Azure |
| "GraphRAG experience?" | ontology-driven, 60% manual intervention reduction |
| "Email address?" | vamshidhargoud300@gmail.com |
| "Current company?" | Capgemini |
| "Certifications?" | AWS Certified ML, Azure AI Fundamentals |

---

## ⚙️ Configuration

### Change LLM Provider

Edit `src/pipeline.py` or pass env vars:

```cmd
set LLM_MODE=groq
set GROQ_API_KEY=gsk-...
```

Supported: `tokenrouter` (default), `groq`, `ollama`

### Change ASR Model

Edit `src/pipeline.py`:
```python
asr = LaptopWhisperASR(model_size="tiny")   # faster, less accurate
asr = LaptopWhisperASR(model_size="small")  # slower, more accurate
```

### Add More Resume Files

1. Drop `.docx` or `.pdf` files into `./resumes/`
2. Re-run:
```cmd
python scripts/ingest_resumes.py --dir ./resumes --clear
```

The `--clear` flag wipes the old collection and re-ingests everything.

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `TOKENROUTER_API_KEY not set` | Run `set TOKENROUTER_API_KEY=sk-...` |
| `Piper not found` | Download from [Piper releases](https://github.com/rhasspy/piper/releases) and add to PATH |
| `No module named soundfile` | `pip install soundfile` |
| `ChromaDB collection empty` | Run `python scripts/ingest_resumes.py` first |
| `ASR returns empty text` | Speak louder / closer to mic, or switch to text input |
| `TTS no audio` | Piper model missing — download `en_US-lessac-medium.onnx` to `./models/` |

---

## 🔐 Security Note

Your `TOKENROUTER_API_KEY` should be kept private. Never commit it to git. The `.gitignore` already excludes `.env` files.

---

## 📜 License

MIT. Built with Pipecat, faster-whisper, ChromaDB, TokenRouter, and Piper.
