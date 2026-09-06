#!/usr/bin/env python3
"""
Ingest Vamshidhar Goud's resumes into ChromaDB for voice-agent RAG.

Usage:
  python scripts/ingest_resumes.py

Place your resume files in ./resumes/ as .docx or .pdf.
Supported filenames (auto-detected):
  - *One_Page*Resume*.docx / .pdf  -> tagged as "one_page"
  - *Vamshidhar*Goud*.docx / .pdf -> tagged as "detailed"

If no files are found, uses embedded resume text.
"""
import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple
import os
import sys

# Adds the parent/root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now your import will work
# from src.your_module import your_function

from src.services.rag_chroma import ChromaRAG


# ============================================================
# FALLBACK RESUME TEXT (pre-extracted from uploaded files)
# ============================================================

DETAILED_RESUME_CHUNKS = [
    {
        "text": "Vamshidhar Goud is a Senior GenAI and Agentic AI Engineer based in Hyderabad, India. "
                "He has 6+ years of experience shipping production-grade RAG, GraphRAG, and multi-agent platforms "
                "at enterprise scale across GCP and Azure. Contact: +91 6303610573, vamshidhargoud300@gmail.com, "
                "linkedin.com/in/vamsdharngoud. Key metrics: 98% latency reduction, 94% retrieval accuracy, "
                "70% faster proposal generation, $2M+ in influenced initiatives.",
        "metadata": {"source": "detailed_resume", "section": "summary", "topic": "overview"}
    },
    {
        "text": "At Capgemini (April 2021 -- Present, Hyderabad), Vamshidhar works as Senior GenAI / Agentic AI Developer. "
                "He architected enterprise GenAI/Agentic AI platforms spanning 30,000+ documents and multimodal content. "
                "He optimized multivector hybrid retrieval and chunking to cut latency by 97%.",
        "metadata": {"source": "detailed_resume", "section": "experience", "company": "Capgemini", "topic": "overview"}
    },
    {
        "text": "Agent Builder Platform: Vamshidhar built an Agent Builder platform for autonomous agent composition and execution. "
                "This reduced agent-mapping latency from 120 seconds to 1--2 seconds, a 98% improvement. "
                "The platform enables dynamic creation and deployment of AI agents without manual configuration.",
        "metadata": {"source": "detailed_resume", "section": "experience", "company": "Capgemini", "topic": "agent_builder"}
    },
    {
        "text": "PO and Commercial Book Reconciliation Agent: Designed an autonomous Purchase Order and Commercial Book reconciliation agent "
                "for an enterprise CPG client. It automates extraction, validation, and discrepancy detection to cut manual effort by 60%+. "
                "Technologies: Python, LLMs, Agentic AI, Document Processing, Business Rules.",
        "metadata": {"source": "detailed_resume", "section": "experience", "company": "Capgemini", "topic": "reconciliation_agent"}
    },
    {
        "text": "Gemini + RAG RFP Automation: Led a Gemini + RAG RFP automation platform that reduced proposal generation time by 70%. "
                "Built a cloud-agnostic LLM layer for dynamic switching between Vertex AI and Azure OpenAI. "
                "This enables automatic generation of technical proposals from enterprise knowledge bases.",
        "metadata": {"source": "detailed_resume", "section": "experience", "company": "Capgemini", "topic": "rfp_automation"}
    },
    {
        "text": "AgentVista: Engineered AgentVista, a real-time agent observability platform using RabbitMQ, Loki, and Grafana. "
                "It covers agent throughput, ERP posting status, error rates, and SLA compliance monitoring. "
                "This provides full visibility into distributed agent workflows in production.",
        "metadata": {"source": "detailed_resume", "section": "experience", "company": "Capgemini", "topic": "agentvista"}
    },
    {
        "text": "GraphRAG and Agent Registry: Designed ontology-driven GraphRAG and agent-registry architecture using LlamaIndex, CrewAI, and n8n. "
                "This cut manual intervention by 60% across distributed workflows. "
                "The system uses knowledge graphs to improve retrieval and agent routing decisions.",
        "metadata": {"source": "detailed_resume", "section": "experience", "company": "Capgemini", "topic": "graphrag"}
    },
    {
        "text": "Search and Security Improvements: Improved search accuracy from 68% to 94% and cut user search time by 50%. "
                "Hardened production LLM applications with NeMo Guardrails against prompt injection and jailbreak attacks. "
                "Led production root-cause analysis across LLM, search, API, and agent layers.",
        "metadata": {"source": "detailed_resume", "section": "experience", "company": "Capgemini", "topic": "search_security"}
    },
    {
        "text": "Deep Algorithms (December 2019 -- April 2021): Worked as Associate Data Scientist in India. "
                "Built and deployed production ML and time-series forecasting models on AWS, improving prediction accuracy by 15% "
                "across 10,000+ daily predictions. Delivered ML/DL/RL proof-of-concepts for 8+ client initiatives representing $2M+ in potential revenue. "
                "Built multilingual Python/Django/Flask applications supporting 12+ languages.",
        "metadata": {"source": "detailed_resume", "section": "experience", "company": "Deep Algorithms", "topic": "overview"}
    },
    {
        "text": "Technical Skills -- GenAI and Agentic AI: RAG, GraphRAG, Multi-Agent Systems, Agent Orchestration, "
                "LlamaIndex, CrewAI, AutoGen, LangChain, LangGraph, n8n, NeMo Guardrails.",
        "metadata": {"source": "detailed_resume", "section": "skills", "topic": "genai"}
    },
    {
        "text": "Technical Skills -- LLMs and Platforms: GPT-4, Claude, Gemini, Vertex AI, Azure OpenAI, Model Routing, Embeddings.",
        "metadata": {"source": "detailed_resume", "section": "skills", "topic": "llms"}
    },
    {
        "text": "Technical Skills -- Cloud and Infrastructure: GCP, Azure, AWS, Docker, Kubernetes, CI/CD, Cloud Run, SageMaker.",
        "metadata": {"source": "detailed_resume", "section": "skills", "topic": "cloud"}
    },
    {
        "text": "Technical Skills -- Data and Search: Vector Databases, Graph Databases, MongoDB, SQL Server, Elasticsearch, Hybrid and Semantic Search.",
        "metadata": {"source": "detailed_resume", "section": "skills", "topic": "data"}
    },
    {
        "text": "Technical Skills -- Engineering: Python, SQL, FastAPI, REST APIs, Microservices, System Design, RabbitMQ, Loki, Grafana.",
        "metadata": {"source": "detailed_resume", "section": "skills", "topic": "engineering"}
    },
    {
        "text": "Project -- AI-Powered Insect Sound Recognition: Insect species identification from audio using transfer learning "
                "with EfficientNet and ResNet architectures. Improved accuracy from 78.5% to 94.7%.",
        "metadata": {"source": "detailed_resume", "section": "projects", "topic": "insect_recognition"}
    },
    {
        "text": "Project -- Medical AI Automated Clinical Trial Evaluation: Ensemble object-detection pipeline using Faster R-CNN and Cascade R-CNN "
                "automating clinical-trial evaluation. Cut analysis time by 90% with 95%+ confidence scores.",
        "metadata": {"source": "detailed_resume", "section": "projects", "topic": "medical_ai"}
    },
    {
        "text": "Education: PG Diploma in AI and Machine Learning from NIT Warangal, GPA 3.8/4.0 (2019--2020). "
                "B.Tech in Electronics and Communication Engineering from JNTU Hyderabad (2015--2019).",
        "metadata": {"source": "detailed_resume", "section": "education", "topic": "degrees"}
    },
    {
        "text": "Certifications and Awards: AWS Certified Machine Learning -- Specialty (Active). "
                "Microsoft Certified: Azure AI Fundamentals. Capgemini Aspiring Architect Award and Associate Innovator Award.",
        "metadata": {"source": "detailed_resume", "section": "education", "topic": "certifications"}
    },
]

ONE_PAGE_RESUME_CHUNKS = [
    {
        "text": "Vamshidhar Goud -- AI Lead | Principal AI Engineer | Senior GenAI and Agentic AI Engineer. "
                "Based in Hyderabad, India. Contact: +91 6303610573, vamshidhargoud300@gmail.com, linkedin.com/in/vamsdharngoud/. "
                "6+ years building production AI platforms, autonomous workflows, RAG/GraphRAG systems, and multi-cloud LLM solutions across GCP and Azure.",
        "metadata": {"source": "one_page_resume", "section": "summary", "topic": "overview"}
    },
    {
        "text": "Key Achievements Summary: 98% lower agent-mapping latency, 97% lower search latency, 94% retrieval accuracy, "
                "70% faster RFP generation, 60%+ lower manual effort, 85% lower manual data-processing effort.",
        "metadata": {"source": "one_page_resume", "section": "summary", "topic": "metrics"}
    },
    {
        "text": "Capgemini Experience (April 2021 -- Present): Architected enterprise GenAI and Agentic AI solutions integrating LLMs, RAG, "
                "multi-agent orchestration, workflow automation, and enterprise data. Built a private GenAI assistant for 30,000+ documents "
                "and multimodal content with 94% retrieval accuracy and 85% reduction in manual data processing.",
        "metadata": {"source": "one_page_resume", "section": "experience", "company": "Capgemini", "topic": "overview"}
    },
    {
        "text": "Enterprise Private GPT and Agentic AI Project: Python, Vertex AI, GPT-4, LlamaIndex, CrewAI, MongoDB, Cloud Run. "
                "30,000+ documents processed. 94% retrieval accuracy. 85% reduction in manual data processing. Secure enterprise knowledge discovery.",
        "metadata": {"source": "one_page_resume", "section": "projects", "company": "Capgemini", "topic": "private_gpt"}
    },
    {
        "text": "AgentVista and GraphRAG Project: Python, RabbitMQ, Loki, Grafana, Graph DB, Ontology, GPT-4/Claude, n8n. "
                "Real-time agent observability plus ontology-driven task hierarchy generation and dynamic agent routing.",
        "metadata": {"source": "one_page_resume", "section": "projects", "company": "Capgemini", "topic": "agentvista_graphrag"}
    },
    {
        "text": "Additional Skills from One-Page Resume: Claude Code, GitHub Copilot for AI-assisted development.",
        "metadata": {"source": "one_page_resume", "section": "skills", "topic": "ai_tools"}
    },
    {
        "text": "Academic Award: NIT Warangal Third Place -- Academic Excellence.",
        "metadata": {"source": "one_page_resume", "section": "education", "topic": "awards"}
    },
]


# ============================================================
# DOCX / PDF EXTRACTION
# ============================================================

def extract_docx_text(path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except ImportError:
        print("[WARN] python-docx not installed. Using fallback text.")
        return ""
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return ""


def extract_pdf_text(path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except ImportError:
        print("[WARN] PyPDF2 not installed. Using fallback text.")
        return ""
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return ""


def detect_resume_type(filename: str) -> str:
    """Detect if file is one_page or detailed resume."""
    lower = filename.lower()
    if "one_page" in lower or "onepage" in lower or "updated_one_page" in lower:
        return "one_page"
    return "detailed"


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Simple sliding-window chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_chunks_from_file(path: str) -> List[dict]:
    """Extract and chunk a resume file. Returns list of {text, metadata} dicts."""
    path = Path(path)
    resume_type = detect_resume_type(path.name)

    if path.suffix.lower() == ".docx":
        text = extract_docx_text(str(path))
    elif path.suffix.lower() == ".pdf":
        text = extract_pdf_text(str(path))
    else:
        text = ""

    if not text.strip():
        print(f"[WARN] No text extracted from {path.name}. Using fallback.")
        return []

    # Simple section detection
    sections = []
    lines = text.split("\n")
    current_section = "general"
    current_text = []

    section_keywords = {
        "summary": ["summary", "professional summary"],
        "experience": ["experience", "professional experience", "work experience"],
        "skills": ["skills", "technical skills"],
        "projects": ["projects", "selected projects"],
        "education": ["education", "certifications"],
    }

    for line in lines:
        stripped = line.strip().lower()
        detected = False
        for sec_name, keywords in section_keywords.items():
            if any(kw in stripped for kw in keywords) and len(stripped) < 60:
                if current_text:
                    sections.append((current_section, "\n".join(current_text)))
                current_section = sec_name
                current_text = []
                detected = True
                break
        if not detected:
            current_text.append(line)

    if current_text:
        sections.append((current_section, "\n".join(current_text)))

    # Create chunks with metadata
    result = []
    for sec_name, sec_text in sections:
        chunks = chunk_text(sec_text, chunk_size=600, overlap=80)
        for i, chunk in enumerate(chunks):
            result.append({
                "text": chunk,
                "metadata": {
                    "source": resume_type,
                    "section": sec_name,
                    "topic": f"{sec_name}_chunk_{i}",
                    "filename": path.name,
                }
            })

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ingest resumes into ChromaDB")
    parser.add_argument("--dir", default="./resumes", help="Directory containing resume files")
    parser.add_argument("--collection", default="voice_kb", help="ChromaDB collection name")
    parser.add_argument("--clear", action="store_true", help="Clear existing collection before ingest")
    args = parser.parse_args()

    print("=" * 60)
    print("  Resume Ingestion for Voice Agent RAG")
    print("=" * 60)

    # Initialize RAG
    rag = ChromaRAG(collection_name=args.collection, top_k=5)

    if args.clear:
        print("[INFO] Clearing existing collection...")
        try:
            rag.client.delete_collection(args.collection)
            rag = ChromaRAG(collection_name=args.collection, top_k=5)
        except Exception as e:
            print(f"[WARN] Could not clear collection: {e}")

    all_chunks = []

    # Try to read files from directory
    resume_dir = Path(args.dir)
    if resume_dir.exists():
        files = list(resume_dir.glob("*.docx")) + list(resume_dir.glob("*.pdf"))
        if files:
            print(f"[INFO] Found {len(files)} resume file(s) in {args.dir}")
            for file in files:
                print(f"  📄 Processing: {file.name}")
                chunks = extract_chunks_from_file(str(file))
                all_chunks.extend(chunks)
        else:
            print(f"[INFO] No .docx or .pdf files found in {args.dir}")

    # Fallback to embedded text if no files extracted
    if not all_chunks:
        print("[INFO] Using embedded resume text (fallback)...")
        all_chunks = DETAILED_RESUME_CHUNKS + ONE_PAGE_RESUME_CHUNKS

    # Deduplicate by text content
    seen = set()
    unique_chunks = []
    for chunk in all_chunks:
        text = chunk["text"].strip()
        if text and text not in seen:
            seen.add(text)
            unique_chunks.append(chunk)

    print(f"[INFO] Total unique chunks to ingest: {len(unique_chunks)}")

    # Prepare for ChromaDB
    texts = [c["text"] for c in unique_chunks]
    ids = [f"resume_{i:04d}" for i in range(len(unique_chunks))]
    metadatas = [c.get("metadata", {}) for c in unique_chunks]

    # Ingest
    rag.add_documents(texts=texts, ids=ids, metadatas=metadatas)

    # Verify
    count = rag.collection.count()
    print(f"[INFO] Collection '{args.collection}' now has {count} documents")
    print("\n✅ Resume ingestion complete!")
    print("\nTest questions you can ask:")
    print('  "How many years of experience does Vamshidhar have?"')
    print('  "Tell me about his Agent Builder project."')
    print('  "What was the latency improvement?"')
    print('  "What technologies does he know?"')
    print('  "Tell me about his GraphRAG experience."')


if __name__ == "__main__":
    main()
