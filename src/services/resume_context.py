"""Low-memory keyword retrieval for the Render free-tier deployment."""
import re
from typing import List


RESUME_CHUNKS = [
    "Vamshidhar Goud is a Senior GenAI and Agentic AI Engineer based in Hyderabad, India. He has 6+ years of experience shipping production-grade RAG, GraphRAG, and multi-agent platforms at enterprise scale across GCP and Azure. Key metrics include 98% latency reduction, 94% retrieval accuracy, 70% faster proposal generation, and $2M+ in influenced initiatives. Contact: +91 6303610573, vamshidhargoud300@gmail.com, linkedin.com/in/vamsdharngoud/.",
    "At Capgemini from April 2021 to the present, Vamshidhar works as a Senior GenAI / Agentic AI Developer. He architected enterprise GenAI and Agentic AI platforms spanning 30,000+ documents and multimodal content, and optimized retrieval to cut latency by 97%.",
    "Agent Builder Platform: Vamshidhar built a platform for autonomous agent composition and execution. It reduced agent-mapping latency from 120 seconds to 1-2 seconds, a 98% improvement, and enables dynamic creation and deployment of AI agents without manual configuration.",
    "PO and Commercial Book Reconciliation Agent: He designed an autonomous agent for an enterprise CPG client that automates extraction, validation, and discrepancy detection, cutting manual effort by more than 60%. Technologies included Python, LLMs, Agentic AI, document processing, and business rules.",
    "Gemini plus RAG RFP Automation: He led a platform that reduced proposal generation time by 70%, with a cloud-agnostic LLM layer switching between Vertex AI and Azure OpenAI.",
    "AgentVista: He engineered a real-time agent observability platform using RabbitMQ, Loki, and Grafana to monitor agent throughput, ERP posting status, error rates, and SLA compliance.",
    "GraphRAG and Agent Registry: He designed an ontology-driven GraphRAG and agent-registry architecture using LlamaIndex, CrewAI, and n8n. It reduced manual intervention by 60% and used knowledge graphs to improve retrieval and agent routing.",
    "Search and security: He improved search accuracy from 68% to 94%, cut user search time by 50%, and hardened LLM applications with NeMo Guardrails against prompt injection and jailbreak attacks.",
    "At Deep Algorithms from December 2019 to April 2021, he worked as an Associate Data Scientist, deploying AWS machine-learning and time-series models that improved prediction accuracy by 15% across 10,000+ daily predictions.",
    "Technical skills include RAG, GraphRAG, multi-agent systems, agent orchestration, LlamaIndex, CrewAI, AutoGen, LangChain, LangGraph, n8n, NeMo Guardrails, GPT-4, Claude, Gemini, Vertex AI, Azure OpenAI, GCP, Azure, AWS, Docker, Kubernetes, Python, SQL, FastAPI, RabbitMQ, Loki, and Grafana.",
    "Projects include AI-powered insect sound recognition using EfficientNet and ResNet, improving accuracy from 78.5% to 94.7%, and automated clinical-trial evaluation using Faster R-CNN and Cascade R-CNN, cutting analysis time by 90% with 95%+ confidence.",
    "Education: PG Diploma in AI and Machine Learning from NIT Warangal, GPA 3.8/4.0, and B.Tech in Electronics and Communication Engineering from JNTU Hyderabad. Certifications include AWS Certified Machine Learning Specialty and Microsoft Azure AI Fundamentals.",
]


class ResumeContext:
    """Tiny keyword retriever that replaces ChromaDB on memory-constrained hosts."""

    def __init__(self, chunks: List[str] = None, top_k: int = 4):
        self.chunks = chunks or RESUME_CHUNKS
        self.top_k = top_k

    def query(self, query_text: str) -> List[str]:
        query_terms = set(re.findall(r"[a-z0-9]+", query_text.lower()))
        scored = []
        for chunk in self.chunks:
            chunk_terms = set(re.findall(r"[a-z0-9]+", chunk.lower()))
            score = len(query_terms & chunk_terms)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        matches = [chunk for score, chunk in scored if score > 0]
        return matches[: self.top_k] or self.chunks[: self.top_k]

    def build_prompt(self, query: str, contexts: List[str]) -> str:
        context_block = "\n".join(f"- {context}" for context in contexts)
        return f"Resume context:\n{context_block}\n\nUser question: {query}\nAnswer in first person, using only the resume context."
