#!/usr/bin/env python3
"""
Quick test script to verify resume RAG is working.
Run this after ingest_resumes.py.

Usage:
  python scripts/test_rag.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.rag_chroma import ChromaRAG


def main():
    print("=" * 60)
    print("  Resume RAG Test")
    print("=" * 60)

    rag = ChromaRAG(collection_name="voice_kb", top_k=3)
    count = rag.collection.count()
    print(f"\n📚 Collection has {count} documents\n")

    test_questions = [
        "How many years of experience does Vamshidhar have?",
        "Tell me about his Agent Builder project.",
        "What was the latency improvement?",
        "What technologies does he know?",
        "Tell me about his GraphRAG experience.",
        "What is his email address?",
        "Which company does he work for?",
        "What certifications does he have?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        contexts = rag.query(q)
        if contexts:
            print(f"  Retrieved {len(contexts)} chunks:")
            for i, ctx in enumerate(contexts, 1):
                preview = ctx[:150].replace("\n", " ")
                print(f"    [{i}] {preview}...")
        else:
            print("  ⚠️  No relevant chunks found!")
        print()

    print("✅ RAG test complete.")


if __name__ == "__main__":
    main()
