# Capstone Project: SecureGuard Banking AI
**A Privacy-First, Hybrid AI Architecture for Financial Services**

## 1. Executive Summary
Financial institutions face a critical dilemma: they need the efficiency of Generative AI, but cannot risk using public models (like ChatGPT) due to **data privacy laws**, **hallucinations**, and **lack of domain expertise**.

**SecureGuard** is a purpose-built, self-hosted AI platform designed to bridge this gap. It implements a **Hybrid Architecture** combining Retrieval-Augmented Generation (RAG) for accuracy with Parameter-Efficient Fine-Tuning (PEFT/LoRA) for style and security.

---

## 2. Problem Statement
1.  **Privacy Risks:** Sending unencrypted customer data (PII) to external APIs (OpenAI/Anthropic) violates GDPR/CCPA.
2.  **Hallucinations:** General-purpose LLMs invent fake interest rates or policies when asked specific banking questions.
3.  **Accuracy vs. Cost:** Training a massive model from scratch is prohibitively expensive; using small models often leads to poor grammar and reasoning.
4.  **Regulatory Compliance:** Banks need audit trails, explainability, and role-based access control (RBAC), which standard chatbots lack.

---

## 3. The Solution: SecureGuard Architecture

This project delivers a vertically integrated AI stack solution:

### A. The "Brain" (Hybrid Intelligence)
*   **Base Model:** TinyLlama 1.1B (Lightweight, deployable on edge/consumer hardware).
*   **Layer 1: RAG (Memory):** Uses **Pinecone Vector Database** to retrieve real-time, accurate bank policies (e.g., "Interest rate is 4.5% today") before the model speaks.
    *   *Solves: Hallucinations & Stale Data.*
*   **Layer 2: Fine-Tuning (Style):** Uses **QLoRA** (Quantized Low-Rank Adaptation) to train the model on specific banking dialogue (Bitext dataset) without retraining the whole brain.
    *   *Solves: Tone, Vocabulary, and Compliance phrasing.*

### B. The "Shield" (Security & PII)
*   **Zero-Trust PII Masking:** An intermediate layer that regex-scans and redacts Credit Card numbers, SSNs, and Names *before* they touch the AI model or database.
*   **Role-Based Access (RBAC):** 'Tellers' access general info; 'Managers' access sensitive transaction history.
*   **Audit Logging:** Every query and response is cryptographically logged for compliance reviews.

---

## 4. Technical Implementation (Proof of Work)

| Component | Technology | Status |
| :--- | :--- | :--- |
| **Backend** | Python, FastAPI, AsyncIO | ✅ Completed |
| **LLM Engine** | PyTorch, HuggingFace, Ollama | ✅ Completed |
| **Vector DB** | Pinecone (Serverless) | ✅ Verified |
| **Training** | Custom PyTorch Loop (Colab/Local) | ✅ Integrated |
| **Security** | PII Redaction Middleware | ✅ Active |
| **Frontend** | HTML5/JS (Dashboard) | ✅ Functional |

---

## 5. Key Differentiators (Why this wins)
1.  **Fully Local capable:** Can run entirely offline (Air-gapped) for maximum security.
2.  **Cost Efficiency:** Uses a 1.1B parameter model (runs on CPU) but achieves high accuracy via RAG.
3.  **Modular Brain:** The "Strategy Pattern" allows hot-swapping models (Ollama -> Custom -> GPT-4) instantly based on server load or difficulty.

## 6. Research Benchmarks (In Progress)
We are currently conducting a comparative study:
*   **Benchmark A:** Zero-shot TinyLlama (Baseline).
*   **Benchmark B:** TinyLlama + RAG (Context Aware).
*   **Benchmark C:** Fine-Tuned TinyLlama (Domain Adapted).

*Hypothesis: The combination of RAG + Fine-tuning (Matrix architecture) will outperform GPT-3.5 on banking specific tasks while costing 99% less.*

---

## 7. Future Roadmap
*   **Federated Learning:** Train on encrypted customer data across branches without centralized aggregation.
*   **Voice Interface:** Integration with Whisper AI for phone banking.
*   **Agentic Actions:** Allow the AI to securely perform transfers (under human supervision).
