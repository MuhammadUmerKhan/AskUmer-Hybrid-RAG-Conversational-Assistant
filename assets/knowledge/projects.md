# Projects — Muhammad Umer Khan

## DineMate
Agentic, voice-enabled food ordering system.

**Problem:** Restaurants take orders by phone and app, then retype them for the kitchen — slow, error-prone, and invisible to staff until someone transcribes it.

**Evidence:** Four-agent system running on Streamlit Cloud, hardened after a self-run red-team pass that found and fixed price-tampering and IDOR flaws.

**How it works:**
- Four LangGraph agents split the work: a chatbot agent extracts order details, an order-management agent handles modifications and cancellations, a kitchen agent pushes confirmed orders to staff, and an admin agent updates menu and pricing live.
- Voice ordering runs Whisper ASR for speech-to-text and Torch TTS for spoken replies, so a customer can order without touching a screen.
- openai/gpt-oss-120b handles food-domain language understanding; SQLite with indexed tables holds menu, orders and users.
- Role-gated dashboards for customers, kitchen, support and admin, with analytics on revenue trends, top items, peak hours and spending patterns.
- Orders stay editable for ten minutes, then lock — the kitchen never works from a moving target.
- Red-teamed the deployment and fixed what it found: price tampering on the client side, and IDOR gaps in order access.

**Stack:** LangGraph, gpt-oss-120b, Whisper, Torch TTS, SQLite

**Live:** https://dinemate.streamlit.app/
**Code:** https://github.com/MuhammadUmerKhan/DineMate-Agentic-AI-for-Automated-Food-Ordering

## AskUmer
Production Hybrid-RAG conversational assistant.

**Problem:** A recruiter spends under a minute on a portfolio. Standard RAG breaks on multi-hop questions and ships with no safety gate and no evaluation.

**Evidence:** 898 indexed chunks behind a DeepEval CI gate at 0.7 faithfulness and answer relevancy — it is the assistant answering questions on this page.

**How it works:**
- Hybrid retrieval: Qdrant dense search (BAAI/bge-base-en-v1.5, 768-dim) and rank-bm25 sparse search, fused with Reciprocal Rank Fusion and reranked by a FlashRank cross-encoder down to the top five documents.
- A LangGraph ReAct agent decides per turn whether to search at all, so a follow-up answerable from conversation memory costs zero database calls.
- NeMo Guardrails runs as an input gate — jailbreaks, prompt injection and off-topic queries are blocked before any retrieval or LLM call happens.
- Portkey gateway handles failover: gpt-oss-120b falls back to gpt-oss-20b, three retries on 429/503, with response caching in front.
- A two-phase DeepEval suite generates live answers, then scores them LLM-as-judge on Faithfulness and Answer Relevancy; CI exits non-zero if any sample falls below 0.7.
- Observability through three tools at once — Logfire, LangSmith and Portkey.

**Stack:** LangGraph, Qdrant, FlashRank, NeMo Guardrails, Portkey, DeepEval
**Code:** https://github.com/MuhammadUmerKhan/AskUmer-Hybrid-RAG-Conversational-Assistant

## Diagnosify
LLM-powered medical report insights.

**Problem:** Patients cannot read their own lab reports, and clinicians lose consultation time explaining them line by line.

**Evidence:** Built in 12 hours · Top 10 at the SMIT AI Hackathon.

**How it works:**
- Accepts PDF, PNG and JPEG. PyPDF2 extracts text from PDFs, with pytesseract and OpenCV as an optional path for photographed reports.
- Extracted values are structured into JSON, and every result is classified Normal, Borderline or Critical rather than left as raw numbers.
- Groq's llama-4-scout-17b-16e-instruct writes the plain-language explanation and the summary.
- A RAG chatbot answers follow-up questions grounded only on the uploaded report, retrieving over FAISS with sentence-transformers embeddings.
- RAGAS scores the chatbot's faithfulness on its own evaluation page, so the grounding claim is measured rather than asserted.
- Chat history persists to MongoDB Atlas; results export as a formatted PDF via reportlab.

**Stack:** llama-4-scout, FAISS, RAGAS, PyPDF2, MongoDB

**Live:** https://smit-hackathon-ai-medical-report-analyzer.streamlit.app/
**Code:** https://github.com/MuhammadUmerKhan/Diagnosify-LLM-Powered-Medical-Report-Insights

## Housing Society Management Platform
White-label SaaS for housing-society gate security and operations.

**Problem:** Housing societies run visitor and staff access control on paper gate registers — slow, unauditable, and impossible to verify when the connection drops.

**Evidence:** Live in production for a paying client since April 2026, on a white-label architecture where each society gets its own Supabase project, web build and mobile build.

**How it works:**
- A Vite + React admin panel and a single Expo app serving both residents and gate keepers, role-gated, over a shared Supabase backend.
- Gate passes validate with no network at all: QR codes are signed with HMAC-SHA256 via @noble/hashes (expo-crypto has no keyed HMAC), and scans queue in expo-sqlite until connectivity returns.
- Replay is prevented by a scan_nonce UUID under a UNIQUE(qr_code_id, scan_nonce) constraint with ON CONFLICT DO NOTHING — a re-sent scan is idempotent instead of a second entry in the log.
- The licence check has three states — ACTIVE, INACTIVE, UNKNOWN — and a network failure resolves to UNKNOWN and fails open. A gate must never be locked shut by an unreachable API.
- Roles live in Supabase app_metadata, never raw_user_meta_data, which users can edit themselves. The service-role key exists only inside Edge Functions; clients hold the anon key and go through RLS.
- HMAC keys and tokens sit in the device Keychain/Keystore via expo-secure-store — never AsyncStorage, never the JS bundle.
- Every admin action writes to an append-only audit log.

**Stack:** React, Expo, Supabase, PostgreSQL, HMAC-SHA256

## LexiAgent
Autonomous legal-document analysis agent.

**Problem:** Reading contracts and NDAs by hand is slow, and the clause that matters is usually the one you skim past.

**Evidence:** Classifies, extracts, risk-flags and summarises a document in a single pass, returning clauses as structured JSON.

**How it works:**
- LangGraph orchestrates a fixed sequence: load and chunk, classify document type, extract clauses, analyse risks, summarise.
- Task-specific model routing — qwen-qwq-32b for classification and clause extraction, llama-4-scout-17b for summarisation, risk analysis and reasoning.
- Handles PDF, DOCX and TXT through pdfplumber and python-docx, chunked with RecursiveCharacterTextSplitter.
- Detects the document type (NDA, lease, employment contract) and flags ambiguous or problematic language, rather than only condensing it.
- A chat interface answers questions against the loaded document once the pipeline has run.

**Stack:** LangGraph, qwen-qwq-32b, llama-4-scout, pdfplumber, Streamlit

**Live:** https://lexiagent-ai-powered-autonomous-legal-document-analyst.streamlit.app/
**Code:** https://github.com/MuhammadUmerKhan/LexiAgent-Autonomous-Legal-Document-Analysis

## SupportGenie
Dual fine-tuned LLM customer-support chatbot.

**Problem:** Generic support bots answer banking questions with fluent, plausible, off-domain text — confidently wrong in the one domain where that is expensive.

**Evidence:** Two models fine-tuned to final loss 1.78 and 1.49 on a single free-tier Colab T4, on 17,000 and 1,764 instruction pairs.

**How it works:**
- Mistral-7B-Instruct-v0.3 fine-tuned with LoRA (r=32, alpha=64, dropout=0.05) on 17,000 instruction-response pairs — roughly 12 hours of training.
- LLaMA-3-8B-Instruct fine-tuned with LoRA (r=8, alpha=16, dropout=0.1) on 1,764 pairs — roughly 10 hours.
- Both trained in 4-bit quantisation to fit a single free-tier T4; the base data was about 1,000 real bank FAQ pairs, expanded with synthetic augmentation.
- FAISS semantic retrieval serves a matched FAQ where one exists, falling through to the fine-tuned models only when it does not.
- Sentiment classification tags every exchange positive, negative or neutral, feeding a live analytics dashboard.
- Chat history, feedback and analytics persist to MongoDB.

**Stack:** LoRA / PEFT, Mistral-7B, LLaMA-3-8B, FAISS, MongoDB

**Live:** https://ai-powered-customer-support-and-analytics-system.streamlit.app/
**Code:** https://github.com/MuhammadUmerKhan/SupportGenie-Dual-Fine-Tuned-LLM-Customer-Support-Chatbot

## SmartSearch
LLM-based semantic search engine.

**Problem:** Search returns ten links and leaves the reading and synthesis to you.

**Evidence:** Returns one cited answer assembled from live search results — or from a set of URLs you supply yourself.

**How it works:**
- Google Custom Search API fetches live results; Newspaper3k scrapes the article bodies behind them.
- Content is chunked, embedded with all-MiniLM-L6-v2 and indexed into FAISS per query.
- The model answers only from retrieved chunks and returns its source links alongside the answer.
- Text is capped at 5,000 characters per document, which keeps latency and token cost predictable.
- A custom-URL mode indexes pages you provide, so the same pipeline searches your own sources instead of the open web.
- The generation model is swappable across llama-3.3-70b-versatile, gemma2-9b-it and qwen3-32b.

**Stack:** LangChain, FAISS, all-MiniLM-L6-v2, Llama 3.3, Streamlit

**Live:** https://ai-powered-search-engine-with-llms.streamlit.app/
**Code:** https://github.com/MuhammadUmerKhan/SmartSearch-LLM-Based-Semantic-Search-Engine

## Community Membership & Services Platform
Bilingual mobile + admin platform for a global membership community.

**Problem:** A membership community with thousands of member families had no digital system for household registration or benefit applications — every request went through manual, paper-based review.

**Evidence:** In development for a client-commissioned rollout targeting 5,000–50,000 members globally.

**How it works:**
- A bilingual Expo app (English and Urdu, full RTL) alongside a Vite + React admin portal, in a pnpm workspaces monorepo — pnpm rather than npm because it catches the phantom-dependency bugs that bite React Native and Metro.
- A person-centric family graph rather than flat household rows: two-sided edges that model polygamy, divorce, deceased members and duplicate detection.
- One submission engine drives every module — zakat, awards, jobs, fees, announcements — through per-module views, instead of five parallel CRUD stacks.
- Shared Zod schemas are the data contract, enforced at three boundaries: the form resolver, the Edge Function validator, and a database CHECK constraint. Drift between those three is the classic failure mode when running JavaScript without TypeScript.
- Push fan-out respects Expo's send limits with chunking, receipt checks and dead-token cleanup.
- RTL is done properly — physical properties mapped to logical ones, icon mirroring, Western digits with locale-aware dates.

**Stack:** React, Expo, Supabase, PostgreSQL, Zod

## AuraClaw
Autonomous AI agent framework built from scratch in pure Python, inspired by OpenClaw.

**Problem:** Off-the-shelf agent frameworks are heavy, opinionated, and hard to control once you need behaviour they did not anticipate.

**Evidence:** A reusable foundation now underneath multiple agent products rather than a one-off.

**How it works:**
- LangGraph orchestration over a custom tool registry, written from scratch in pure Python.
- A multi-provider LLM gateway, so one agent can be re-pointed between providers without touching its logic.
- A WhatsApp channel for talking to the agent directly from a phone.
- Modular by construction — tools, memory and channels are each swappable in isolation.

**Stack:** Python, LangGraph, LangChain, WhatsApp
