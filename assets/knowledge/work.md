# Work

Every build by Muhammad Umer Khan: the four on the homepage first, then the archive. Case studies are in full; receipts are links.

## Interviewly

2026 · verified · Product, build and deployment, solo

Live AI voice interviews for hiring teams, where no model ever scores a candidate.

Stack: LiveKit, Deepgram, Cartesia, Groq, LangGraph, FastAPI, Next.js, Supabase.

[case study](https://buildwithumer.site/work/interviewly) · [live app](https://interviewlyy.vercel.app)

### The problem

Companies get more candidates than their recruiters can meet. A first-round technical interview run by a voice agent saves those hours, but only if the recruiter trusts what comes back, the candidate gets a fair hearing, and the tool never becomes the one deciding who is hired.

### What I built

- Real spoken interviews over WebRTC: LiveKit carries the audio, Deepgram Nova-3 transcribes it, Cartesia Sonic-3 speaks, and a Groq model reached through a LangGraph graph writes every line. LiveKit's own reply generation is never used.
- Two models, and neither scores. One asks the questions live; a separate pass after the interview writes what was and was not said. The live model has no path to score or review data at all, because under NYC Local Law 144 a tool that scores candidates is an automated employment decision tool, and one that writes notes for a human is not.
- Plain code runs the interview. Topic coverage, the minimum number of questions, the follow-up cap and the time limit are decided in Python: the model can propose a follow-up, and the controller decides whether it may ask it.
- An invite link carries a random token, never a candidate id, and the browser trades it once for a short-lived credential.

### How it holds up

- Live: the web app on Vercel, the API, voice worker, Redis and the Judge0 code runner on AWS, the database on Supabase. A merge to main deploys both.
- Proctoring counts strikes on the server: fullscreen exits, window switches, gaze and a verified screen share, with the evidence kept.
- A hiring decision is refused until the recruiter has scored every answer by hand.
- A candidate whose connection drops rejoins at the question they were on, instead of starting over.

### If you're building something similar

- Put the legal line in the architecture, not in a prompt.
- Let code decide the flow, and let the model only word it.
- Remove an attack surface rather than defend it: the closing line is scripted, and candidates have no channel to question the model.

## WhatsApp hotel-lead bot

2026 · private · End-to-end build at Saylani Tech

Turns each WhatsApp hotel enquiry into a qualified lead, routes it to the team, and relays their answer back.

Stack: FastAPI, GPT-4o-mini, Pydantic, PostgreSQL, Supabase, Wazzup.

Live in production for a travel agency, built at Saylani Tech.

[case study](https://buildwithumer.site/work/whatsapp-lead-automation)

### The problem

A travel agency that resells hotel rooms took every enquiry by hand on WhatsApp. Customers wrote in Roman Urdu, Urdu, English or Arabic, left out dates and room counts, asked about several cities in one thread, and waited while someone retyped each request for the team that had the prices.

### What I built

- GPT-4o-mini only proposes: it returns a typed patch through structured outputs, and Python validates it and merges it into the lead. An empty field never erases what the customer said earlier, and a correction always gets a fresh confirmation.
- One enquiry can cover several stays, such as two cities on one trip. Each message is routed to the stay it is about, and the customer confirms the whole requirement before anything reaches the team.
- Every confirmed lead gets a query ID and goes to the team's WhatsApp group. A team member answers by swiping on the lead or quoting its ID, the bot matches the answer to its lead, and it reaches the right customer.
- It replies in the language the customer wrote in, and ignores the acknowledgements, greetings and chatter that would otherwise open false leads.

### How it holds up

- Live in production on the agency's WhatsApp, with separate development and production webhooks.
- 233 tests cover the webhook, corrections, confirmation gating, dates, multi-city stays and model failures.
- The model is chosen from a database table, then the environment, then a built-in default, so switching models needs no deploy.
- A daily operations summary goes to the team's group, and an invoice swiped in the group is forwarded to the right customer.

### If you're building something similar

- Let the model propose and the code decide.
- Confirm before a lead reaches a human, and again after every correction.
- Give every lead an ID people can quote, so an answer never reaches the wrong customer.

## Society-management platform

2026 · private · Main developer since June 2026, in a two-person team

A white-label platform that runs a housing society: residents, visitors, staff, complaints and the gate.

Stack: React, Expo, Supabase, PostgreSQL, Row-level security.

White-label society-management software built at Revera Innovations, live with its first housing society.

[case study](https://buildwithumer.site/work/society-management-platform)

### The problem

Housing societies ran their gates and their residents on paper registers and phone calls. Nobody could audit who came in, a person the society had banned could still talk their way through, and nothing kept working when the gate lost its connection. Every society also needed its own branded deployment.

### What I built

- I joined after the first release. The original developer built the core: signed QR passes that validate offline, the offline scan queue, the licence check and the audit log. Since June I have been the platform's main developer.
- Admin roles: I consolidated the platform to four roles and added scoped sub-admins. A super-admin grants sections, and the database enforces each grant through row-level security, so a sub-admin cannot read or write a section it was not given, even by calling the API directly.
- A gate blocklist: a society can flag a person or a vehicle, and the database refuses to issue them a pass, through a trigger on manual entries and a check in the event-arrival function. Gate phones keep a copy of the list, so the guard is warned even offline.
- Gate-mounted scanners: I added USB scanner input, so drivers hold their phone up to a fixed reader and the scan runs through the same pipeline as the camera.

### How it holds up

- Live with its first housing society, each society on its own Supabase project, web build and mobile build.
- Every grant, revocation and blocklist change is written to the audit log.
- I fixed a refetch storm in the gate app's list of people inside, which set in at around 310 people.
- I reskinned every gate-keeper screen to the new design system with no change to its behaviour.

### If you're building something similar

- Enforce permissions in the database. A hidden button is not access control.
- Make a ban a database rule, so no app version can skip it.
- Keep what the gate needs on the device. The network fails at the worst moment.

## AskUmer

2026 · verified · Design, build, evaluation

An assistant that refuses to ship worse answers: CI fails below 0.7 faithfulness.

Stack: LangGraph, Qdrant, BM25, FlashRank, NeMo Guardrails, Portkey, DeepEval, FastAPI.

[case study](https://buildwithumer.site/work/askumer) · [source](https://github.com/MuhammadUmerKhan/AskUmer-Hybrid-RAG-Conversational-Assistant) · [CI workflow](https://github.com/MuhammadUmerKhan/AskUmer-Hybrid-RAG-Conversational-Assistant/blob/main/.github/workflows/ci.yml)

### The problem

A founder or recruiter gives a portfolio under a minute. A chatbot that answers from a vague memory of the site, or makes things up, does more harm than no chatbot. Plain RAG also breaks on multi-hop questions and ships with no safety gate and no way to tell when a change made it worse.

### What I built

- Hybrid retrieval: Qdrant dense search (BAAI/bge-base-en-v1.5, 768-dim) and BM25 sparse search, fused with Reciprocal Rank Fusion and reranked by a FlashRank cross-encoder down to the top five documents.
- A LangGraph ReAct agent decides on every turn whether to search at all, so a follow-up answerable from the conversation costs zero database calls.
- NeMo Guardrails runs as an input gate: jailbreaks, prompt injection and off-topic questions are blocked before any retrieval or model call.
- A Portkey gateway handles failover from gpt-oss-120b to gpt-oss-20b, with three retries on 429 and 503 and response caching in front.

### How it holds up

- 898 indexed chunks.
- A two-phase DeepEval suite generates live answers, then scores Faithfulness and Answer Relevancy with an LLM judge. CI exits non-zero if any sample falls below 0.7.
- Every request is traced in Logfire, LangSmith and Portkey.
- A lighter build of it runs this site's chat: FAISS and BM25 search over these pages and my résumé, answered by gpt-oss-120b on Groq, without the agent or the guardrails.

### If you're building something similar

- Decide what worse means before you build: pick the metrics and the threshold, and make CI enforce them.
- Put guardrails before retrieval, not after generation. It is cheaper and stops injection earlier.
- Let the agent skip retrieval when the answer is already in the conversation.

## DineMate

2025 · verified · Design, build, red-team

Found and fixed a price-tampering flaw before anyone could exploit it.

Stack: LangGraph, gpt-oss-120b, Whisper, Torch TTS, SQLite.

[case study](https://buildwithumer.site/work/dinemate) · [live demo](https://dinemate.streamlit.app/) · [security report](https://github.com/MuhammadUmerKhan/DineMate-Agentic-AI-for-Automated-Food-Ordering/blob/main/SECURITY.md) · [source](https://github.com/MuhammadUmerKhan/DineMate-Agentic-AI-for-Automated-Food-Ordering)

### The problem

Restaurants take orders by phone and app, then retype them for the kitchen: slow, error-prone, and invisible to staff until someone transcribes them. An AI ordering system that takes money is also an attack surface.

### What I built

- Four LangGraph agents split the work: one extracts order details, one handles changes and cancellations, one pushes confirmed orders to the kitchen, one lets an admin update the menu and prices live.
- Voice ordering runs Whisper for speech-to-text and Torch TTS for spoken replies, so a customer can order without touching a screen.
- Orders stay editable for ten minutes, then lock, so the kitchen never works from a moving target.
- Role-gated dashboards for customers, kitchen, support and admin, with analytics on revenue, top items and peak hours.

### How it holds up

- Red-teamed across 8 attack categories mapped to the OWASP LLM Top 10 and MITRE ATLAS.
- The critical finding: the browser could tamper with prices. Fixed by recomputing every total on the server.
- A Llama Prompt Guard 2 classifier screens input.
- The remaining findings, including an IDOR gap, are documented as accepted risks in the published report, not hidden.

### If you're building something similar

- Never trust a total the browser sends. Recompute it on the server.
- Red-team before launch, then publish what is still open.
- Give orders a short edit window and lock them after it.

## AuraClaw agent framework

2026 · in progress · Design and build

A from-scratch agent framework: one tool registry and LLM gateway under several agent products.

Stack: Python, LangGraph, LangChain.

## Community membership platform

2026 · in progress · Full-stack build

A bilingual English and Urdu member app and admin portal, with every module running on one submission engine.

Stack: React, Expo, Supabase, PostgreSQL, Zod.

## Wholesale B2B website

2026 · private · Site, admin panel and Shopify theme, at Saylani Tech

A clothing exporter's catalogue and enquiry site with its own admin panel, where an edit goes live within a minute and needs no redeploy.

Stack: Next.js 15, Firebase, Amazon S3, Shopify.

## Supermarket WhatsApp chatbot

2026 · private · WhatsApp admin and human handover, in a team build at Saylani Tech

A supermarket chain's shopping assistant on WhatsApp. My part: the admin view of WhatsApp conversations and the handover to a human.

Stack: FastAPI, OpenAI Agents SDK, Qdrant, React.

## SupportGenie

2025 · verified · Fine-tuning and build

Two LLMs fine-tuned on a single free-tier GPU to final loss 1.78 and 1.49, answering banking support questions.

Stack: LoRA, Mistral-7B, LLaMA-3-8B, FAISS, MongoDB.

[live demo](https://ai-powered-customer-support-and-analytics-system.streamlit.app/) · [source](https://github.com/MuhammadUmerKhan/SupportGenie-Dual-Fine-Tuned-LLM-Customer-Support-Chatbot)

## Diagnosify

2025 · verified · Hackathon build

Lab reports read by OCR and explained in plain language. Top 10 at the SMIT AI Hackathon, built in 12 hours.

Stack: OCR, FAISS, RAGAS, MongoDB.

[live demo](https://smit-hackathon-ai-medical-report-analyzer.streamlit.app/) · [source](https://github.com/MuhammadUmerKhan/Diagnosify-LLM-Powered-Medical-Report-Insights)

## LexiAgent

2025 · verified · Design and build

Contracts classified, clause-extracted, risk-flagged and summarised in one pass.

Stack: LangGraph, pdfplumber, Streamlit.

[live demo](https://lexiagent-ai-powered-autonomous-legal-document-analyst.streamlit.app/) · [source](https://github.com/MuhammadUmerKhan/LexiAgent-Autonomous-Legal-Document-Analysis)

## SmartSearch

2025 · verified · Design and build

One cited answer assembled from live search results instead of ten links.

Stack: LangChain, FAISS, all-MiniLM-L6-v2.

[live demo](https://ai-powered-search-engine-with-llms.streamlit.app/) · [source](https://github.com/MuhammadUmerKhan/SmartSearch-LLM-Based-Semantic-Search-Engine)

## Dental-clinic voice receptionist

Undated · private · Voice agent build

Answers the call, offers open Cal.com slots, books, and creates the lead and deal in HubSpot.

Stack: Vapi, Retell AI, Cal.com, HubSpot.

## Portfolio for a performance-marketing consultant

2026-10 · private · Figma to production

A Figma design shipped pixel-perfect.

Stack: Next.js 16, TypeScript, Tailwind CSS v4, Motion.

Shown anonymised until the client agrees to be named.

[case study](https://buildwithumer.site/work/marketing-portfolio-build)

### The problem

A consultant had a finished Figma design: a long homepage, six case-study pages, and springs and hover states specified in the prototype. The site had to match it exactly at every width and stay easy to update.

### What I built

- Matched the Figma frame to the pixel at 1600px: a 13,312px page with every section at its Figma coordinate.
- A one-design-pixel unit, min(0.0625vw, 1px), lets the hero and feature layouts scale with the viewport from 1024px up and flow normally below.
- Motion uses the spring values from the prototype file itself and turns off under reduced motion.
- All copy, links and case studies live in typed content files, so the client changes text without touching layout.

### How it holds up

- Six case-study pages from one template, prerendered at build.
- Built on Next.js 16 with Tailwind CSS v4.

### If you're building something similar

- Read motion values out of the prototype file instead of matching them by eye.
- Separate content from layout before the first copy change arrives.
