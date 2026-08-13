import httpx, os, cachetools, warnings
from typing import List
from langsmith import traceable
from functools import lru_cache
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.config import (
    GROQ_API_KEY, RESUME_PATH, MODEL_NAME, EMBEDDING_MODEL, OPENROUTER_API_KEY,
    PORTFOLIO_SITE, KNOWLEDGE_ROUTES, KNOWLEDGE_CACHE_DIR, KNOWLEDGE_TIMEOUT, CONTACT,
)
from src.logger import logging

# Ignore warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class OpenRouterEmbeddings(Embeddings):
    """Custom LangChain Embeddings class that uses langchain-openrouter."""
    
    def __init__(self, model: str, api_key: str):
        self.chat_router = ChatOpenRouter(
            model=model,
            api_key=api_key
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.chat_router.client.embeddings.generate(
            model=self.chat_router.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.chat_router.client.embeddings.generate(
            model=self.chat_router.model,
            input=text,
        )
        return response.data[0].embedding

class CustomDocChatbot:
    """A RAG-based chatbot for answering questions using a resume PDF."""
    
    query_cache = cachetools.TTLCache(maxsize=500, ttl=600)

    def __init__(self):
        """Initialize the chatbot with LLM, embeddings, and RAG chain."""
        self.llm = self.configure_llm()
        self.embeddings = None
        self.vector_db = None
        self.qa_chain = None
        self.http_client = httpx.AsyncClient(timeout=15.0)
        logger.info({"message": "🤖 CustomDocChatbot initialized"})

    def configure_llm(self):
        """Configure the Groq LLM with specified model and API key."""
        try:
            llm = ChatGroq(
                model_name=MODEL_NAME,
                temperature=0.5,
                groq_api_key=GROQ_API_KEY
            )
            logger.info({"message": "✅ Groq LLM configured successfully"})
            return llm
        except Exception as e:
            logger.error({"message": f"❌ Failed to configure LLM: {str(e)}"})
            raise 
    
    @traceable(run_type="tool", name="Embeddings_Initializer")
    def configure_embedding_model(self):
        """Configure OpenRouter embeddings using ChatOpenRouter."""
        try:
            embeddings = OpenRouterEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=OPENROUTER_API_KEY
            )
            logger.info({"message": "✅ OpenRouter embeddings configured"})
            return embeddings
        except Exception as e:
            logger.error({"message": f"❌ Failed to configure OpenRouter embeddings: {str(e)}"})
            raise

    @lru_cache(maxsize=1)
    @traceable(run_type="tool", name="Corpus_Loader")
    def load_documents(self):
        """Load the résumé PDF plus the portfolio's published knowledge routes.

        The résumé alone is ~4k characters and says nothing about how anything
        was built. The portfolio publishes markdown generated from its single
        source of truth on every deploy; pulling it in triples the corpus and
        keeps the bot in sync with the site automatically.
        """
        try:
            if not os.path.exists(RESUME_PATH):
                raise FileNotFoundError(f"PDF not found at {RESUME_PATH}")
            docs = PyPDFLoader(RESUME_PATH).load()
            logger.info({"message": f"📄 Loaded {len(docs)} résumé page(s) from {RESUME_PATH}"})

            docs.extend(self._load_knowledge_routes())
            chars = sum(len(d.page_content) for d in docs)
            logger.info({"message": f"📚 Corpus ready: {len(docs)} documents, {chars:,} chars"})
            return docs
        except Exception as e:
            logger.error({"message": f"❌ Error loading corpus: {str(e)}"})
            raise

    @traceable(run_type="tool", name="Knowledge_Routes")
    def _load_knowledge_routes(self):
        """Fetch the portfolio's markdown routes, falling back to the snapshot.

        Live-first so the bot tracks the site without anyone remembering to
        sync it. Snapshot-second so a portfolio outage, a DNS blip or a cold
        start without egress degrades the answers rather than failing the
        boot — the résumé still loads either way.

        A successful fetch refreshes the snapshot on disk, so the fallback
        stays useful instead of ageing into the very staleness this fixes.
        """
        docs = []
        os.makedirs(KNOWLEDGE_CACHE_DIR, exist_ok=True)

        for route in KNOWLEDGE_ROUTES:
            url = f"{PORTFOLIO_SITE}/{route}"
            cache_path = os.path.join(KNOWLEDGE_CACHE_DIR, route)
            text, origin = None, None
            try:
                res = httpx.get(url, timeout=KNOWLEDGE_TIMEOUT, follow_redirects=True)
                res.raise_for_status()
                text, origin = res.text, "live"
                with open(cache_path, "w", encoding="utf8") as fh:
                    fh.write(text)
            except Exception as e:
                logger.warning({"message": f"⚠️ {route}: live fetch failed ({e}); trying snapshot"})
                if os.path.exists(cache_path):
                    with open(cache_path, encoding="utf8") as fh:
                        text, origin = fh.read(), "snapshot"

            if not text or not text.strip():
                logger.error({"message": f"❌ {route}: no content from live or snapshot — skipped"})
                continue

            docs.append(Document(page_content=text, metadata={"source": route, "origin": origin}))
            logger.info({"message": f"📥 {route}: {len(text):,} chars ({origin})"})

        return docs

    @traceable(run_type="tool", name="Text_Splitter")
    def split_documents(self, docs):
        """Chunk the corpus so every chunk says what it belongs to.

        Plain recursive splitting left 11 of 20 project chunks with no
        heading: a chunk reading "How it works: hybrid retrieval, Qdrant
        dense..." arrived with nothing identifying it as AskUmer. Retrieval
        then handed the model several such orphans from different projects at
        once and it stapled them together — describing AskUmer as running on
        React/Expo/Supabase (that is the housing platform) and the Revera
        try-on system as using SmartSearch's search stack.

        Splitting on markdown headings first and stamping the heading onto
        every chunk makes each one self-identifying, so facts cannot migrate
        between projects.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900, chunk_overlap=120, add_start_index=True
            )
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "doc"), ("##", "section")],
                strip_headers=False,
            )

            splits = []
            for doc in docs:
                source = doc.metadata.get("source", "résumé")
                is_markdown = str(source).endswith(".md")

                sections = (
                    header_splitter.split_text(doc.page_content) if is_markdown else [doc]
                )
                for section in sections:
                    label = section.metadata.get("section") or section.metadata.get("doc") or source
                    for chunk in splitter.split_documents([section]):
                        # The tag is what stops a chunk being attributed to the
                        # wrong project once it is out of document order.
                        chunk.page_content = f"[Source: {source} — {label}]\n{chunk.page_content}"
                        chunk.metadata["source"] = source
                        chunk.metadata["section"] = label
                        splits.append(chunk)

            tagged = sum(1 for s in splits if s.page_content.startswith("[Source:"))
            logger.info({"message": f"📑 Created {len(splits)} chunks ({tagged} source-tagged)"})
            return splits
        except Exception as e:
            logger.error({"message": f"❌ Error splitting documents: {str(e)}"})
            raise

    # Terse by design. Every token here is paid on every request, and a long
    # rulebook does not make a small model more obedient — it dilutes the
    # instructions that matter. Rules are ordered by how often they bind:
    # grounding first, because hallucinated employers and invented metrics are
    # the only failure that actually costs Umer a job.
    PROMPT = """You are Muhammad Umer Khan, an AI engineer, answering a visitor on your portfolio. First person.

RULES
1. Use only CONTEXT. Never invent metrics, dates, employers, model names or clients.
2. Every chunk is tagged [Source: file — section]. Facts belong ONLY to their own tag. Never describe one project with another's stack, metrics or features.
3. Greeting or small talk ("hi", "thanks"): one warm line, invite a question. It is not a lookup — never answer it with "not in context".
4. Genuinely absent: say so in one line, then offer the nearest thing you can answer. Never guess.
5. Dates: "Feb 2025 — Present" means still there. Never infer a role ended, and never state availability that CONTEXT does not.
6. Max 3 sentences, or up to 5 bullets for a real list. No preamble, no sign-off.
7. Keep specifics — model names, thresholds, tools. They are the evidence.
8. Claim exactly what CONTEXT supports. Never upgrade "in development" to "shipped" or a threshold into an achieved score.
9. Give contact details ONLY if asked how to reach you: {contact}

CONTEXT:
{context}

Q: {question}
A:"""

    def _build_index(self):
        """Build the retriever and chain once per process.

        This used to live inside the per-query path: every single message
        re-chunked the corpus, rebuilt BM25, and re-embedded every chunk
        through the embeddings API before answering. Identical work, identical
        result, on every request — the dominant cost and latency in the
        service, and it would have scaled with the larger corpus.
        """
        if self.qa_chain is not None:
            return

        splits = self.split_documents(self.load_documents())

        if self.embeddings is None:
            self.embeddings = self.configure_embedding_model()

        self.vector_db = FAISS.from_documents(splits, self.embeddings)
        faiss_retriever = self.vector_db.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12}
        )

        # BM25 earns its keep on this corpus: exact tokens like "DeepEval",
        # "scan_nonce", "QLoRA" or "gpt-oss-120b" are precisely what a
        # technical visitor types, and lexical match beats embeddings on them.
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 4

        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            partial_variables={"contact": CONTACT},
            template=self.PROMPT,
        )

        # No ConversationBufferMemory. There was one, rebuilt per query, so it
        # never retained anything — but "fixing" it would have been worse than
        # the bug: main.py holds ONE chatbot instance for the whole service and
        # POST /chat carries no session id, so a persistent buffer would feed
        # one visitor's conversation into the next visitor's answers. Stateless
        # is the correct design until the API can identify a session; it also
        # skips the question-condensing LLM call on every turn.
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,
            verbose=False,
        )
        logger.info({"message": f"🚀 Index + chain built once: {len(splits)} chunks"})

    @traceable(run_type="chain", name="RAG_Pipeline")
    def setup_and_query(self, question: str):
        """Answer a question against the prebuilt index."""
        try:
            self._build_index()
            result = self.qa_chain.invoke({"question": question, "chat_history": []})
            response = result["answer"].strip()
            logger.info({"message": f"💬 Query: {question} | Answer: {response}"})
            return response
        except Exception as e:
            logger.error({"message": f"❌ Error in RAG pipeline: {str(e)}"})
            raise

    async def query(self, question: str) -> str:
        """Process a user query through the RAG chain with caching."""
        try:
            if question in self.query_cache:
                response = self.query_cache[question]
                logger.info({"message": f"💾 Cache hit for query: {question}"})
                return response

            response = self.setup_and_query(question)
            self.query_cache[question] = response
            return response
        except Exception as e:
            logger.error({"message": f"❌ Query error: {str(e)}"})
            raise

    async def shutdown(self):
        """Clean up resources on shutdown."""
        await self.http_client.aclose()
        logger.info({"message": "🛑 HTTP client closed"})