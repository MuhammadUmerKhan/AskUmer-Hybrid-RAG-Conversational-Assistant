import os
from dotenv import load_dotenv
from src.logger import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info({"message": "📂 Loaded environment variables"})

# Configuration settings for the RAG application
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error({"message": "❌ Missing GROQ_API_KEY in .env"})
    raise ValueError("Missing GROQ_API_KEY in .env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.error({"message": "❌ Missing OPENROUTER_API_KEY in .env"})
    raise ValueError("Missing OPENROUTER_API_KEY in .env")

RESUME_PATH = os.getenv("RESUME_PATH", os.path.join("assets", "Muhammad_Umer_Khan_AI_Resume.pdf"))
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

# ---------------------------------------------------------------------------
# Knowledge sources
#
# The résumé is one page — roughly 4k characters — and it was the bot's entire
# knowledge. It cannot answer anything about how the projects are actually
# built: not the DeepEval threshold, the offline HMAC/scan_nonce design, the
# LoRA configs, or the red-team findings.
#
# The portfolio already publishes that material as markdown generated from its
# single source of truth (src/data.js) on every deploy, precisely so machines
# can read it. Ingesting those routes triples the corpus AND means the bot
# re-syncs itself whenever the site redeploys — no manual copy step, so it
# cannot drift out of date again.
# ---------------------------------------------------------------------------
PORTFOLIO_SITE = os.getenv("PORTFOLIO_SITE", "https://umerr.vercel.app").rstrip("/")

# llms.txt is deliberately excluded — it is an index of links, not prose, and
# would only add retrievable noise.
KNOWLEDGE_ROUTES = ("about.md", "experience.md", "projects.md", "education.md")

# Snapshot committed alongside the code. Used when the live fetch fails, so a
# portfolio outage or a cold start with no egress cannot take the bot down.
KNOWLEDGE_CACHE_DIR = os.getenv("KNOWLEDGE_CACHE_DIR", os.path.join("assets", "knowledge"))
KNOWLEDGE_TIMEOUT = float(os.getenv("KNOWLEDGE_TIMEOUT", "15"))

# Public contact details, defined once. These used to be hardcoded inside the
# prompt string, where the LinkedIn URL silently drifted out of agreement with
# the one on the portfolio — the bot was handing visitors a different address.
# Written as a markdown bullet list, not a "·"-joined sentence. The frontend
# renders each line as its own bullet; joined inline, four long values wrapped
# into an unreadable paragraph with URLs colliding mid-line.
CONTACT = "\n".join([
    "",
    "- Email: muhammadumerk546@gmail.com",
    "- WhatsApp: +92 343 2187868",
    "- LinkedIn: https://www.linkedin.com/in/muhammadumerkhan-ai/",
    f"- Portfolio: {PORTFOLIO_SITE}",
])

# Validate resume path
if not os.path.exists(RESUME_PATH):
    logger.error({"message": f"❌ PDF not found at {RESUME_PATH}"})
    raise FileNotFoundError(f"PDF not found at {RESUME_PATH}")
logger.info({"message": f"✅ Validated resume path: {RESUME_PATH}"})