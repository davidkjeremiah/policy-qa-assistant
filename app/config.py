import os

from dotenv import load_dotenv

load_dotenv()

POLICY_DIR = os.getenv("POLICY_DIR", "policies")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
# GEMINI_API_KEY is accepted as a fallback so an existing Gemini setup works unchanged.
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "25"))       # seconds per model call
REQUEST_BUDGET = float(os.getenv("REQUEST_BUDGET", "50"))  # total model time per request (contract: 60 s)

TOP_K = int(os.getenv("TOP_K", "8"))                  # sections sent to the model
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))     # cosine gate for early refusal
