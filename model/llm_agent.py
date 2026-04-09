"""
model/llm_agent.py
────────────────────────────────────────────────────────────────────────────────
LLM-based triage agent.

Uses an OpenAI-compatible REST endpoint (configurable via environment variables)
to call a language model for triage decisions.

Environment variables (MANDATORY)
──────────────────────────────────
    API_BASE_URL  – base URL of the OpenAI-compatible API
                    e.g. https://api.openai.com/v1
                         https://api-inference.huggingface.co/v1
    MODEL_NAME    – model identifier, e.g. "gpt-4o-mini", "Qwen/Qwen2.5-72B"
    HF_TOKEN      – bearer token (HuggingFace token or OpenAI key)

The agent sends the ticket text to the LLM and parses a structured JSON
response containing category, priority, and route.

Design
──────
• System prompt includes the full action space to constrain output.
• JSON-only output is enforced via a strict prompt instruction.
• A regex-based JSON extractor handles models that wrap JSON in markdown.
• Falls back to the SFT model if the LLM call fails or returns unparseable output.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

from model.sft_model import TriageModel

# ─── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert IT support ticket triage specialist.
Your job is to analyse a support ticket and make three decisions:

1. CATEGORY — one of: hardware, software, network, access, security
2. PRIORITY  — one of: P1 (critical/system down), P2 (major impact),
                        P3 (partial issue), P4 (minor issue)
3. ROUTE     — one of: hardware_team, dev_team, network_team, auth_team, security_team

Category → Route mapping (default, override only when justified):
  hardware  → hardware_team
  software  → dev_team
  network   → network_team
  access    → auth_team
  security  → security_team

Priority guidance:
  P1: Production down, data breach, >100 users blocked, revenue impact
  P2: Major feature broken, large team blocked, SLA risk
  P3: Partial functionality, workaround available, small team affected
  P4: Minor inconvenience, cosmetic issue, single user affected

Respond ONLY with a valid JSON object and nothing else:
{
  "category": "<category>",
  "priority": "<priority>",
  "route": "<route>",
  "reasoning": "<one sentence explaining your decision>"
}"""

_USER_PROMPT_TEMPLATE = """Triage this IT support ticket:

\"\"\"
{ticket_text}
\"\"\"

Respond with JSON only."""


# ─── JSON extractor ───────────────────────────────────────────────────────────

_JSON_PATTERN = re.compile(r"\{[^{}]+\}", re.DOTALL)

def _extract_json(text: str) -> Optional[Dict]:
    """Extract the first JSON object from model output."""
    match = _JSON_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# ─── Valid values ─────────────────────────────────────────────────────────────

_VALID_CATEGORIES = {"hardware", "software", "network", "access", "security"}
_VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}
_VALID_ROUTES     = {"auth_team", "network_team", "hardware_team", "security_team", "dev_team"}


def _validate_parsed(parsed: Dict) -> bool:
    return (
        parsed.get("category") in _VALID_CATEGORIES
        and parsed.get("priority")  in _VALID_PRIORITIES
        and parsed.get("route")     in _VALID_ROUTES
    )


# ─── Agent class ──────────────────────────────────────────────────────────────

class LLMTriageAgent:
    """
    LLM-powered triage agent using an OpenAI-compatible API.

    Falls back to the SFT baseline model if LLM is unavailable or fails.

    Parameters
    ----------
    use_fallback : bool
        If True, fall back to SFT model on LLM failure.
    """

    def __init__(self, use_fallback: bool = True):
        self._api_base  = os.environ.get("API_BASE_URL",  "https://api.openai.com/v1")
        self._model     = os.environ.get("MODEL_NAME",    "gpt-4o-mini").strip()
        self._token = self._resolve_token()
        self._use_fallback = use_fallback
        
        # Flexibility: Support both name variants for Gemini/Google
        self._gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._groq_key = os.environ.get("GROQ_API_KEY")

        # Initialise Groq client
        self._using_groq = False
        if _GROQ_AVAILABLE and self._groq_key and len(self._groq_key) > 5:
            try:
                # If using default Llama but no specific model name provided, ensure it's a Groq model
                if "llama" in self._model.lower() and "gpt" in self._model.lower(): # Case where default is weird
                    pass 
                
                self._groq_client = Groq(api_key=self._groq_key)
                self._using_groq = True
                print(f"[LLMAgent] Initialised Groq model: {self._model}")
            except Exception as e:
                self._last_error = f"Groq init failed: {e}"
                print(f"[LLMAgent] {self._last_error}")
                self._using_groq = False
        else:
            if not _GROQ_AVAILABLE:
                print("[LLMAgent] Groq library not installed")
            if not self._groq_key:
                print("[LLMAgent] Groq API key missing")

        # Initialise Gemini client
        self._using_gemini = False
        if not self._using_groq and _GEMINI_AVAILABLE and self._gemini_key:
            try:
                # Automagic: if a Gemini key is provided, but model name is the default Llama, switch to Gemini
                if "llama" in self._model.lower() or "gpt" in self._model.lower():
                     self._model = "gemini-1.5-flash"
                     print(f"[LLMAgent] Auto-switching to {self._model} for Gemini backend")

                genai.configure(api_key=self._gemini_key)
                self._gemini_model = genai.GenerativeModel(self._model)
                self._using_gemini = True
                print(f"[LLMAgent] Initialised Gemini model: {self._model}")
            except Exception as e:
                self._last_error = f"Gemini init failed: {e}"
                print(f"[LLMAgent] {self._last_error}")
                self._using_gemini = False
        else:
            if not self._using_groq:
                if not _GEMINI_AVAILABLE:
                    print("[LLMAgent] Gemini library not installed")
                if not self._gemini_key:
                    print("[LLMAgent] Gemini API key missing")

        # Initialise OpenAI client
        if _OPENAI_AVAILABLE and self._token:
            self._client = OpenAI(
                api_key=self._token,
                base_url=self._api_base,
            )
        else:
            self._client = None
        
        self._last_error: Optional[str] = None

        # SFT fallback model (always initialised)
        self._fallback_model: Optional[TriageModel] = None
        if use_fallback:
            self._fallback_model = TriageModel().fit()

    def _resolve_token(self) -> str:
        """Intelligently pick the correct token based on API base URL."""
        hf_token = os.environ.get("HF_TOKEN")
        oi_token = os.environ.get("OPENAI_API_KEY")

        # Case 1: OpenRouter endpoint (explicitly handled)
        if "openrouter.ai" in self._api_base.lower():
            return oi_token or hf_token or ""

        # Case 2: OpenAI endpoint
        if "openai.com" in self._api_base.lower():
            return oi_token or hf_token or ""
        
        # Case 3: HuggingFace endpoint
        if "huggingface.co" in self._api_base.lower():
            return hf_token or oi_token or ""
        
        # Case 4: Default/Unknown endpoint (prefer HF token or fallback to OpenAI)
        return hf_token or oi_token or ""

    # ── Main inference ────────────────────────────────────────────────────────

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Triage a ticket using the LLM.

        Returns
        -------
        dict : {
            "action":    {category, priority, route},
            "reasoning": str,
            "source":    "llm" | "sft_fallback",
        }
        """
        # Try Groq first if configured
        if self._using_groq:
            result = self._groq_predict(text)
            if result is not None:
                return result

        # Try Gemini next if configured
        if self._using_gemini:
            result = self._gemini_predict(text)
            if result is not None:
                return result

        # Try OpenAI next
        if self._client is not None:
            result = self._llm_predict(text)
            if result is not None:
                return result

        # Fallback path
        if self._use_fallback and self._fallback_model is not None:
            sft_result = self._fallback_model.predict_with_confidence(text)
            return {
                "action":    sft_result["action"],
                "reasoning": "SFT fallback (LLM unavailable)",
                "source":    "sft_fallback",
                "confidence": sft_result["confidence"],
            }

        err_detail = f" (Last Error: {self._last_error})" if self._last_error else ""
        if "403" in err_detail:
             err_detail += ". Update token permissions at: https://huggingface.co/settings/tokens"
        
        raise RuntimeError(
            f"LLM call failed{err_detail}. "
            "Verify your API keys and MODEL_NAME in .env."
        )

    def get_backend_status(self) -> Dict[str, Any]:
        """Provides diagnostic information about LLM backends."""
        return {
            "active_backend": "groq" if self._using_groq else ("gemini" if self._using_gemini else ("openai/hf" if self._client else "none")),
            "model": self._model,
            "groq_key": bool(self._groq_key and len(self._groq_key) > 5),
            "gemini_key": bool(self._gemini_key and len(self._gemini_key) > 5),
            "hf_token": bool(self._token and len(self._token) > 5),
            "last_error": self._last_error,
            "connected": self._using_groq or self._using_gemini or self._client is not None,
            "libs": {
                "groq": _GROQ_AVAILABLE,
                "gemini": _GEMINI_AVAILABLE,
                "openai": _OPENAI_AVAILABLE
            }
        }

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _llm_predict(self, text: str) -> Optional[Dict[str, Any]]:
        """Call the LLM API and parse the structured response."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": _USER_PROMPT_TEMPLATE.format(
                        ticket_text=text
                    )},
                ],
                temperature=0.1,  # Low temperature for deterministic triage
                max_tokens=256,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            print(f"[LLMAgent] API call failed: {self._last_error}")
            return None

        parsed = _extract_json(raw_output)
        if parsed is None or not _validate_parsed(parsed):
            print(f"[LLMAgent] Failed to parse valid action from: {raw_output[:200]}")
            return None

        return {
            "action": {
                "category": parsed["category"],
                "priority":  parsed["priority"],
                "route":     parsed["route"],
            },
            "reasoning": parsed.get("reasoning", ""),
            "source":    "llm",
        }

    def _gemini_predict(self, text: str) -> Optional[Dict[str, Any]]:
        """Call the Gemini API and parse the structured response."""
        try:
            # Construct prompt for Gemini
            prompt = f"{_SYSTEM_PROMPT}\n\n{_USER_PROMPT_TEMPLATE.format(ticket_text=text)}"
            
            response = self._gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                )
            )
            raw_output = response.text
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            print(f"[LLMAgent] Gemini call failed: {self._last_error}")
            return None

        parsed = _extract_json(raw_output)
        if parsed is None or not _validate_parsed(parsed):
            print(f"[LLMAgent] Failed to parse valid action from Gemini: {raw_output[:200]}")
            return None

        return {
            "action": {
                "category": parsed["category"],
                "priority":  parsed["priority"],
                "route":     parsed["route"],
            },
            "reasoning": parsed.get("reasoning", ""),
            "source":    "llm (gemini)",
        }

    def _groq_predict(self, text: str) -> Optional[Dict[str, Any]]:
        """Call the Groq API and parse the structured response."""
        try:
            response = self._groq_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": _USER_PROMPT_TEMPLATE.format(
                        ticket_text=text
                    )},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            print(f"[LLMAgent] Groq call failed: {self._last_error}")
            return None

        parsed = _extract_json(raw_output)
        if parsed is None or not _validate_parsed(parsed):
            print(f"[LLMAgent] Failed to parse valid action from Groq: {raw_output[:200]}")
            return None

        return {
            "action": {
                "category": parsed["category"],
                "priority":  parsed["priority"],
                "route":     parsed["route"],
            },
            "reasoning": parsed.get("reasoning", ""),
            "source":    "llm (groq)",
        }

    # ── Batch inference ───────────────────────────────────────────────────────

    def predict_batch(self, texts: list) -> list:
        """Predict for a list of ticket texts."""
        return [self.predict(t) for t in texts]
