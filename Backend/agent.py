"""
LangGraph-based agentic visa assistant.

The agent uses the LLM brain to:
  1. Extract travel context (destination, source country, purpose,
     duration, budget, …) from the conversation – no hardcoded alias
     dictionaries.
  2. Map the extracted destination to the correct data-folder name
     (USA / Singapore / Qatar) using the LLM.
  3. Build a keyword-rich RAG query from ALL remembered context and
     retrieve 10 chunks.
  4. Generate a grounded, conversational answer.
  5. Ask follow-up questions when important context is still missing.
  6. Answer off-topic questions too, then steer back to visa info.

Graph:
  START → extract_info ─┬─ (dest known) → retrieve → respond → END
                        └─ (no dest)   → respond  ────────→ END
"""

import json
import logging
import re
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_API_KEY, LLM_MODEL, SUPPORTED_COUNTRIES, RETRIEVAL_TOP_K
from rag import RAGPipeline

# ── Regex-based country detection (fallback for LLM failures) ──────

_COUNTRY_ALIASES: dict[str, list[str]] = {
    "USA": [
        r"\busa\b", r"\bu\.?s\.?a?\.?\b", r"\bunited\s+states\b",
        r"\bamerica\b", r"\bthe\s+states\b", r"\bus\b",
    ],
    "Singapore": [
        r"\bsingapore\b", r"\bsingapura\b", r"\b(?:sg)\b",
    ],
    "Qatar": [
        r"\bqatar\b", r"\bkatar\b", r"\bdoha\b",
    ],
}


def _regex_detect_country(text: str) -> Optional[str]:
    """Return a supported country name if one is mentioned in *text*."""
    text_lower = text.lower()
    for country, patterns in _COUNTRY_ALIASES.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return country
    return None

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list                          # full conversation history
    user_message: str                       # current user input
    destination_country: Optional[str]      # resolved folder name or None
    source_country: Optional[str]
    travel_purpose: Optional[str]
    duration: Optional[str]
    budget: Optional[str]
    extra_keywords: Optional[str]           # any other useful keywords
    rag_context: str
    response: str


# ── Agent ──────────────────────────────────────────────────────────

class VisaAssistantAgent:

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            max_output_tokens=2048,
        )
        self.rag = rag_pipeline
        self.graph = self._build_graph()
        self.sessions: dict[str, dict] = {}

    # ── graph ──────────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("extract_info", self._extract_info)
        g.add_node("retrieve", self._retrieve)
        g.add_node("respond", self._respond)

        g.add_edge(START, "extract_info")
        g.add_conditional_edges(
            "extract_info", self._route,
            {"retrieve": "retrieve", "respond": "respond"},
        )
        g.add_edge("retrieve", "respond")
        g.add_edge("respond", END)
        return g.compile()

    # ── node: extract_info ─────────────────────────────────────────

    def _extract_info(self, state: AgentState) -> dict:
        msgs = state["messages"]
        user_msg = state["user_message"]

        # Carry forward what we already know
        cur = {
            "destination_country": state.get("destination_country"),
            "source_country": state.get("source_country"),
            "travel_purpose": state.get("travel_purpose"),
            "duration": state.get("duration"),
            "budget": state.get("budget"),
            "extra_keywords": state.get("extra_keywords"),
        }

        history = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in msgs[-12:]
        )

        supported = ", ".join(SUPPORTED_COUNTRIES)

        prompt = f"""You are a precise information-extraction assistant.

TASK: Read the conversation and the latest user message. Extract travel
details the user has stated or clearly implied. Do NOT guess or invent
information that was not mentioned.

We have visa data for these destination countries: {supported}.
The folder names are exactly: USA, Singapore, Qatar.

CRITICAL - DIRECTIONAL LANGUAGE PATTERNS:
- "travel FROM X TO Y" = source: X, destination: Y
- "going TO X FROM Y" = source: Y, destination: X  
- "X to Y travel" = source: X, destination: Y
- "visa FOR X" = destination: X
- "visiting X" = destination: X  
- "I am FROM X" = source: X
- "I want to go TO X" = destination: X
- "I am travelling TO X" = destination: X
- "travelling TO X" = destination: X
- "travel TO X" = destination: X
- "I am going TO X" = destination: X

If the user says "America", "US", "U.S.", "United States", "the states",
etc., the destination_country should be "USA".
If they say "SG", "Singapura", etc., it should be "Singapore".
If they say "Katar", "Doha", etc., it should be "Qatar".
If they mention a country we do NOT have data for, set destination_country
to that country name anyway (we will tell the user we don't support it).

CONVERSATION:
{history}

LATEST MESSAGE: {user_msg}

ALREADY KNOWN:
  destination_country = {cur['destination_country'] or 'unknown'}
  source_country      = {cur['source_country'] or 'unknown'}
  travel_purpose      = {cur['travel_purpose'] or 'unknown'}
  duration            = {cur['duration'] or 'unknown'}
  budget              = {cur['budget'] or 'unknown'}
  extra_keywords      = {cur['extra_keywords'] or 'unknown'}

IMPORTANT: Pay close attention to directional prepositions (FROM/TO). The destination
is where the person wants to TRAVEL TO, not where they are coming FROM.

Return ONLY a JSON object (no markdown fences) with these keys:
  "destination_country" – one of {supported} or the raw name or null
  "source_country"      – string or null
  "travel_purpose"      – string or null
  "duration"            – string or null
  "budget"              – string or null
  "extra_keywords"      – comma-separated useful keywords or null

If a field was already known and the user did not change it, keep it.
"""

        extracted: dict = {}
        try:
            result = self.llm.invoke(prompt)
            raw = result.content.strip()
            logger.info("🧠 LLM extraction raw output: %s", raw[:500])
            if "```" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip() if "```json" in raw else raw.split("```")[1].split("```")[0].strip()
            extracted = json.loads(raw)
            logger.info("📋 Parsed extraction: %s", extracted)
        except Exception as exc:
            logger.warning("Extraction parse failed: %s", exc)

        # Merge: new values override only when non-null
        def pick(key):
            new_val = extracted.get(key)
            if new_val and str(new_val).lower() not in ("null", "none", "unknown", ""):
                return str(new_val)
            return cur.get(key)

        # Filter out generic/vague travel_purpose values that aren't real purposes
        _INVALID_PURPOSES = {
            "traveling", "travelling", "travel", "trip", "visit",
            "going", "moving", "flying", "visa", "general",
        }
        raw_purpose = pick("travel_purpose")
        if raw_purpose and raw_purpose.strip().lower() in _INVALID_PURPOSES:
            raw_purpose = cur.get("travel_purpose")  # revert to previous value

        dest = pick("destination_country")
        # Normalise dest to exact folder name if it is one of our supported ones
        if dest:
            for sc in SUPPORTED_COUNTRIES:
                if dest.lower() == sc.lower():
                    dest = sc
                    break

        # ── FALLBACK: regex-based detection when LLM missed the country ──
        if not dest:
            regex_dest = _regex_detect_country(user_msg)
            if regex_dest:
                logger.info("🔧 Regex fallback detected destination: %s (LLM missed it)", regex_dest)
                dest = regex_dest
            else:
                # Also scan recent conversation history for a country mention
                for m in reversed(msgs[-6:]):
                    regex_dest = _regex_detect_country(m.get("content", ""))
                    if regex_dest:
                        logger.info("🔧 Regex fallback detected destination from history: %s", regex_dest)
                        dest = regex_dest
                        break

        logger.info("✅ Final extraction → dest=%s, source=%s, purpose=%s",
                     dest, pick("source_country"), raw_purpose)

        return {
            "destination_country": dest,
            "source_country": pick("source_country"),
            "travel_purpose": raw_purpose,
            "duration": pick("duration"),
            "budget": pick("budget"),
            "extra_keywords": pick("extra_keywords"),
        }

    # ── routing ────────────────────────────────────────────────────

    def _route(self, state: AgentState) -> str:
        dest = state.get("destination_country")
        if dest:  # Use RAG if ANY destination is present (even unsupported)
            return "retrieve"
        return "respond"  # No destination → ask for one

    # ── node: retrieve ─────────────────────────────────────────────

    def _retrieve(self, state: AgentState) -> dict:
        country = state["destination_country"]
        user_msg = state["user_message"]

        # Build a rich query from ALL known context
        parts = [user_msg]
        if state.get("source_country"):
            parts.append(f"traveling from {state['source_country']}")
        if state.get("travel_purpose"):
            parts.append(f"purpose: {state['travel_purpose']}")
        if state.get("duration"):
            parts.append(f"duration: {state['duration']}")
        if state.get("budget"):
            parts.append(f"budget: {state['budget']}")
        if state.get("extra_keywords"):
            parts.append(state["extra_keywords"])
        query = " ".join(parts)

        # Try to retrieve from RAG (only works for supported countries)
        if country in SUPPORTED_COUNTRIES:
            logger.info("🔍 RAG Query for %s: %s", country, query)
            docs = self.rag.retrieve(country, query, k=RETRIEVAL_TOP_K)
            if docs:
                logger.info("📄 Retrieved %d chunks from %s", len(docs), country)
                
                # Debug: Print retrieved chunks to terminal
                for i, doc in enumerate(docs, 1):
                    title = doc.metadata.get("title", "Unknown")
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    logger.info("   [%d] %s", i, title)
                    logger.info("       %s", content_preview.replace('\n', ' '))
                
                chunks = []
                for i, doc in enumerate(docs, 1):
                    title = doc.metadata.get("title", "")
                    chunks.append(f"[Source {i}] {title}\n{doc.page_content}")
                return {"rag_context": "\n\n---\n\n".join(chunks)}
            else:
                logger.info("❌ No documents found for %s with query: %s", country, query)
        else:
            logger.info("⚠️  Country %s not supported (available: %s)", country, ", ".join(SUPPORTED_COUNTRIES))
        
        # Destination not supported or no docs found
        return {"rag_context": "No relevant information found in the database."}

    # ── node: respond ──────────────────────────────────────────────

    def _respond(self, state: AgentState) -> dict:
        msgs = state["messages"]
        user_msg = state["user_message"]
        dest = state.get("destination_country") or ""
        source = state.get("source_country") or ""
        purpose = state.get("travel_purpose") or ""
        duration = state.get("duration") or ""
        budget = state.get("budget") or ""
        extra = state.get("extra_keywords") or ""
        rag_ctx = state.get("rag_context") or ""

        history = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in msgs[-10:]
        )

        supported = ", ".join(SUPPORTED_COUNTRIES)

        # ── Build a FOCUSED prompt based on what we know ──────────
        # Use completely different prompts depending on state to avoid
        # the small model getting confused by competing instructions.

        if dest and rag_ctx and rag_ctx != "No relevant information found in the database.":
            # ===== DESTINATION IS KNOWN — give visa info, ask next question =====
            prompt = self._build_dest_known_prompt(
                dest, source, purpose, duration, budget, extra,
                rag_ctx, history, user_msg, supported,
            )
        elif dest and dest not in SUPPORTED_COUNTRIES:
            # ===== UNSUPPORTED DESTINATION =====
            prompt = (
                f"You are a friendly visa assistant. The user wants to travel "
                f"to \"{dest}\" but you only have visa data for: {supported}.\n\n"
                f"CONVERSATION:\n{history}\nUSER: {user_msg}\n\n"
                f"Politely tell the user you don't have data for {dest}. "
                f"Ask if they meant one of the supported countries or if "
                f"they'd like help with one of those instead.\n"
                f"Always end with 'What else can I help you with?' or similar.\n"
                f"Respond now:"
            )
        else:
            # ===== NO DESTINATION YET — ask for it =====
            prompt = (
                f"You are a friendly visa assistant chatbot with visa data "
                f"for: {supported}.\n\n"
                f"CONVERSATION:\n{history}\nUSER: {user_msg}\n\n"
                f"The user has not told you their destination country yet. "
                f"Answer their message helpfully, then ask which country "
                f"they want to travel to (you support {supported}).\n"
                f"Be friendly and conversational.\n"
                f"Always end with 'What else can I help you with?' or similar.\n"
                f"Respond now:"
            )

        logger.info("📝 Respond prompt mode: dest=%s, source=%s, purpose=%s",
                     dest or "NONE", source or "NONE", purpose or "NONE")

        try:
            result = self.llm.invoke(prompt)
            return {"response": result.content}
        except Exception as exc:
            logger.exception("Response generation failed: %s", exc)
            return {
                "response": (
                    "I\'m sorry, I encountered an error while generating a "
                    "response. Please try again."
                )
            }

    # ── prompt builder when destination IS known ───────────────────

    def _build_dest_known_prompt(
        self, dest, source, purpose, duration, budget, extra,
        rag_ctx, history, user_msg, supported,
    ) -> str:
        """Build a focused prompt that strongly grounds answers in RAG data."""

        context_summary = f"Destination: {dest}"
        if source:
            context_summary += f", Source country: {source}"
        if purpose:
            context_summary += f", Purpose: {purpose}"
        if duration:
            context_summary += f", Duration: {duration}"
        if budget:
            context_summary += f", Budget: {budget}"

        # Decide what to ask next
        if not source:
            next_action = (
                f"MANDATORY NEXT QUESTION: Ask the user which country they are "
                f"travelling FROM / their nationality. This determines visa "
                f"waiver eligibility and specific requirements. "
                f"DO NOT ask for the destination — you already know it is {dest}."
            )
        elif not purpose:
            next_action = (
                f"NEXT QUESTION: Ask the user about their travel purpose "
                f"(tourism, business, work, study, etc.). "
                f"We already know: {source} → {dest}. Do NOT ask for destination or source again."
            )
        elif not duration:
            next_action = (
                f"NEXT QUESTION: Ask about their planned duration of stay. "
                f"We already know: {source} → {dest} for {purpose}. "
                f"Do NOT repeat questions about destination, source, or purpose."
            )
        elif not budget:
            next_action = (
                f"NEXT QUESTION: Ask about their approximate budget for the trip. "
                f"We already know: {source} → {dest} for {purpose}, duration: {duration}. "
                f"Do NOT repeat questions about destination, source, purpose, or duration."
            )
        else:
            next_action = (
                f"CONVERSATIONAL MODE: We have good context ({context_summary}). "
                f"Give comprehensive advice. "
                f"Do NOT ask for information you already have."
            )

        prompt = f"""You are a friendly, knowledgeable visa assistant chatbot.
You have visa data for: {supported}.

TRAVEL CONTEXT (what we already know about this user):
  {context_summary}

─── RETRIEVED VISA INFORMATION (ground your answer ONLY in this data) ───
{rag_ctx}
─── END OF RETRIEVED VISA INFORMATION ───

CONVERSATION HISTORY:
{history}

USER MESSAGE: {user_msg}

{next_action}

CRITICAL INSTRUCTIONS:
1. READ the retrieved visa information above CAREFULLY. Your answer MUST be
   based on and cite specifics from this data. Look for visa waiver programs,
   visa-free entry, specific visa categories, fees, validity periods, required
   documents, and processing times mentioned in the sources.
2. If the retrieved data says a country's citizens do NOT need a visa or are
   eligible for visa-free entry or visa waiver, YOU MUST state that clearly.
3. If the retrieved data mentions specific visa types (B-1, B-2, work permit,
   employment pass, etc.), explain them with details from the sources.
4. NEVER invent visa rules, fees, or requirements not present in the data.
5. NEVER ask for the destination country — it is {dest}.
6. Be thorough and detailed: mention visa types, requirements, fees,
   processing times, and any special programs from the retrieved data.
7. ALWAYS end your response with a helpful closing like "What else can I
   help you with?" or "Is there anything else you'd like to know?"
Respond now:"""
        return prompt

    # ── public chat interface ──────────────────────────────────────

    def get_or_create_session(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],
                "destination_country": None,
                "source_country": None,
                "travel_purpose": None,
                "duration": None,
                "budget": None,
                "extra_keywords": None,
            }
        return self.sessions[session_id]

    def chat(self, session_id: str, user_message: str) -> str:
        session = self.get_or_create_session(session_id)

        initial: AgentState = {
            "messages": session["messages"].copy(),
            "user_message": user_message,
            "destination_country": session["destination_country"],
            "source_country": session["source_country"],
            "travel_purpose": session["travel_purpose"],
            "duration": session["duration"],
            "budget": session["budget"],
            "extra_keywords": session["extra_keywords"],
            "rag_context": "",
            "response": "",
        }

        result = self.graph.invoke(initial)

        # Persist
        session["messages"].append({"role": "user", "content": user_message})
        session["messages"].append({"role": "assistant", "content": result["response"]})

        for key in ("destination_country", "source_country", "travel_purpose",
                     "duration", "budget", "extra_keywords"):
            session[key] = result.get(key) or session[key]

        # Keep history bounded
        if len(session["messages"]) > 40:
            session["messages"] = session["messages"][-40:]

        return result["response"]
