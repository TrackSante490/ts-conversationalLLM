import os
import json
from pathlib import Path
from typing import Any, Dict, List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from datetime import datetime
import re

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
MAX_USER_MESSAGES = int(os.getenv("MAX_USER_MESSAGES", "10"))
TRACKSANTE_COLLECTION = os.getenv("TRACKSANTE_COLLECTION", "tracksante_policy")
_VECTOR_DB_CACHE = None

BASE_DIR = Path(__file__).resolve().parent                  
PROJECT_ROOT = BASE_DIR.parent                              
PERSIST_DIR = (PROJECT_ROOT / f"vectordb/{TRACKSANTE_COLLECTION}").resolve()
STATE_DIR = (PROJECT_ROOT / "rag_state").resolve()
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_DIR = STATE_DIR / "sessions" / SESSION_ID
SESSION_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_PATH = SESSION_DIR / "chat_history.json"
USER_CONTEXT_PATH = SESSION_DIR / "user_context.json"

SYSTEM_RULES = """
You are TrackSante's AI Measurement Coach (pre-screening intake assistant) for a data acquisition device.
You lead the conversation and the user may give short answers.

This conversation serves TWO purposes:
1) To guide the user toward the most relevant TrackSante sensor to measure next.
2) To collect enough structured information to serve as a doctor-facing summary,
   reducing the need for the clinician to repeat this intake.

Device sensors:
- Humidity Sensor: measures humidity
- CO2 Sensor: measures CO2
- Pulse Oxygen Sensor: one sensor that measures BOTH SpO2 and heart rate
- Temperature Sensor: measures temperature

You must run a guided intake:
- Start by asking what concern brought the user in today.
- Ask approximately 5–7 concise, high-yield questions total.
- These questions must collectively be sufficient for:
  (1) a doctor-facing summary,
  (2) identifying potential red flags,
  (3) choosing the most appropriate TrackSante sensor (if applicable).

IMPORTANT FLOW RULE:
- You MAY recommend a sensor before the intake is fully complete.
- However, recommending a sensor does NOT end the intake by itself.
- If key information for the doctor summary is still missing,
  you MUST continue asking the remaining necessary questions.

Conversation end requirement (VERY IMPORTANT):
- By the end of the intake, you MUST end in ONE of the following outcomes:
  Outcome A:
    - recommend exactly ONE TrackSante sensor,
    - complete the intake so the doctor-facing summary is sufficient.
  Outcome B:
    - determine that the concern does not match what the device can measure
      OR requires in-person evaluation,
    - recommend NO_SENSOR_MATCH and advise in-person or urgent medical care.

Safety and scope:
- Do NOT diagnose, treat, or provide medical advice.
- Do NOT recommend medications.
- Do NOT speculate about causes or conditions
  (no “infection”, “allergy”, “inflammation”, “cardiovascular disease”, etc.).
- Use neutral, observational language focused on symptoms and measurements.

Red flag handling:
- If urgent red flags appear, set suggested_next_step="urgent_care"
  and recommended_sensor="NO_SENSOR_MATCH".
- Ask only the most essential clarifying question(s), then direct to urgent care.

Sensor selection guidance (non-diagnostic, measurement-focused):
- Choose Pulse Oxygen Sensor when the main concern involves:
  shortness of breath, chest discomfort, dizziness/lightheadedness,
  palpitations, fainting, or unexplained severe fatigue.
  Measurements: ["spo2","heart_rate"].
- Choose Temperature Sensor when the main concern involves:
  feeling feverish/chills, feeling unusually hot/cold,
  recent illness-like symptoms, or when temperature is the clearest first datapoint.
  Measurements: ["temperature"].
- Choose CO2 Sensor when the main concern involves:
  feeling worse indoors, headaches or drowsiness in a specific room,
  crowded/enclosed spaces, or ventilation/air-quality concerns.
  Measurements: ["co2"].
- Choose Humidity Sensor when the main concern involves:
  dryness or irritation (throat/eyes),
  symptoms worsened by heating or dry indoor air,
  humidifier/dehumidifier concerns.
  Measurements: ["humidity"].
- If multiple sensors could apply, ask 1–2 clarifying questions,
  then choose the single best first sensor.
- If none apply or the concern is outside measurement scope,
  choose NO_SENSOR_MATCH.

Guided intake question policy:
- Ask at most 2 questions per assistant turn.
- Prefer short, specific questions.
- Prioritize these information categories (ask only what’s relevant):
  - symptom description and location
  - onset and duration
  - severity (including functional impact)
  - timing or pattern (constant vs episodic)
  - triggers or context (exertion, indoors/outdoors, time of day)
  - associated symptoms (e.g., shortness of breath, dizziness, fever)
  - relevant history (known conditions, recent illness or changes)
- Continue asking questions until the doctor-facing summary is adequate,
  even if a sensor has already been recommended.

Grounding:
- Use retrieved knowledge base snippets as source of truth for
  sensor rules and question checklists.
- If retrieved context is insufficient, rely on the rules in this prompt
  and ask targeted questions.

Output:
Return ONLY valid JSON with exactly these keys:
{{
  "assistant_message": "string",
  "questions_to_ask": ["string"],
  "collected_facts": {{"string": "any"}},
  "triage_flags": ["string"],
  "suggested_next_step": "continue_intake" | "urgent_care" | "proceed_to_measurements" | "in_person_evaluation",
  "recommended_measurements": ["string"],
  "recommended_sensor": "Humidity Sensor" | "CO2 Sensor" | "Pulse Oxygen Sensor" | "Temperature Sensor" | "NO_SENSOR_MATCH",
  "sensor_confidence": number,
  "sensor_rationale": ["string"],
  "handoff_summary": {{
    "symptoms_summary": "string",
    "relevant_history": "string",
    "lifestyle_factors": "string",
    "recommended_sensor": "string",
    "recommended_measurements": ["string"],
    "open_questions": ["string"]
  }},
  "sources": ["string"]
}}

Rules for fields:
- collected_facts must only include information explicitly stated by the user.
- questions_to_ask must include exactly the next questions you want the user to answer.
- sensor_rationale must be non-diagnostic and based only on user-provided information.
- If recommended_sensor is one of:
  Humidity Sensor / CO2 Sensor / Pulse Oxygen Sensor / Temperature Sensor,
  set suggested_next_step="proceed_to_measurements".
- Recommending a sensor does NOT automatically end the intake.
- If recommended_sensor is NO_SENSOR_MATCH,
  set suggested_next_step to "in_person_evaluation"
  (or "urgent_care" if red flags are present).
- If suggested_next_step is "proceed_to_measurements",
  assistant_message MUST include the sentence:
  "From what you shared in this conversation, I recommend using <SENSOR_NAME>."
- If suggested_next_step is "in_person_evaluation" or "urgent_care",
  assistant_message MUST clearly state that in-person medical evaluation is needed.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human",
     "User message:\n{user_message}\n\n"
     "Conversation window:\n{turn_budget_note}\n\n"
     "Known structured context so far (JSON):\n{user_context}\n\n"
     "Retrieved context snippets:\n{context}\n\n"
     "Return ONLY the JSON object."
    )
])

def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def merge_facts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for k, v in (new or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        merged[k] = v
    return merged
    
#TODO fix this function so that it parses the response for the word "sensor" and if the word sensor is present, it extracts it by reading the word before it, and that gets outputted to the UI. 
# def extract_recommended_sensor(data: Dict[str, Any]) -> str:
#     if "Sensor" in response2.content:
#         return response2.content

def count_user_turns(chat_history_raw: List[Dict[str, Any]]) -> int:
    return sum(1 for m in chat_history_raw if m.get("role") == "user")


def invoke_json_response(
    llm: ChatOllama,
    chat_history: List[Any],
    user_message: str,
    user_context: Dict[str, Any],
    context: str,
    turn_budget_note: str,
) -> (Dict[str, Any], str):
    response = llm.invoke(
        PROMPT.format_messages(
            chat_history=chat_history,
            user_message=user_message,
            user_context=json.dumps(user_context, ensure_ascii=False),
            context=context,
            turn_budget_note=turn_budget_note,
        )
    )
    raw = (response.content or "").strip()

    try:
        data = json.loads(raw)
    except Exception:
        repair_msg = (
            "Your output was not valid JSON. Re-output ONLY valid JSON matching the schema, "
            "with no extra text."
        )
        response2 = llm.invoke(
            PROMPT.format_messages(
                chat_history=chat_history,
                user_message=repair_msg,
                user_context=json.dumps(user_context, ensure_ascii=False),
                context=context,
                turn_budget_note=turn_budget_note,
            )
        )
        raw = (response2.content or "").strip()
        data = json.loads(raw)

    return data, raw


def finalize_if_window_reached(
    llm: ChatOllama,
    chat_history: List[Any],
    user_context: Dict[str, Any],
    context: str,
    turn_budget_note: str,
    current_user_turns: int,
    data: Dict[str, Any],
    raw: str,
) -> (Dict[str, Any], str):
    if current_user_turns < MAX_USER_MESSAGES:
        return data, raw
    if data.get("suggested_next_step") != "continue_intake":
        return data, raw

    force_finalize_message = (
        f"Conversation window reached ({current_user_turns}/{MAX_USER_MESSAGES} user messages). "
        "You must finalize now and ask no more questions. "
        "If enough information is present, recommend exactly one TrackSante sensor and set "
        "suggested_next_step='proceed_to_measurements'. "
        "If information is insufficient or concern is outside measurement scope, set "
        "recommended_sensor='NO_SENSOR_MATCH' and suggested_next_step='in_person_evaluation' "
        "(or 'urgent_care' for red flags)."
    )
    return invoke_json_response(
        llm=llm,
        chat_history=chat_history,
        user_message=force_finalize_message,
        user_context=user_context,
        context=context,
        turn_budget_note=turn_budget_note,
    )


def add_exit_metadata(data: Dict[str, Any], current_user_turns: int) -> Dict[str, Any]:
    result = dict(data)
    next_step = result.get("suggested_next_step", "continue_intake")

    should_exit = (next_step != "continue_intake") or (current_user_turns >= MAX_USER_MESSAGES)

    if next_step == "proceed_to_measurements":
        next_page = "measurements"
        exit_reason = "sensor_recommended"
    elif next_step == "urgent_care":
        next_page = "urgent_care"
        exit_reason = "urgent_care"
    elif next_step == "in_person_evaluation":
        next_page = "in_person_evaluation"
        exit_reason = "in_person_evaluation"
    elif current_user_turns >= MAX_USER_MESSAGES:
        next_page = "summary"
        exit_reason = "max_messages_reached"
    else:
        next_page = None
        exit_reason = None

    result["user_message_count"] = current_user_turns
    result["max_user_messages"] = MAX_USER_MESSAGES
    result["should_exit"] = should_exit
    result["exit_command"] = "exit" if should_exit else None
    result["next_page"] = next_page
    result["exit_reason"] = exit_reason
    return result

def load_vectordb() -> Chroma:
    global _VECTOR_DB_CACHE
    if _VECTOR_DB_CACHE is not None:
        return _VECTOR_DB_CACHE

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    _VECTOR_DB_CACHE = Chroma(
        collection_name=TRACKSANTE_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR)
    )
    return _VECTOR_DB_CACHE


def retrieve_context(vs: Chroma, query: str, k: int = 4) -> (str, List[str]):
    # TODO upgrade to MMR
    docs = vs.similarity_search(query, k=k)

    sources = []
    chunks = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        sources.append(str(src))
        chunks.append(f"[{src}] {d.page_content}")

    context = "\n\n".join(chunks) if chunks else "(no relevant snippets retrieved)"
    return context, sources

def chat_once(
    user_id: str,
    user_message: str,
    session_id: str | None = None,
) -> Dict[str, Any]:
    """
    Single-turn chat call for FastAPI.

    Privacy + behavior:
    - New chat (session_id is None): starts with EMPTY memory (no carryover).
    - Ongoing chat (session_id provided): loads that session's history/context.
    - Storage is per-user and per-session on disk.
    """

    # temporary until we implement log ins, user_id for filesystem safety
    safe_user_id = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id.strip() or "anonymous")

    #decide if this is a new session
    is_new_session = session_id is None
    sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    session_dir = (STATE_DIR / "sessions" / safe_user_id / sid)
    session_dir.mkdir(parents=True, exist_ok=True)

    chat_history_path = session_dir / "chat_history.json"
    user_context_path = session_dir / "user_context.json"

    # load persisted state ONLY if continuing an existing session
    if is_new_session:
        chat_history_raw = []
        user_context = {}
    else:
        chat_history_raw = load_json(chat_history_path, default=[])
        user_context = load_json(user_context_path, default={})

    chat_history: List[Any] = []
    for item in chat_history_raw:
        if item.get("role") == "user":
            chat_history.append(HumanMessage(content=item.get("content", "")))
        else:
            chat_history.append(AIMessage(content=item.get("content", "")))

    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        timeout=90,
    )

    vs = load_vectordb()
    context, sources = retrieve_context(vs, user_message, k=4)

    prior_user_turns = count_user_turns(chat_history_raw)
    current_user_turns = prior_user_turns + 1
    turn_budget_note = (
        f"This is user message {current_user_turns} of maximum {MAX_USER_MESSAGES}. "
        "If at or over the max, finalize now and ask no more questions."
    )

    data, raw = invoke_json_response(
        llm=llm,
        chat_history=chat_history,
        user_message=user_message,
        user_context=user_context,
        context=context,
        turn_budget_note=turn_budget_note,
    )

    data, raw = finalize_if_window_reached(
        llm=llm,
        chat_history=chat_history,
        user_context=user_context,
        context=context,
        turn_budget_note=turn_budget_note,
        current_user_turns=current_user_turns,
        data=data,
        raw=raw,
    )

    #update stored context
    user_context = merge_facts(user_context, data.get("collected_facts", {}))

    # append history + persist
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=raw))

    chat_history_raw_out = []
    for m in chat_history:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        chat_history_raw_out.append({"role": role, "content": m.content})

    save_json(chat_history_path, chat_history_raw_out)
    save_json(user_context_path, user_context)

    if "sources" not in data or not data["sources"]:
        data["sources"] = sources

    data = add_exit_metadata(data, current_user_turns)
    data["session_id"] = sid
    return data


def main():
    #loading persisted state
    chat_history_raw = load_json(CHAT_HISTORY_PATH, default=[])
    user_context = load_json(USER_CONTEXT_PATH, default={})

    # raw history to LangChain messages
    chat_history = []
    for item in chat_history_raw:
        if item["role"] == "user":
            chat_history.append(HumanMessage(content=item["content"]))
        else:
            chat_history.append(AIMessage(content=item["content"]))

    llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

    vs = load_vectordb()

    print("TrackSanté RAG Guidance Agent. Write 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ["exit", "quit"]:
            break

        context, sources = retrieve_context(vs, user_message, k=4)

        prior_user_turns = sum(1 for m in chat_history if isinstance(m, HumanMessage))
        current_user_turns = prior_user_turns + 1
        turn_budget_note = (
            f"This is user message {current_user_turns} of maximum {MAX_USER_MESSAGES}. "
            "If at or over the max, finalize now and ask no more questions."
        )

        data, raw = invoke_json_response(
            llm=llm,
            chat_history=chat_history,
            user_message=user_message,
            user_context=user_context,
            context=context,
            turn_budget_note=turn_budget_note,
        )

        data, raw = finalize_if_window_reached(
            llm=llm,
            chat_history=chat_history,
            user_context=user_context,
            context=context,
            turn_budget_note=turn_budget_note,
            current_user_turns=current_user_turns,
            data=data,
            raw=raw,
        )

        # update stored context
        user_context = merge_facts(user_context, data.get("collected_facts", {}))

        chat_history.append(HumanMessage(content=user_message))
        chat_history.append(AIMessage(content=raw))

        chat_history_raw = []
        for m in chat_history:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            chat_history_raw.append({"role": role, "content": m.content})

        save_json(CHAT_HISTORY_PATH, chat_history_raw)
        save_json(USER_CONTEXT_PATH, user_context)

        # error correction for missing sources if any
        if "sources" not in data or not data["sources"]:
            data["sources"] = sources

        data = add_exit_metadata(data, current_user_turns)

        print("\nAssistant:", data["assistant_message"])
        # for i, q in enumerate(data.get("questions_to_ask", []), 1):
        #     print(f"  {i}. {q}")
        # print()

        if data.get("should_exit"):
            print("\nConversation complete. Type 'exit' to leave this session.")
            break


if __name__ == "__main__":
    main()
