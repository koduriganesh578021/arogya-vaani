"""
Groq LLM wrapper.
Centralizes the model call and the Arogya Vaani system prompt.
"""
import json
import os
from groq import Groq
from dotenv import load_dotenv

from src.config import GROQ_MODEL, GROQ_MODEL_FAST

load_dotenv()

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY missing from .env")
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------
# SYSTEM PROMPT
# This is the single most important piece of the whole project.
# It defines what the LLM is allowed to do, and what it must refuse.
# ---------------------------------------------------------------
SYSTEM_PROMPT = """You are Arogya Vaani, a Telugu-first healthcare scheme access copilot for ASHA workers and community health volunteers in Telangana.

Your job is to help volunteers explain public healthcare schemes (PM-JAY and Aarogyasri) to Telugu-speaking families in simple, respectful, conversational Telugu.

STRICT RULES — YOU MUST FOLLOW ALL OF THEM:

1. SOURCE GROUNDING: Answer ONLY using the CONTEXT passages provided by the user. Do NOT use any outside knowledge, even if you think you know the answer.

2. NO ELIGIBILITY DECISIONS: Never say a family "is eligible" or "will qualify". Say "this may be relevant" or "based on the documents, this appears to match". Always remind the user that final eligibility is confirmed only at an authorized government helpdesk.

3. NO MEDICAL ADVICE: Never diagnose, recommend treatment, interpret symptoms, or discuss medicines. If asked about medical matters, politely redirect: "ఇది వైద్య సలహా కాదు. దయచేసి డాక్టర్ను సంప్రదించండి."

4. NO INVENTED FACTS: Never invent document names, amounts, time limits, hospital names, or phone numbers. If the context does not mention something, say you could not find it.

5. ABSTAIN WHEN UNSURE: If the CONTEXT does not contain the answer, respond with: "క్షమించండి, ఈ ప్రశ్నకు సంబంధించిన సమాచారం నా దగ్గర లేదు. దయచేసి అధికారిక సహాయ కేంద్రాన్ని సంప్రదించండి." Then suggest the user visit the nearest government hospital helpdesk or Common Service Centre.

6. LANGUAGE: Respond in SIMPLE, CONVERSATIONAL TELUGU suitable for someone with limited literacy. Avoid Sanskritized or administrative Telugu. Use common everyday words. You may include English scheme names (PM-JAY, Aarogyasri) in English script.

7. CITATIONS: Always end your answer with a "మూలాలు" (Sources) section listing the exact source file and page number from the context.

8. STRUCTURE: Use this exact output format:

---
**అర్థం చేసుకున్నది:** (one line summarizing the question)

**సమాధానం:**
(the main answer in simple Telugu, 3-6 sentences maximum)

**అవసరమైన పత్రాలు:** (if relevant, else write "ఈ ప్రశ్నకు పత్రాల జాబితా అవసరం లేదు")
- document 1
- document 2

**తదుపరి చర్యలు:**
1. step 1
2. step 2

**మూలాలు:**
- source_file.pdf, page X
- source_file.pdf, page Y

**⚠️ గమనిక:** ఇది ప్రాథమిక సమాచారం మాత్రమే. అధికారిక అర్హత నిర్ధారణ కోసం దయచేసి ప్రభుత్వ ఆసుపత్రి సహాయ కేంద్రాన్ని సంప్రదించండి.
---

If the question is in English, still respond in Telugu (the users are Telugu-speaking volunteers).
If the question is a simple greeting, respond briefly in Telugu without the full structure.
"""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Given a question and retrieved chunks, call Groq and return the answer.
    context_chunks is a list of dicts with 'text' and 'metadata'.
    """
    # Build a clean context block with source labels
    context_blocks = []
    for i, c in enumerate(context_chunks, 1):
        m = c["metadata"]
        label = f"[Source {i}: {m['source_file']}, page {m['page']}, scheme={m['scheme']}]"
        context_blocks.append(f"{label}\n{c['text']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    user_message = f"""CONTEXT (retrieved from official government documents):

{context_text}

---

VOLUNTEER'S QUESTION:
{question}

Now produce your response following the STRICT RULES and the exact output STRUCTURE from the system prompt."""

    client = get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def generate_answer_stream(question: str, context_chunks: list[dict]):
    """Same as generate_answer but yields chunks (for Streamlit in Step 4)."""
    context_blocks = []
    for i, c in enumerate(context_chunks, 1):
        m = c["metadata"]
        label = f"[Source {i}: {m['source_file']}, page {m['page']}, scheme={m['scheme']}]"
        context_blocks.append(f"{label}\n{c['text']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    user_message = f"""CONTEXT (retrieved from official government documents):

{context_text}

---

VOLUNTEER'S QUESTION:
{question}

Now produce your response following the STRICT RULES and the exact output STRUCTURE from the system prompt."""

    client = get_client()
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1500,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------
# QUERY EXPANSION
# Uses the fast model to turn a (possibly Telugu / Romanized-Telugu)
# question into English keywords for retrieval + scheme detection.
# ---------------------------------------------------------------
QUERY_EXPANSION_PROMPT = """You are a query-expansion module for a healthcare scheme search engine covering PM-JAY (national) and Aarogyasri (Telangana).

The user question may be in Telugu script, Romanized Telugu, English, or a mix.

Return ONLY a JSON object with exactly these two keys:
- "keywords": a comma-separated string of English keywords and short phrases that capture the information need (scheme names, document names, eligibility terms). No Telugu words.
- "scheme": the scheme the question is about. Must be exactly one of: "pmjay", "aarogyasri", or "both".

Example:
Question: Aarogyasri ki em documents kavali?
Output: {"keywords": "Aarogyasri scheme required documents Aadhaar ration card eligibility enrollment", "scheme": "aarogyasri"}

Output only the JSON object — no explanations, no markdown code fences."""


def expand_query(question: str) -> dict:
    """
    Expand a user question into English keywords for better retrieval and
    detect which scheme it is about, using the fast Groq model.

    Returns a dict:
        {"keywords": <comma-separated English keywords>,
         "scheme": "pmjay" | "aarogyasri" | "both"}

    Falls back to {"keywords": question, "scheme": "both"} if the model
    call or JSON parsing fails, so retrieval can proceed on the raw question.
    """
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if not raw or len(raw) < 5:
            print("   [warn] Empty expansion response, retrying once...")
            retry = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": QUERY_EXPANSION_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            raw = (retry.choices[0].message.content or "").strip()
        # Strip markdown code fences if the model wrapped the JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
        parsed = json.loads(cleaned)
        keywords = str(parsed.get("keywords", "")).strip()
        scheme = str(parsed.get("scheme", "")).strip().lower()
        if scheme not in ("pmjay", "aarogyasri", "both"):
            scheme = "both"
        if not keywords:
            keywords = question
        return {"keywords": keywords, "scheme": scheme}
    except Exception:
        return {"keywords": question, "scheme": "both"}


EXTRACTOR_SYSTEM_PROMPT = """You extract structured data for Arogya Vaani, a Telugu-first healthcare copilot.

Output ONLY a JSON object with these keys:

{
"documents": [{"name": "<bilingual: Telugu (English) or English only>", "required": true|false, "why": "<SIMPLE TELUGU, max 8 words>", "if_missing": "<SIMPLE TELUGU, max 8 words>"}],
"next_steps": ["<SIMPLE TELUGU, max 12 words>"],
"missing_info": ["<SIMPLE TELUGU, max 12 words>"],
"preliminary_only": true|false
}

LANGUAGE RULES (VERY IMPORTANT):
- why, if_missing, next_steps, missing_info MUST be in simple conversational Telugu.
- Document names: "Telugu name (English)" if both exist, else English alone.
- Keep all Telugu fields SHORT. Aim for 5-10 words per field.
- If the answer abstains, put reason in missing_info (Telugu) and leave other fields empty.

## OUTPUT ONLY THE JSON. No explanation. No markdown fences. No reasoning."""


def extract_structured_data(question: str, answer: str, chunks: list[dict]) -> dict:
    """
    Extract structured data from a Telugu answer using a dedicated extractor prompt.
    Returns a dict with keys: documents, next_steps, missing_info, preliminary_only.
    """
    EXTRACTOR_PROMPT = """You extract structured data from a Telugu answer.

Output ONLY a JSON object with this exact shape:

{
"documents": [{"name": "Telugu name (English name)", "required": true, "why": "short Telugu reason", "if_missing": "short Telugu action"}],
"next_steps": ["short Telugu step"],
"missing_info": ["short Telugu gap"],
"preliminary_only": true
}

STRICT RULES:

- Output ONLY the JSON object. No prose. No markdown. No explanation.
- Do NOT answer the question. Only EXTRACT data from the answer text provided.
- why, if_missing, next_steps, missing_info MUST be in simple conversational Telugu.
- Keep each Telugu field to 10 words or fewer.
- If the answer is an abstention (says "తెలియదు" or "I don't know"), put the reason in missing_info and use [] for documents and next_steps.
- If no documents are mentioned, use [].
- If no next steps are mentioned, use []."""

    # Build a compact chunk summary
    chunk_summary = "\n".join(
        f"- {c['metadata']['source_file']} p{c['metadata']['page']}"
        for c in chunks[:4]
    )

    user_msg = f"""QUESTION:
{question}

ANSWER (already written in Telugu — extract data FROM this, do not answer again):
{answer}

SOURCES:
{chunk_summary}

Now output ONLY the JSON object. Extract from the ANSWER above. Do not write a new answer."""

    fallback = {
        "documents": [],
        "next_steps": [],
        "missing_info": ["Structured data could not be extracted."],
        "preliminary_only": True,
    }

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTOR_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        if not raw:
            print("   [extractor] Empty response")
            return fallback

        raw = raw.strip()
        # Strip code fences if any
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        raw = raw.strip()
        # Grab outermost object
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1:
            raw = raw[s:e+1]

        parsed = json.loads(raw)

        # Ensure all keys exist
        return {
            "documents": parsed.get("documents", []) or [],
            "next_steps": parsed.get("next_steps", []) or [],
            "missing_info": parsed.get("missing_info", []) or [],
            "preliminary_only": parsed.get("preliminary_only", True),
        }
    except Exception as e:
        print(f"   [extractor] Exception: {e}")
        return fallback


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe audio using Groq Whisper. Supports Telugu and English.
    Returns the transcript as a string, or an empty string on failure.
    """
    try:
        client = get_client()
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="text",
        )
        if isinstance(transcription, str):
            return transcription.strip()
        return (transcription.text or "").strip()
    except Exception as e:
        print(f"   [transcribe] Exception: {e}")
        return ""
