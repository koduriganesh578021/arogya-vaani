"""
Groq LLM wrapper.
Centralizes the model call and the Arogya Vaani system prompt.
"""
import os
from groq import Groq
from dotenv import load_dotenv

from src.config import GROQ_MODEL

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
