"""
Arogya Vaani — Streamlit UI
Run:  streamlit run src/app.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.retrieval import Retriever
from src.llm import generate_answer_stream, expand_query, extract_structured_data, transcribe_audio
from src.config import TOP_K_FINAL
from src.rules import check_pmjay, check_aarogyasri, summarize

# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Arogya Vaani",
    page_icon="🏥",
    layout="wide",
)

# -------------------------------------------------------------------
# CUSTOM CSS — Telugu font optimization
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans Telugu', 'Segoe UI', sans-serif;
    }
    .telugu-text {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.05rem;
        line-height: 1.8;
    }
    .source-box {
        background-color: #f0f4f8;
        border-left: 4px solid #2e7d32;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .disclaimer {
        background-color: #fff8e1;
        border-left: 4px solid #f57c00;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# CACHE THE RETRIEVER
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Arogya Vaani engine...")
def load_retriever():
    return Retriever()

STATUS_ICONS = {
    "met": "🟢",
    "not_met": "🔴",
    "unclear": "🟡",
}

STATUS_LABELS = {
    "met": "Present",
    "not_met": "Missing",
    "unclear": "Unclear",
}


def _render_precheck_results(rules, scheme):
    st.markdown("---")
    st.markdown(f"#### Preliminary Indicators — {scheme}")

    for r in rules:
        icon = STATUS_ICONS.get(r["status"], "⚪")
        label = STATUS_LABELS.get(r["status"], r["status"])
        with st.container():
            st.markdown(f"**{icon} {r['factor']}** — *{label}*")
            st.caption(r["reason"])

    summary = summarize(rules)
    counts = summary["counts"]

    st.markdown("---")
    st.markdown(
        f"**Summary:** {counts['met']} present · "
        f"{counts['not_met']} missing · {counts['unclear']} unclear"
    )

    if summary["summary"] == "preliminary_missing":
        st.warning(summary["summary_text"])
    elif summary["summary"] == "preliminary_positive":
        st.success(summary["summary_text"])
    else:
        st.info(summary["summary_text"])

    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <b>This is NOT an official eligibility decision.</b> "
        "Visit the nearest government hospital helpdesk, Aarogyamithra "
        "counter, or Common Service Centre for official verification."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_structured_cards(structured: dict):
    """Render documents, next_steps, and missing_info as visual cards."""
    docs = structured.get("documents") or []
    steps = structured.get("next_steps") or []
    missing = structured.get("missing_info") or []

    # Documents card
    if docs:
        st.markdown("### 📄 అవసరమైన పత్రాలు")
        for d in docs:
            name = d.get("name", "—")
            required = d.get("required", False)
            why = d.get("why", "")
            if_missing = d.get("if_missing", "")
            tag_color = "#d32f2f" if required else "#f57c00"
            tag_text = "అవసరం" if required else "షరతులతో"
            st.markdown(
                f"<div style='border-left:4px solid {tag_color};"
                f"padding:8px 12px;margin:6px 0;background:#fafafa;border-radius:4px'>"
                f"<b>{name}</b> "
                f"<span style='background:{tag_color};color:white;padding:1px 8px;"
                f"border-radius:10px;font-size:0.75rem;margin-left:6px'>{tag_text}</span>"
                f"<br><small><b>Why:</b> {why}</small>"
                f"<br><small><b>If missing:</b> {if_missing}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Next steps card
    if steps:
        st.markdown("### ➡️ తదుపరి చర్యలు")
        steps_html = ""
        for i, s in enumerate(steps, 1):
            steps_html += (
                f"<div style='display:flex;align-items:flex-start;margin:8px 0'>"
                f"<div style='background:#2e7d32;color:white;min-width:26px;height:26px;"
                f"border-radius:50%;display:flex;align-items:center;justify-content:center;"
                f"font-weight:bold;margin-right:10px'>{i}</div>"
                f"<div style='padding-top:2px'>{s}</div>"
                f"</div>"
            )
        st.markdown(
            f"<div style='background:#f5f5f5;padding:12px;border-radius:6px'>{steps_html}</div>",
            unsafe_allow_html=True,
        )

    # Missing info card
    if missing:
        st.markdown("### ⚠️ అస్పష్టమైన సమాచారం")
        for m in missing:
            st.markdown(
                f"<div style='border-left:4px solid #f57c00;padding:8px 12px;"
                f"background:#fff8e1;border-radius:4px;margin:6px 0'>{m}</div>",
                unsafe_allow_html=True,
            )


def _build_download_text(question: str, answer: str, structured: dict, chunks: list) -> str:
    """Build a plain-text version of the answer for download."""
    lines = []
    lines.append("=" * 60)
    lines.append("ఆరోగ్య వాణి (Arogya Vaani) — సమాధానం")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"ప్రశ్న (Question): {question}")
    lines.append("")
    lines.append("సమాధానం (Answer):")
    lines.append(answer.strip())
    lines.append("")

    docs = structured.get("documents") or []
    if docs:
        lines.append("-" * 60)
        lines.append("అవసరమైన పత్రాలు (Required documents):")
        for d in docs:
            req = "అవసరం" if d.get("required") else "షరతులతో"
            lines.append(f"  • {d.get('name', '—')} [{req}]")
            if d.get("why"):
                lines.append(f"      ఎందుకు: {d['why']}")
            if d.get("if_missing"):
                lines.append(f"      లేకపోతే: {d['if_missing']}")
        lines.append("")

    steps = structured.get("next_steps") or []
    if steps:
        lines.append("-" * 60)
        lines.append("తదుపరి చర్యలు (Next steps):")
        for i, s in enumerate(steps, 1):
            lines.append(f"  {i}. {s}")
        lines.append("")

    missing = structured.get("missing_info") or []
    if missing:
        lines.append("-" * 60)
        lines.append("అస్పష్టమైన సమాచారం (Missing information):")
        for m in missing:
            lines.append(f"  ⚠ {m}")
        lines.append("")

    if chunks:
        lines.append("-" * 60)
        lines.append("మూలాలు (Sources):")
        seen = set()
        for c in chunks:
            key = (c["metadata"]["source_file"], c["metadata"]["page"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  • {key[0]}, page {key[1]}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("⚠️ ఇది ప్రాథమిక సమాచారం మాత్రమే.")
    lines.append("అధికారిక అర్హత నిర్ధారణ కోసం దయచేసి ప్రభుత్వ ఆసుపత్రి")
    lines.append("సహాయ కేంద్రాన్ని సంప్రదించండి.")
    lines.append("")
    lines.append("This tool does not provide medical advice or")
    lines.append("official eligibility decisions.")
    lines.append("=" * 60)

    return "\n".join(lines)


def _render_audio_player(text: str, key_suffix: str = ""):
    """Render a Listen button that speaks the given text via browser TTS."""
    import streamlit.components.v1 as components

    # Escape backticks and backslashes so the text is safe inside JS template literals
    safe_text = (
        text.replace("\\", "\\\\")
            .replace("`", "\\`")
            .replace("${", "\\${")
    )

    html = f"""
<div style="display:flex;align-items:center;gap:10px;margin-top:8px">
  <button id="speak-btn-{key_suffix}" style="
    background:#1976d2;color:white;border:none;padding:8px 16px;
    border-radius:6px;cursor:pointer;font-size:15px;
  ">🔊 వినండి (Listen)</button>
  <button id="stop-btn-{key_suffix}" style="
    background:#c62828;color:white;border:none;padding:8px 16px;
    border-radius:6px;cursor:pointer;font-size:15px;display:none;
  ">⏹ ఆపండి (Stop)</button>
  <span id="status-{key_suffix}" style="font-size:13px;color:#555"></span>
</div>

<script>
(function() {{
  const text = `{safe_text}`;
  const speakBtn = document.getElementById('speak-btn-{key_suffix}');
  const stopBtn = document.getElementById('stop-btn-{key_suffix}');
  const status = document.getElementById('status-{key_suffix}');
  const synth = window.speechSynthesis;

  function pickVoice() {{
    const voices = synth.getVoices();
    // Preference order: Telugu, Hindi, Indian English, English
    return voices.find(v => v.lang === 'te-IN')
        || voices.find(v => v.lang.startsWith('te'))
        || voices.find(v => v.lang === 'hi-IN')
        || voices.find(v => v.lang.startsWith('hi'))
        || voices.find(v => v.lang === 'en-IN')
        || voices.find(v => v.lang.startsWith('en'))
        || voices[0];
  }}

  speakBtn.addEventListener('click', () => {{
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const v = pickVoice();
    if (v) {{ u.voice = v; u.lang = v.lang; }}
    u.rate = 0.95;
    u.pitch = 1.0;
    u.onstart = () => {{
      speakBtn.style.display = 'none';
      stopBtn.style.display = 'inline-block';
      status.textContent = 'వింటున్నారు... (' + (u.lang || 'default') + ')';
    }};
    u.onend = () => {{
      speakBtn.style.display = 'inline-block';
      stopBtn.style.display = 'none';
      status.textContent = '';
    }};
    u.onerror = (e) => {{
      speakBtn.style.display = 'inline-block';
      stopBtn.style.display = 'none';
      status.textContent = 'Audio error: ' + e.error;
    }};
    synth.speak(u);
  }});

  stopBtn.addEventListener('click', () => {{
    synth.cancel();
    speakBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
    status.textContent = '';
  }});

  // Voices load asynchronously in some browsers
  if (speechSynthesis.onvoiceschanged !== undefined) {{
    speechSynthesis.onvoiceschanged = () => {{}};
  }}
}})();
</script>
"""

    components.html(html, height=70)

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏥 Arogya Vaani")
    st.caption("Telugu-first healthcare scheme copilot")
    st.markdown("---")

    st.markdown("### 📋 Scheme Filter")
    scheme_choice = st.radio(
        "Which scheme is the family asking about?",
        ["Both", "PM-JAY", "Aarogyasri"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎯 Quick Questions")
    quick_questions = [
        "Aarogyasri ki em documents kavali?",
        "PM-JAY eligibility emiti?",
        "Aarogyasri card ela apply cheyali?",
        "PM-JAY lo em treatments cover avutayi?",
        "Hospital ki velladaniki em theesukellali?",
    ]
    for q in quick_questions:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_sources = st.checkbox("Show retrieved sources", value=True)
    show_debug = st.checkbox("Show debug info", value=False)

    st.markdown("---")
    st.caption("Built for the Unleash LLM Innovation Challenge 2026")
    st.caption("Team: HealthWhisperers")

# -------------------------------------------------------------------
# MAIN AREA
# -------------------------------------------------------------------
st.title("🏥 Arogya Vaani")
st.markdown(
    "<p class='telugu-text'>"
    "తెలంగాణలోని కుటుంబాలకు PM-JAY మరియు ఆరోగ్యశ్రీ పథకాల గురించి "
    "సరళమైన తెలుగులో సమాచారం అందించే సహాయకుడు."
    "</p>",
    unsafe_allow_html=True,
)
st.caption("A Telugu-first healthcare scheme access copilot for ASHA workers and community volunteers.")

tab_chat, tab_precheck = st.tabs(["💬 Ask a question", "📋 Preliminary eligibility check"])

with tab_chat:
    # -------------------------------------------------------------------
    # CHAT HISTORY
    # -------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "_last_audio_id" not in st.session_state:
        st.session_state["_last_audio_id"] = None

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg["role"] == "assistant" and msg.get("structured"):
                _render_structured_cards(msg["structured"])
            if msg["role"] == "assistant" and msg.get("content"):
                _render_audio_player(
                    msg["content"],
                    key_suffix=f"hist_{abs(hash(msg['content'])) % 100000}",
                )
                if msg.get("question"):
                    from datetime import datetime
                    ts = msg.get("timestamp", datetime.now().strftime("%Y%m%d-%H%M"))
                    download_text = _build_download_text(
                        msg["question"],
                        msg["content"],
                        msg.get("structured", {}),
                        msg.get("sources", []),
                    )
                    st.download_button(
                        label="📥 Download explanation",
                        data=download_text.encode("utf-8"),
                        file_name=f"arogya-vaani-{ts}.txt",
                        mime="text/plain",
                        key=f"hist_download_{ts}_{id(msg)}",
                    )
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} chunks retrieved)", expanded=False):
                    for i, s in enumerate(msg["sources"], 1):
                        m = s["metadata"]
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<b>[{i}] {m['source_file']}</b> — page {m['page']}<br>"
                            f"<small>Scheme: <b>{m['scheme']}</b> · Type: {m['doc_type']}</small><br>"
                            f"<small>{s['text'][:300]}...</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # -------------------------------------------------------------------
    # INPUT — chat_input OR sidebar quick question
    # -------------------------------------------------------------------
    # --- Voice input (mic) ---
    st.markdown("##### 🎙️ లేదా మాట్లాడండి (Or speak)")
    audio_value = st.audio_input(
        "Record your question",
        key="voice_input_widget",
        label_visibility="collapsed",
    )

    # --- Text input ---
    user_question = st.chat_input("Ask about PM-JAY or Aarogyasri (Telugu or English)…")

    # --- If a quick question was clicked, use that ---
    if st.session_state.pending_question:
        user_question = st.session_state.pending_question
        st.session_state.pending_question = None

    # --- If audio was recorded, transcribe it ---
    if audio_value is not None and not st.session_state.get("_last_audio_id") == id(audio_value):
        with st.spinner("🎙️ Transcribing your voice..."):
            try:
                audio_bytes = audio_value.read()
                transcript = transcribe_audio(audio_bytes, filename="question.wav")
                if transcript:
                    st.session_state["_last_audio_id"] = id(audio_value)
                    st.success(f"🎙️ **Transcribed:** {transcript}")
                    user_question = transcript
                else:
                    st.warning("Could not transcribe the audio. Please try again or type your question.")
            except Exception as e:
                st.error(f"Audio error: {e}")

    # -------------------------------------------------------------------
    # PROCESS THE QUESTION
    # -------------------------------------------------------------------
    if user_question:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Retrieve
        with st.chat_message("assistant"):
            retriever = load_retriever()

            with st.spinner("🔍 Expanding query and retrieving relevant scheme information..."):
                expanded = expand_query(user_question)
                combined_query = f"{user_question} {expanded['keywords']}"
                chunks = retriever.search(
                    combined_query,
                    k=TOP_K_FINAL + 1,
                    scheme=expanded["scheme"],
                )

            if show_debug:
                st.caption(f"Expanded keywords: {expanded['keywords']}")
                st.caption(f"Detected scheme: {expanded['scheme']}")

            # Filter by scheme if user chose one
            if scheme_choice != "Both":
                scheme_map = {"PM-JAY": "pmjay", "Aarogyasri": "aarogyasri"}
                target = scheme_map[scheme_choice]
                filtered = [c for c in chunks if c["metadata"]["scheme"] == target]
                if filtered:
                    chunks = filtered

            if show_debug:
                st.caption(f"Retrieved {len(chunks)} chunks: " +
                           ", ".join(f"{c['metadata']['source_file']} p{c['metadata']['page']}"
                                     for c in chunks))

            # Stream the answer
            st.markdown("### 📝 సమాధానం")
            placeholder = st.empty()
            full_answer = ""

            try:
                for token in generate_answer_stream(user_question, chunks):
                    full_answer += token
                    placeholder.markdown(full_answer)
            except Exception as e:
                st.error(f"Error calling LLM: {e}")
                full_answer = "క్షమించండి, సాంకేతిక సమస్య ఏర్పడింది. దయచేసి మళ్లీ ప్రయత్నించండి."

            # Extract structured cards
            structured = {"documents": [], "next_steps": [], "missing_info": [], "preliminary_only": True}
            if full_answer and len(full_answer) > 50 and "క్షమించండి" not in full_answer:
                try:
                    structured = extract_structured_data(user_question, full_answer, chunks)
                except Exception as e:
                    st.caption(f"(Structured extraction skipped: {e})")

            _render_structured_cards(structured)

            _render_audio_player(full_answer, key_suffix=f"live_{abs(hash(user_question)) % 100000}")

            # Download button
            download_text = _build_download_text(user_question, full_answer, structured, chunks)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            st.download_button(
                label="📥 సమాధానాన్ని డౌన్‌లోడ్ చేయండి (Download explanation)",
                data=download_text.encode("utf-8"),
                file_name=f"arogya-vaani-{timestamp}.txt",
                mime="text/plain",
                key=f"download_{timestamp}",
            )

            # Save to history
            from datetime import datetime
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": chunks,
                "structured": structured,
                "question": user_question,
                "timestamp": datetime.now().strftime("%Y%m%d-%H%M"),
            })

            # Show sources inline after the answer (for the current turn)
            if show_sources:
                with st.expander(f"📚 Sources ({len(chunks)} chunks retrieved)", expanded=False):
                    for i, s in enumerate(chunks, 1):
                        m = s["metadata"]
                        st.markdown(
                            f"<div class='source-box'>"
                            f"<b>[{i}] {m['source_file']}</b> — page {m['page']}<br>"
                            f"<small>Scheme: <b>{m['scheme']}</b> · Type: {m['doc_type']}</small><br>"
                            f"<small>{s['text'][:300]}...</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

with tab_precheck:
    st.markdown("### 📋 Preliminary Eligibility Pre-Check")
    st.caption(
        "This is a deterministic rule-based check. No AI is involved. "
        "Results are INDICATIVE ONLY and do not constitute an official "
        "eligibility decision."
    )

    scheme_for_check = st.radio(
        "Select the scheme to check:",
        ["PM-JAY", "Aarogyasri"],
        key="precheck_scheme",
        horizontal=True,
    )

    with st.form("precheck_form"):
        if scheme_for_check == "PM-JAY":
            st.markdown("**Tell us about the family:**")
            has_aadhaar = st.checkbox("Family has an Aadhaar card")
            has_ration_card = st.checkbox("Family has a Ration Card")
            has_secc_listing = st.checkbox(
                "Family is listed in SECC 2011 / has a known SECC ID"
            )
            is_bpl_family = st.checkbox("Family is Below Poverty Line (BPL)")
            family_size = st.number_input(
                "Family size (number of members)",
                min_value=0, max_value=30, value=0, step=1,
            )
            submitted = st.form_submit_button("Run preliminary check")

            if submitted:
                rules = check_pmjay(
                    has_aadhaar=has_aadhaar,
                    has_ration_card=has_ration_card,
                    has_secc_listing=has_secc_listing,
                    is_bpl_family=is_bpl_family,
                    family_size=int(family_size),
                )
                _render_precheck_results(rules, scheme="PM-JAY")

        else:  # Aarogyasri
            st.markdown("**Tell us about the family:**")
            is_telangana_resident = st.checkbox("Family lives in Telangana")
            has_white_ration_card = st.checkbox(
                "Family has a White Ration Card"
            )
            has_aadhaar2 = st.checkbox("Family has an Aadhaar card")
            submitted = st.form_submit_button("Run preliminary check")

            if submitted:
                rules = check_aarogyasri(
                    has_aadhaar=has_aadhaar2,
                    has_white_ration_card=has_white_ration_card,
                    is_telangana_resident=is_telangana_resident,
                )
                _render_precheck_results(rules, scheme="Aarogyasri")

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div class='disclaimer'>"
    "⚠️ <b>గమనిక:</b> ఇది ప్రాథమిక సమాచారం మాత్రమే. "
    "అధికారిక అర్హత నిర్ధారణ కోసం దయచేసి ప్రభుత్వ ఆసుపత్రి సహాయ కేంద్రాన్ని "
    "లేదా అధికారిక ప్రభుత్వ వెబ్‌సైట్‌ను సంప్రదించండి. "
    "This tool does not provide medical advice or official eligibility decisions."
    "</div>",
    unsafe_allow_html=True,
)

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
