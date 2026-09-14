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
from src.llm import generate_answer_stream, expand_query
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

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
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
    user_question = st.chat_input("Ask about PM-JAY or Aarogyasri (Telugu or English)…")

    # If a quick question was clicked, use that
    if st.session_state.pending_question:
        user_question = st.session_state.pending_question
        st.session_state.pending_question = None

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

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": chunks,
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
