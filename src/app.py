"""
Arogya Vaani — Streamlit UI
Run:  streamlit run src/app.py
"""
import sys
from pathlib import Path
from datetime import datetime

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
    page_title="Arogya Vaani — Telangana Scheme Copilot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# GLOBAL STYLES
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;700&display=swap');

    :root {
        --av-green: #2e7d32;
        --av-green-light: #e8f5e9;
        --av-amber: #f57c00;
        --av-amber-light: #fff8e1;
        --av-red: #d32f2f;
        --av-red-light: #ffebee;
        --av-bg: #fafafa;
        --av-card-bg: #ffffff;
        --av-text: #212121;
        --av-text-muted: #616161;
        --av-border: #e0e0e0;
    }

    /* Base Layout and Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--av-text);
    }
    .block-container {
        max-width: 1100px;
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
        margin: 0 auto;
    }

    /* Telugu Typography */
    .telugu, [lang="te"], .telugu-text, [class*="telugu"] {
        font-family: 'Noto Sans Telugu', 'Inter', sans-serif !important;
        line-height: 1.7;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Inter', 'Noto Sans Telugu', sans-serif;
        font-weight: 600;
        color: var(--av-text);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        min-height: 44px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--av-green);
        color: #ffffff;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1b5e20;
        color: #ffffff;
    }
    .stButton > button:not([kind="primary"]) {
        background-color: var(--av-card-bg);
        border: 1px solid var(--av-green);
        color: var(--av-green);
    }
    .stButton > button:not([kind="primary"]):hover {
        background-color: var(--av-green-light);
    }

    /* Inputs and Forms */
    .stTextInput input, .stTextArea textarea, .stNumberInput input, [data-baseweb="input"] {
        border-radius: 8px !important;
        border: 1px solid var(--av-border) !important;
        padding: 10px !important;
    }

    /* Nav buttons — styled as tabs */
    button[aria-label="💬 Ask a question"][kind="primary"],
    button[aria-label="📋 Preliminary eligibility check"][kind="primary"] {
        background: transparent !important;
        color: #2e7d32 !important;
        border: none !important;
        border-bottom: 3px solid #2e7d32 !important;
        border-radius: 0 !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    button[aria-label="💬 Ask a question"][kind="secondary"],
    button[aria-label="📋 Preliminary eligibility check"][kind="secondary"] {
        background: transparent !important;
        color: #616161 !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 0 !important;
        padding: 10px 16px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    button[aria-label="💬 Ask a question"][kind="secondary"]:hover,
    button[aria-label="📋 Preliminary eligibility check"][kind="secondary"]:hover {
        color: #2e7d32 !important;
        background: #f5f5f5 !important;
    }

    /* Sidebar Radio Styling */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
    [data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background: transparent !important;
        border: none !important;
        padding: 4px 8px !important;
        color: var(--av-text);
        cursor: pointer;
    }

    /* Audio Input Card Styling */
    div[data-testid="stAudioInput"] {
        background: #fafafa !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        margin-bottom: 8px !important;
        min-height: 56px !important;
    }
    div[data-testid="stAudioInput"] label {
        font-size: 13px !important;
        color: #757575 !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stAudioInput"] audio,
    div[data-testid="stAudioInput"] > div > audio + div {
        height: 32px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--av-border);
        padding: 20px;
    }

    /* User messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        justify-content: flex-end;
        background: transparent;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) > div:first-child:not(:last-child) {
        display: none !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) > div:last-child {
        background: #e8e8e8;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        max-width: 75%;
        margin-left: auto;
        color: #212121;
        font-size: 15px;
    }

    /* Assistant messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #ffffff;
        border-left: 4px solid #2e7d32;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    /* Document Cards */
    .av-doc-row {
        display: flex;
        gap: 14px;
        padding: 16px 18px;
        border-radius: 10px;
        background: #ffffff;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .av-doc-icon {
        min-width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: bold;
    }
    .av-doc-icon.green { background: #e8f5e9; color: #2e7d32; }
    .av-doc-icon.amber { background: #fff8e1; color: #f57c00; }
    .av-doc-icon.red   { background: #ffebee; color: #d32f2f; }
    .av-doc-body { flex: 1; }
    .av-doc-name { font-weight: 600; font-size: 16px; color: #212121; }
    .av-doc-meta { font-size: 13px; color: #616161; margin-top: 6px; line-height: 1.55; }
    .av-doc-meta b { color: #424242; }

    /* Next Steps Timeline */
    .av-step {
        position: relative;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 10px 0;
    }
    .av-step::before {
        content: '';
        position: absolute;
        left: 15px;
        top: 42px;
        bottom: -10px;
        width: 2px;
        background: #d4e5d6;
    }
    .av-step:last-child::before { display: none; }
    .av-step-num {
        min-width: 32px; height: 32px;
        background: #2e7d32;
        color: white;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 600;
        font-size: 14px;
        z-index: 1;
        position: relative;
    }
    .av-step-text {
        padding-top: 5px;
        font-size: 15px;
        color: #212121;
        line-height: 1.55;
    }

    /* Action Buttons Row - Match Listen & Download buttons */
    div[data-testid="stDownloadButton"],
    .stDownloadButton {
        display: block !important;
        height: 48px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stDownloadButton"] > button,
    .stDownloadButton > button,
    div[data-testid="stDownloadButton"] button {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        padding: 0 16px !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        border: 1px solid #2e7d32 !important;
        color: #2e7d32 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        font-family: 'Inter', 'Noto Sans Telugu', -apple-system, BlinkMacSystemFont, sans-serif !important;
        width: 100% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        background: #f1f8e9 !important;
        border-color: #2e7d32 !important;
        color: #1b5e20 !important;
    }
    div[data-testid="stDownloadButton"] button p,
    .stDownloadButton button p {
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #2e7d32 !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    div[data-testid="stDownloadButton"] button:hover p,
    .stDownloadButton button:hover p {
        color: #1b5e20 !important;
    }

    /* Empty State Guidance & Chips */
    .av-empty-state {
        text-align: center;
        padding: 40px 20px 20px 20px;
        background: #fafafa;
        border-radius: 12px;
        margin: 20px 0 16px 0;
    }
    section.main div[data-testid="stButton"] > button:has(
        div:contains("PM-JAY")
    ),
    section.main div[data-testid="stButton"] > button:has(
        div:contains("Aarogyasri")
    ),
    section.main div[data-testid="stButton"] > button:has(
        div:contains("documents")
    ) {
        background: #ffffff !important;
        border: 1px solid #c8e6c9 !important;
        color: #2e7d32 !important;
        border-radius: 20px !important;
        font-size: 13px !important;
        padding: 10px 14px !important;
        height: auto !important;
        min-height: 44px !important;
        white-space: normal !important;
    }

    /* Cards and Containers */
    .av-card {
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid var(--av-border);
        background: var(--av-card-bg);
    }

    /* Custom Source & Disclaimer Boxes */
    .source-box {
        background-color: #f0f4f8;
        border-left: 4px solid var(--av-green);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .disclaimer {
        background-color: var(--av-amber-light);
        border-left: 4px solid var(--av-amber);
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }

    /* Question input row */
    div[data-testid="stTextInput"] input {
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        font-size: 15px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #2e7d32 !important;
        box-shadow: 0 0 0 2px #e8f5e9 !important;
    }

    /* Question Callout Bubble */
    .av-question-bubble {
        background: #f5f5f5;
        border-left: 3px solid #9e9e9e;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    .av-question-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #757575;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .av-question-text {
        font-size: 15px;
        color: #212121;
        line-height: 1.5;
    }
    .av-answer-body {
        color: #1a1a1a !important;
    }

    /* Answer panel — generous spacing */
    .av-answer-panel {
        background: #ffffff;
        border-left: 4px solid #2e7d32;
        border-radius: 12px;
        padding: 24px 28px;
        margin-top: 16px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Input row — compact, bottom-pinned feel */
    section.main div[data-testid="stHorizontalBlock"]:has(
        div[data-testid="stTextInput"] input[placeholder*="ప్రశ్న"]
    ) {
        margin-top: 12px;
        margin-bottom: 8px;
    }

    /* Recent question buttons in sidebar */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        text-align: left !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
        height: auto !important;
        white-space: normal !important;
        line-height: 1.4 !important;
    }

    /* Two-column layout top alignment */
    section.main div[data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
    }

    /* Info Banner */
    .av-info-banner {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #e3f2fd;
        color: #0d47a1;
        border-left: 4px solid #1565c0;
        border-radius: 8px;
        padding: 12px 18px;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .av-info-icon { font-size: 16px; }

    /* Form Card & Inputs */
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e8e8e8 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    .av-card-header {
        font-size: 17px;
        font-weight: 600;
        color: #212121;
        margin-bottom: 18px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f0f0f0;
    }
    section.main div[data-testid="stSelectbox"] > div > div {
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    section.main div[data-testid="stNumberInput"] input {
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
    }

    /* Prominent Run Check Submit Button */
    section.main div[data-testid="stFormSubmitButton"] > button {
        background: #2e7d32 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 20px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100% !important;
        height: 48px !important;
        box-shadow: 0 2px 4px rgba(46,125,50,0.2) !important;
    }
    section.main div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1b5e20 !important;
        box-shadow: 0 3px 6px rgba(46,125,50,0.3) !important;
    }

    /* Rule Outcome Cards */
    .av-rule-card {
        display: flex;
        gap: 12px;
        padding: 14px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #e8e8e8;
        background: #ffffff;
    }
    .av-rule-card.pass { border-left: 4px solid #2e7d32; }
    .av-rule-card.fail { border-left: 4px solid #d32f2f; }
    .av-rule-card.warn { border-left: 4px solid #f57c00; }
    .av-rule-icon {
        min-width: 28px; height: 28px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 14px;
        flex-shrink: 0;
    }
    .av-rule-card.pass .av-rule-icon { background: #e8f5e9; color: #2e7d32; }
    .av-rule-card.fail .av-rule-icon { background: #ffebee; color: #d32f2f; }
    .av-rule-card.warn .av-rule-icon { background: #fff8e1; color: #f57c00; }
    .av-rule-title { font-weight: 600; font-size: 14px; color: #212121; }
    .av-rule-reason {
        font-size: 13px;
        color: #37474f;
        line-height: 1.6;
        margin-top: 5px;
    }

    /* Scheme Result Verdict Cards */
    .av-result-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .av-result-card.eligible {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
    }
    .av-result-card.not-eligible {
        background: #ffebee;
        border: 1px solid #ef9a9a;
    }
    .av-result-card.unclear {
        background: #fff8e1;
        border: 1px solid #ffe082;
    }
    .av-result-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .av-result-card.eligible .av-result-header { color: #1b5e20; }
    .av-result-card.not-eligible .av-result-header { color: #b71c1c; }
    .av-result-card.unclear .av-result-header { color: #e65100; }
    .av-result-reason {
        font-size: 13px;
        line-height: 1.6;
        color: #37474f;
    }

    /* Summary Text Card */
    .av-summary-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-top: 12px;
    }
    .av-summary-card .av-summary-label {
        font-size: 13px;
        font-weight: 600;
        color: #616161;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .av-summary-card .av-summary-text {
        font-size: 14px;
        color: #212121;
        line-height: 1.65;
    }

    /* Top bar logo button */
    div.st-key-home_logo_btn > button,
    section.main div[data-testid="stButton"] > button[kind="secondary"]:has(
        div:contains("ఆరోగ్య వాణి")
    ),
    section.main div[data-testid="stButton"] > button:has(
        p:contains("ఆరోగ్య వాణి")
    ) {
        background: transparent !important;
        border: none !important;
        color: #2e7d32 !important;
        text-align: left !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        line-height: 1.35 !important;
        white-space: pre-line !important;
        box-shadow: none !important;
        height: auto !important;
        min-height: 0 !important;
        width: auto !important;
        cursor: pointer;
    }
    div.st-key-home_logo_btn > button:hover,
    section.main div[data-testid="stButton"] > button[kind="secondary"]:has(
        div:contains("ఆరోగ్య వాణి")
    ):hover {
        background: #f1f8e9 !important;
        border-radius: 8px !important;
    }

    /* Hide Streamlit default chrome */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
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
    summary = summarize(rules)
    counts = summary["counts"]

    if summary["summary"] == "preliminary_positive":
        res_class = "eligible"
        verdict = "అర్హులు (Eligible)"
        icon = "✓"
    elif summary["summary"] == "preliminary_missing":
        res_class = "not-eligible"
        verdict = "అనర్హులు / పత్రాలు అవసరం (Not Eligible / Missing Docs)"
        icon = "✕"
    else:
        res_class = "unclear"
        verdict = "పరిశీలన అవసరం (Needs Verification)"
        icon = "!"

    st.markdown(
        f"""
        <div class="av-result-card {res_class}">
          <div class="av-result-header">
            <span class="av-result-icon">{icon}</span>
            <span>{scheme}: {verdict}</span>
          </div>
          <div class="av-result-reason">{summary["summary_text"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"#### ప్రాథమిక సూచికలు ({scheme})")

    for r in rules:
        status_key = r.get("status", "unclear")
        card_class = "pass" if status_key == "met" else ("fail" if status_key == "not_met" else "warn")
        icon = "✓" if status_key == "met" else ("✕" if status_key == "not_met" else "!")
        factor = r.get("factor", "")
        reason = r.get("reason", "")
        st.markdown(
            f"""
            <div class="av-rule-card {card_class}">
              <div class="av-rule-icon">{icon}</div>
              <div class="av-rule-body">
                <div class="av-rule-title">{factor}</div>
                <div class="av-rule-reason">{reason}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="av-summary-card">
          <div class="av-summary-label">అర్హత నివేదిక సారాంశం (Summary)</div>
          <div class="av-summary-text">
            <b>{counts['met']}</b> present · <b>{counts['not_met']}</b> missing · <b>{counts['unclear']}</b> unclear
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            icon_class = "red" if required else "amber"
            icon_symbol = "✕" if required else "⚠"

            meta_parts = []
            if why:
                meta_parts.append(f"<b>Why:</b> {why}")
            if if_missing:
                meta_parts.append(f"<b>If missing:</b> {if_missing}")
            meta_html = "<br>".join(meta_parts)

            st.markdown(
                f'<div class="av-doc-row" style="border-left:4px solid {tag_color};">'
                f'  <div class="av-doc-icon {icon_class}">{icon_symbol}</div>'
                f'  <div class="av-doc-body">'
                f'    <div class="av-doc-name">{name} <span style="background:{tag_color};color:white;padding:2px 8px;border-radius:10px;font-size:0.75rem;margin-left:6px;font-weight:normal;">{tag_text}</span></div>'
                f'    <div class="av-doc-meta">{meta_html}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Next steps card
    if steps:
        st.markdown("### ➡️ తదుపరి చర్యలు")
        steps_html = ""
        for i, s in enumerate(steps, 1):
            steps_html += (
                f'<div class="av-step">'
                f'<div class="av-step-num">{i}</div>'
                f'<div class="av-step-text">{s}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="background:#fdfdfd;padding:16px 20px;border-radius:10px;border:1px solid #e0e0e0;">{steps_html}</div>',
            unsafe_allow_html=True,
        )

    # Missing info card
    if missing:
        st.markdown("### ⚠️ అస్పష్టమైన సమాచారం")
        for m in missing:
            st.markdown(
                f"<div style='border-left:4px solid #f57c00;padding:12px 16px;"
                f"background:#fff8e1;border-radius:8px;margin:8px 0;font-size:14px'>{m}</div>",
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

    safe_text = (
        text.replace("\\", "\\\\")
            .replace("`", "\\`")
            .replace("${", "\\${")
    )

    html = f"""
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; }}
.av-act-btn {{
  background: #ffffff;
  color: #2e7d32;
  border: 1px solid #2e7d32;
  padding: 0 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 500;
  font-family: 'Inter', 'Noto Sans Telugu', -apple-system, BlinkMacSystemFont, sans-serif;
  height: 48px;
  width: 100%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: all 0.15s ease;
  box-sizing: border-box;
}}
.av-act-btn:hover {{
  background: #f1f8e9;
}}
.av-stop-btn {{
  background: #ffffff;
  color: #d32f2f;
  border: 1px solid #d32f2f;
  padding: 0 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 500;
  font-family: 'Inter', 'Noto Sans Telugu', -apple-system, BlinkMacSystemFont, sans-serif;
  height: 48px;
  width: 100%;
  display: none;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: all 0.15s ease;
  box-sizing: border-box;
}}
.av-stop-btn:hover {{
  background: #ffebee;
}}
</style>
<div style="display:flex;align-items:center;gap:10px;margin:0;padding:0;height:48px;">
  <button id="speak-btn-{key_suffix}" class="av-act-btn">🔊 వినండి</button>
  <button id="stop-btn-{key_suffix}" class="av-stop-btn">⏹ ఆపండి</button>
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

  if (speechSynthesis.onvoiceschanged !== undefined) {{
    speechSynthesis.onvoiceschanged = () => {{}};
  }}
}})();
</script>
"""

    components.html(html, height=48)

# -------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------------
if "sidebar_forced_open" not in st.session_state:
    st.session_state.sidebar_forced_open = True
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "precheck_submitted" not in st.session_state:
    st.session_state.precheck_submitted = False
if "started" not in st.session_state:
    st.session_state.started = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "_last_audio_id" not in st.session_state:
    st.session_state["_last_audio_id"] = None
if "scheme_choice" not in st.session_state:
    st.session_state.scheme_choice = "Both"
if "show_mic" not in st.session_state:
    st.session_state.show_mic = False
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "current_answer" not in st.session_state:
    st.session_state.current_answer = None
if "current_structured" not in st.session_state:
    st.session_state.current_structured = None
if "current_chunks" not in st.session_state:
    st.session_state.current_chunks = []
if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []  # list of dicts: {q, answer, structured, chunks, ts}

# -------------------------------------------------------------------
# TOP APP BAR
# -------------------------------------------------------------------
top_left, top_right = st.columns([2, 1])

with top_left:
    if st.button(
        "💚  ఆరోగ్య వాణి\nAROGYA VAANI • TELANGANA SCHEME COPILOT",
        key="home_logo_btn",
        help="Home",
    ):
        st.session_state.messages = []
        st.session_state.current_question = None
        st.session_state.current_answer = None
        st.session_state.current_structured = None
        st.session_state.current_chunks = []
        st.session_state.active_tab = "chat"
        st.session_state.precheck_submitted = False
        st.session_state["precheck_results"] = None
        st.session_state.started = False
        st.session_state.show_mic = False
        st.rerun()

with top_right:
    st.markdown(
        "<div style='display:flex;gap:8px;justify-content:flex-end;padding-top:8px'>"
        "<span style='background:#e3f2fd;color:#1565c0;padding:6px 14px;"
        "border-radius:20px;font-size:13px;font-weight:500;display:inline-flex;align-items:center;gap:6px;'>👤 ఆశా కార్యకర్త (ASHA Volunteer)</span>"
        "<a href='#' style='background:#e8f5e9;color:#2e7d32;padding:6px 14px;"
        "border-radius:20px;font-size:13px;font-weight:500;text-decoration:none;display:inline-flex;align-items:center;gap:4px;'>📍 TS Health Portal</a>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='border-bottom: 1px solid #e0e0e0; margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# VIEW ROUTING: HOME SCREEN vs WORKSPACE (Chat / Pre-check)
# -------------------------------------------------------------------
show_home = (
    not st.session_state.get("current_answer")
    and len(st.session_state.get("messages", [])) == 0
    and not st.session_state.get("precheck_submitted", False)
    and not st.session_state.get("started", False)
)

if show_home:
    # Hide sidebar on home welcome page
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    scheme_choice = "Both"
    show_sources = True
    show_debug = False
    # ---------------------------------------------------------------
    # HOME SCREEN
    # ---------------------------------------------------------------
    # 1. Centered hero section
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 2rem 0;">
          <h1 style="font-family:'Noto Sans Telugu', sans-serif;font-size:32px;font-weight:700;color:#2e7d32;margin-bottom:8px;">
            నమస్తే! ఆరోగ్య పథకాల సహాయకారికి స్వాగతం
          </h1>
          <p style="font-size:16px;color:#616161;margin:0;">
            ASHA Copilot: Quick guidance for Aarogyasri & PM-JAY schemes in Telangana
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Side-by-side cards
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown(
            """
            <div style="border:2px solid #2e7d32;background:#e8f5e9;border-radius:12px;padding:24px;min-height:220px;display:flex;flex-direction:column;justify-content:space-between;">
              <div>
                <div style="font-size:20px;font-weight:700;color:#212121;margin-bottom:12px;font-family:'Noto Sans Telugu', sans-serif;">
                  🎤 ప్రశ్న అడగండి (Voice Ask)
                </div>
                <p style="font-size:15px;color:#424242;line-height:1.6;font-family:'Noto Sans Telugu', sans-serif;margin-bottom:16px;">
                  పథకాల ప్రయోజనాలు, చికిత్సల వివరాలు తెలుసుకోవడానికి మాట్లాడండి లేదా టైప్ చేయండి.
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("ప్రశ్న అడగడం ప్రారంభించండి →", key="btn_start_chat", type="primary", use_container_width=True):
            st.session_state.active_tab = "chat"
            st.session_state.started = True
            st.rerun()

    with col_right:
        st.markdown(
            """
            <div style="border:2px solid #f57c00;background:#fff8e1;border-radius:12px;padding:24px;min-height:220px;display:flex;flex-direction:column;justify-content:space-between;">
              <div>
                <div style="font-size:20px;font-weight:700;color:#212121;margin-bottom:12px;font-family:'Noto Sans Telugu', sans-serif;">
                  📋 అర్హత తనిఖీ (Eligibility Check)
                </div>
                <p style="font-size:15px;color:#424242;line-height:1.6;font-family:'Noto Sans Telugu', sans-serif;margin-bottom:16px;">
                  రేషన్ కార్డ్, ఆధార్ వివరాలతో లబ్ధిదారుని ప్రాథమిక అర్హతను సులభంగా తనిఖీ చేయండి.
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("అర్హత తనిఖీ చేయండి →", key="btn_start_precheck", type="primary", use_container_width=True):
            st.session_state.active_tab = "precheck"
            st.session_state.started = True
            st.rerun()

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # 3. How to use card
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #e0e0e0;border-radius:12px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
          <div style="font-size:18px;font-weight:700;color:#212121;margin-bottom:16px;font-family:'Noto Sans Telugu', sans-serif;">
            📖 ఎలా ఉపయోగించాలి? (How to use?)
          </div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:20px;">
            <div>
              <div style="color:#2e7d32;font-size:12px;font-weight:700;letter-spacing:0.5px;margin-bottom:4px;">STEP 1</div>
              <div style="font-weight:700;font-size:15px;color:#212121;margin-bottom:6px;font-family:'Noto Sans Telugu', sans-serif;">అవసరమైన పథకాన్ని ఎంచుకోండి</div>
              <div style="font-size:13px;color:#616161;line-height:1.5;font-family:'Noto Sans Telugu', sans-serif;">ఆరోగ్యశ్రీ లేదా PM-JAY పథకాలను ఎంపిక చేసుకోండి.</div>
            </div>
            <div>
              <div style="color:#2e7d32;font-size:12px;font-weight:700;letter-spacing:0.5px;margin-bottom:4px;">STEP 2</div>
              <div style="font-weight:700;font-size:15px;color:#212121;margin-bottom:6px;font-family:'Noto Sans Telugu', sans-serif;">వివరాలు మాట్లాడండి లేదా నమోదు చేయండి</div>
              <div style="font-size:13px;color:#616161;line-height:1.5;font-family:'Noto Sans Telugu', sans-serif;">మైక్ బటన్ నొక్కి తెలుగులో మీ ప్రశ్న అడగండి.</div>
            </div>
            <div>
              <div style="color:#2e7d32;font-size:12px;font-weight:700;letter-spacing:0.5px;margin-bottom:4px;">STEP 3</div>
              <div style="font-weight:700;font-size:15px;color:#212121;margin-bottom:6px;font-family:'Noto Sans Telugu', sans-serif;">అర్హత నివేదికను పొందండి</div>
              <div style="font-size:13px;color:#616161;line-height:1.5;font-family:'Noto Sans Telugu', sans-serif;">అవసరమైన పత్రాల జాబితా మరియు తదుపరి దశల గైడెన్స్ చూడండి.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    # ---------------------------------------------------------------
    # Ensure sidebar is opened by default on entering workspace
    # ---------------------------------------------------------------
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            try {
                const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                const collapsedBtn = window.parent.document.querySelector('[data-testid="stSidebarCollapsedControl"] button')
                                  || window.parent.document.querySelector('[data-testid="collapsedControl"]')
                                  || window.parent.document.querySelector('button[aria-label*="Open sidebar"]')
                                  || window.parent.document.querySelector('button[aria-label*="expand sidebar"]');
                if (sidebar && sidebar.getAttribute('aria-expanded') === 'false' && collapsedBtn) {
                    collapsedBtn.click();
                }
            } catch(e) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )

    # ---------------------------------------------------------------
    # SIDEBAR (Workspace Only)
    # ---------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🏥 Arogya Vaani")
        st.caption("Telugu-first healthcare scheme copilot")
        st.markdown("---")

        st.markdown("### 📋 Scheme Filter")
        scheme_options = ["Both", "PM-JAY", "Aarogyasri"]
        curr_scheme_idx = scheme_options.index(st.session_state.scheme_choice) if st.session_state.scheme_choice in scheme_options else 0
        sidebar_scheme = st.radio(
            "Which scheme is the family asking about?",
            scheme_options,
            index=curr_scheme_idx,
            key="sidebar_scheme_radio",
        )
        if sidebar_scheme != st.session_state.scheme_choice:
            st.session_state.scheme_choice = sidebar_scheme
            st.rerun()
        scheme_choice = st.session_state.scheme_choice

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
            if st.button(q, use_container_width=True, key=f"quick_{abs(hash(q))}"):
                st.session_state["pending_question"] = q
                st.session_state.active_tab = "chat"
                st.session_state.started = True
                st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_sources = st.checkbox("Show retrieved sources", value=True)
        show_debug = st.checkbox("Show debug info", value=False)

        st.markdown("---")
        st.markdown("### 🕐 ఇటీవలి ప్రశ్నలు (Recent)")

        if not st.session_state.recent_questions:
            st.caption("No recent questions yet")
        else:
            for i, item in enumerate(st.session_state.recent_questions[:5]):
                label = item["q"][:40] + ("..." if len(item["q"]) > 40 else "")
                if st.button(label, key=f"recent_{i}", use_container_width=True):
                    st.session_state.current_question = item["q"]
                    st.session_state.current_answer = item["answer"]
                    st.session_state.current_structured = item["structured"]
                    st.session_state.current_chunks = item["chunks"]
                    st.session_state.active_tab = "chat"
                    st.session_state.started = True
                    st.rerun()

        st.markdown("---")
        st.caption("Built for the Unleash LLM Innovation Challenge 2026")
        st.caption("Team: HealthWhisperers")

    # ---------------------------------------------------------------
    # WORKSPACE VIEW (Chat + Pre-check with Button Navigation)
    # ---------------------------------------------------------------
    # --- Custom tab navigation (buttons) ---
    nav_col1, nav_col2, _nav_spacer = st.columns([1.5, 2.5, 3])

    current_tab = st.session_state.get("active_tab", "chat")

    with nav_col1:
        if st.button(
            "💬 Ask a question",
            key="nav_btn_chat",
            use_container_width=True,
            type="primary" if current_tab == "chat" else "secondary",
        ):
            st.session_state.active_tab = "chat"
            st.rerun()

    with nav_col2:
        if st.button(
            "📋 Preliminary eligibility check",
            key="nav_btn_precheck",
            use_container_width=True,
            type="primary" if current_tab == "precheck" else "secondary",
        ):
            st.session_state.active_tab = "precheck"
            st.rerun()

    st.markdown("<div style='border-bottom: 1px solid #e0e0e0; margin-bottom: 20px'></div>", unsafe_allow_html=True)

    if st.session_state.active_tab == "chat":
        # ---------------------------------------------------------------
        # 1. QUESTION PROCESSING FLOW (WHEN SUBMITTED)
        # ---------------------------------------------------------------
        triggered_question = None

        if st.session_state.pending_question:
            triggered_question = st.session_state.pending_question
            st.session_state.pending_question = None

        if triggered_question:
            st.session_state.current_question = triggered_question
            st.session_state.current_answer = None
            st.session_state.current_structured = None
            st.session_state.current_chunks = []

            retriever = load_retriever()
            with st.spinner("🔍 Expanding query and retrieving relevant scheme information..."):
                expanded = expand_query(triggered_question)
                combined_query = f"{triggered_question} {expanded['keywords']}"
                chunks = retriever.search(
                    combined_query,
                    k=TOP_K_FINAL + 1,
                    scheme=expanded["scheme"],
                )

            if st.session_state.scheme_choice != "Both":
                scheme_map = {"PM-JAY": "pmjay", "Aarogyasri": "aarogyasri"}
                target = scheme_map[st.session_state.scheme_choice]
                filtered = [c for c in chunks if c["metadata"]["scheme"] == target]
                if filtered:
                    chunks = filtered

            placeholder = st.empty()
            full_answer = ""
            try:
                for token in generate_answer_stream(triggered_question, chunks):
                    full_answer += token
                    placeholder.markdown(
                        f"<div class='av-answer-panel'><h3>📝 సమాధానం (Answer)</h3>"
                        f"<div class='av-question-bubble'>"
                        f"<div class='av-question-label'>ప్రశ్న (Question)</div>"
                        f"<div class='av-question-text'>{triggered_question}</div>"
                        f"</div>"
                        f"<div class='av-answer-body' style='font-size:15px;line-height:1.7;'>{full_answer}</div></div>",
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error calling LLM: {e}")
                full_answer = "క్షమించండి, సాంకేతిక సమస్య ఏర్పడింది. దయచేసి మళ్లీ ప్రయత్నించండి."

            structured = {"documents": [], "next_steps": [], "missing_info": [], "preliminary_only": True}
            if full_answer and len(full_answer) > 50 and "క్షమించండి" not in full_answer:
                try:
                    structured = extract_structured_data(triggered_question, full_answer, chunks)
                except Exception as e:
                    st.caption(f"(Structured extraction skipped: {e})")

            st.session_state.current_answer = full_answer
            st.session_state.current_structured = structured
            st.session_state.current_chunks = chunks

            # Auto-save to recent questions (deduplicated by question, capped at 10)
            recents = [r for r in st.session_state.recent_questions if r.get("q") != triggered_question]
            recents.insert(0, {
                "q": triggered_question,
                "answer": full_answer,
                "structured": structured,
                "chunks": chunks,
                "ts": datetime.now().strftime("%Y%m%d-%H%M"),
            })
            st.session_state.recent_questions = recents[:10]
            st.rerun()

        # ---------------------------------------------------------------
        # 2. ANSWER PANEL (TOP)
        # ---------------------------------------------------------------
        if not st.session_state.current_answer:
            st.markdown(
                "<div class='av-empty-state'>"
                "<div style='font-size:44px;margin-bottom:12px'>💬</div>"
                "<div style='font-size:18px;font-weight:600;color:#2e7d32;margin-bottom:6px'>"
                "మీ ప్రశ్న ఇక్కడ కనిపిస్తుంది</div>"
                "<div style='font-size:14px;color:#616161;margin-bottom:20px'>"
                "Ask a question to see the guidance here</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='text-align:center;font-size:13px;color:#757575;margin-bottom:10px'>"
                "💡 ప్రయత్నించండి (Try one of these):</div>",
                unsafe_allow_html=True,
            )
            chip_cols = st.columns(3)
            suggestion_chips = [
                "PM-JAY eligibility emiti?",
                "Aarogyasri lo em cover avutundi?",
                "What documents are required for PM-JAY?",
            ]
            for i, chip_text in enumerate(suggestion_chips):
                with chip_cols[i]:
                    if st.button(
                        chip_text,
                        key=f"chip_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_question = chip_text
                        st.rerun()
        else:
            with st.container():
                st.markdown(
                    f"<div class='av-answer-panel'>"
                    f"<h3 style='margin-top:0;'>📝 సమాధానం (Answer)</h3>"
                    f"<div class='av-question-bubble'>"
                    f"<div class='av-question-label'>ప్రశ్న (Question)</div>"
                    f"<div class='av-question-text'>{st.session_state.current_question}</div>"
                    f"</div>"
                    f"<div class='av-answer-body' style='font-size:15px;line-height:1.7;'>{st.session_state.current_answer}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if st.session_state.current_structured:
                    _render_structured_cards(st.session_state.current_structured)

                act_col1, act_col2 = st.columns([1, 1], gap="small")
                with act_col1:
                    _render_audio_player(
                        st.session_state.current_answer,
                        key_suffix=f"workbench_{abs(hash(st.session_state.current_question or '')) % 100000}",
                    )
                with act_col2:
                    download_text = _build_download_text(
                        st.session_state.current_question or "",
                        st.session_state.current_answer,
                        st.session_state.current_structured or {},
                        st.session_state.current_chunks or [],
                    )
                    ts = datetime.now().strftime("%Y%m%d-%H%M")
                    st.download_button(
                        label="📥 డౌన్లోడ్",
                        data=download_text.encode("utf-8"),
                        file_name=f"arogya-vaani-{ts}.txt",
                        mime="text/plain",
                        key="download_current_answer",
                        use_container_width=True,
                    )

                if show_sources and st.session_state.current_chunks:
                    with st.expander(f"📚 Sources ({len(st.session_state.current_chunks)} chunks retrieved)", expanded=False):
                        for i, s in enumerate(st.session_state.current_chunks, 1):
                            m = s["metadata"]
                            st.markdown(
                                f"<div class='source-box'>"
                                f"<b>[{i}] {m['source_file']}</b> — page {m['page']}<br>"
                                f"<small>Scheme: <b>{m['scheme']}</b> · Type: {m['doc_type']}</small><br>"
                                f"<small>{s['text'][:300]}...</small>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

        # ---------------------------------------------------------------
        # 3. AUDIO INPUT CARD (CONDITIONAL, ABOVE INPUT ROW)
        # ---------------------------------------------------------------
        if st.session_state.show_mic:
            audio_value = st.audio_input(
                "🎙️ మీ ప్రశ్నను రికార్డ్ చేయండి (Record your question)",
                key="voice_input_widget",
            )
            if audio_value is not None and st.session_state.get("_last_audio_id") != id(audio_value):
                with st.spinner("🎙️ Transcribing your voice..."):
                    try:
                        audio_bytes = audio_value.read()
                        transcript = transcribe_audio(audio_bytes, filename="question.wav")
                        if transcript:
                            st.session_state["_last_audio_id"] = id(audio_value)
                            st.session_state["pending_question"] = transcript
                            st.rerun()
                        else:
                            st.warning("Could not transcribe the audio. Please try again or type your question.")
                    except Exception as e:
                        st.error(f"Audio error: {e}")

        # ---------------------------------------------------------------
        # 4. QUESTION INPUT ROW (BOTTOM)
        # ---------------------------------------------------------------
        input_col, voice_col, ask_col = st.columns([6, 1.4, 1.4])
        with input_col:
            typed_question = st.text_input(
                "మీ ప్రశ్న",
                placeholder="ప్రశ్న టైప్ చేయండి లేదా మాట్లాడండి... (Type or speak your question)",
                label_visibility="collapsed",
                key="main_question_input",
            )
        with voice_col:
            if st.button("🎤 Voice", key="btn_toggle_voice", use_container_width=True, type="primary" if st.session_state.show_mic else "secondary"):
                st.session_state.show_mic = not st.session_state.show_mic
                st.rerun()
        with ask_col:
            ask_clicked = st.button("➡️ Ask", key="btn_ask_submit", type="primary", use_container_width=True)

        if (ask_clicked or typed_question) and typed_question.strip():
            st.session_state["pending_question"] = typed_question.strip()
            st.rerun()

    else:
        # ---------------------------------------------------------------
        # PRELIMINARY ELIGIBILITY CHECK
        # ---------------------------------------------------------------
        st.markdown(
            """
            <div class="av-info-banner">
              <span class="av-info-icon">ℹ️</span>
              <span>Deterministic rules engine. No AI generation here. Indicative calculations only. | నిర్ణయ ప్రాథమిక సమాచారం. ఇది కేవలం ప్రాథమిక సూచకం.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            scheme_for_check = st.radio(
                "Select the scheme to check:",
                ["PM-JAY", "Aarogyasri"],
                key="precheck_scheme",
                horizontal=True,
            )

            with st.form("precheck_form"):
                st.markdown(
                    """
                    <div class="av-card-header">📝 లబ్ధిదారుని వివరాలు (Enter Details)</div>
                    """,
                    unsafe_allow_html=True,
                )
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
                    submitted = st.form_submit_button("అర్హత తనిఖీ చేయండి (Run Check)")

                    if submitted:
                        st.session_state.precheck_submitted = True
                        rules = check_pmjay(
                            has_aadhaar=has_aadhaar,
                            has_ration_card=has_ration_card,
                            has_secc_listing=has_secc_listing,
                            is_bpl_family=is_bpl_family,
                            family_size=int(family_size),
                        )
                        st.session_state["precheck_results"] = (rules, "PM-JAY")
                        st.rerun()

                else:  # Aarogyasri
                    st.markdown("**Tell us about the family:**")
                    is_telangana_resident = st.checkbox("Family lives in Telangana")
                    has_white_ration_card = st.checkbox(
                        "Family has a White Ration Card"
                    )
                    has_aadhaar2 = st.checkbox("Family has an Aadhaar card")
                    submitted = st.form_submit_button("అర్హత తనిఖీ చేయండి (Run Check)")

                    if submitted:
                        st.session_state.precheck_submitted = True
                        rules = check_aarogyasri(
                            has_aadhaar=has_aadhaar2,
                            has_white_ration_card=has_white_ration_card,
                            is_telangana_resident=is_telangana_resident,
                        )
                        st.session_state["precheck_results"] = (rules, "Aarogyasri")
                        st.rerun()

        with col_result:
            if st.session_state.get("precheck_results"):
                res_rules, res_scheme = st.session_state["precheck_results"]
                _render_precheck_results(res_rules, res_scheme)
            else:
                st.markdown(
                    """
                    <div style="text-align:center;padding:48px 24px;background:#f9f9f9;border:1px dashed #e0e0e0;border-radius:12px;color:#757575;">
                      <div style="font-size:32px;margin-bottom:8px;">📋</div>
                      <div style="font-size:15px;font-weight:600;color:#424242;margin-bottom:4px;">ప్రాథమిక అర్హత ఫలితాలు ఇక్కడ కనిపిస్తాయి</div>
                      <div style="font-size:13px;">Fill out the family details on the left and click Run Check.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# -------------------------------------------------------------------
# FOOTER (GLOBAL)
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-top:3rem;padding-top:1.25rem;border-top:1px dashed #b0bec5;background:var(--av-bg);">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;padding:8px 0;">
        <div style="font-size:12px;color:#616161;max-width:380px;">
          Disclaimer: This is an assistive tool for ASHA volunteers. Final scheme eligibility is subject to official government guidelines.
        </div>
        <div style="font-size:13px;color:#e65100;font-weight:600;font-family:'Noto Sans Telugu', sans-serif;text-align:center;">
          ⚠️ గమనిక: ఇది కేవలం సహాయకారి మాత్రమే. తుది నిర్ణయం ప్రభుత్వ నిబంధనలకు లోబడి ఉంటుంది.
        </div>
        <div style="font-size:12px;font-weight:600;color:#616161;background:#eceff1;padding:4px 10px;border-radius:12px;">
          Team HealthWhisperers
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
