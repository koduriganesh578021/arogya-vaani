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
        background: #fafafa;
        border: 1px dashed #b0bec5;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    div[data-testid="stAudioInput"] label {
        font-size: 14px !important;
        color: var(--av-text-muted) !important;
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

    /* Action Buttons Row */
    div[data-testid="stDownloadButton"] > button {
        border: 1px solid #616161 !important;
        color: #616161 !important;
        background: transparent !important;
        height: 40px !important;
        min-height: 40px !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: #f5f5f5 !important;
        color: #212121 !important;
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
<div style="display:flex;align-items:center;gap:10px;margin-top:16px">
  <button id="speak-btn-{key_suffix}" style="
    background:transparent;color:#2e7d32;border:1px solid #2e7d32;padding:8px 16px;
    border-radius:8px;cursor:pointer;font-size:14px;font-weight:500;font-family:'Inter',sans-serif;
    height:40px;display:inline-flex;align-items:center;gap:6px;transition:all 0.15s;
  ">🔊 వినండి (Listen)</button>
  <button id="stop-btn-{key_suffix}" style="
    background:transparent;color:#d32f2f;border:1px solid #d32f2f;padding:8px 16px;
    border-radius:8px;cursor:pointer;font-size:14px;font-weight:500;font-family:'Inter',sans-serif;
    height:40px;display:none;align-items:center;gap:6px;transition:all 0.15s;
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

    components.html(html, height=70)

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

# -------------------------------------------------------------------
# TOP APP BAR
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex;width:100%;min-height:60px;background:#ffffff;border-bottom:1px solid #e0e0e0;padding:12px 24px;align-items:center;justify-content:space-between;border-radius:8px;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
      <div style="display:flex;align-items:center;gap:12px;">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2e7d32" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
          <path d="M12 9v6m-3-3h6"/>
        </svg>
        <div>
          <div style="font-family:'Noto Sans Telugu', sans-serif;font-size:18px;font-weight:700;color:#2e7d32;line-height:1.2;">ఆరోగ్య వాణి</div>
          <div style="font-size:11px;color:#616161;letter-spacing:0.5px;font-weight:600;text-transform:uppercase;">AROGYA VAANI • Telangana Scheme Copilot</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="background:#e3f2fd;color:#1565c0;border-radius:20px;padding:6px 14px;font-size:13px;font-weight:500;display:inline-flex;align-items:center;gap:6px;">
          👤 ఆశా కార్యకర్త (ASHA Volunteer)
        </span>
        <a href="#" style="background:#e8f5e9;color:#2e7d32;border-radius:20px;padding:6px 14px;font-size:13px;font-weight:500;text-decoration:none;display:inline-flex;align-items:center;gap:4px;">
          📍 TS Health Portal
        </a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# VIEW ROUTING: HOME SCREEN vs WORKSPACE (Chat / Pre-check)
# -------------------------------------------------------------------
show_home = (
    len(st.session_state.get("messages", [])) == 0
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
                st.session_state.active_tab = "chat"
                st.session_state.started = True

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_sources = st.checkbox("Show retrieved sources", value=True)
        show_debug = st.checkbox("Show debug info", value=False)

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
        # CHAT HISTORY / EMPTY STATE
        # ---------------------------------------------------------------
        if not st.session_state.messages:
            st.markdown(
                "<div style='text-align:center;padding:40px 20px;"
                "background:#f5f5f5;border-radius:12px;color:#616161;"
                "margin:20px 0'>"
                "<div style='font-size:32px;margin-bottom:8px'>💬</div>"
                "<div style='font-size:16px;margin-bottom:4px;font-weight:600;font-family:\"Noto Sans Telugu\", sans-serif;'>"
                "మీ ప్రశ్న ఇక్కడ కనిపిస్తుంది</div>"
                "<div style='font-size:14px'>"
                "Ask a question or select a quick question from the sidebar</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
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

        # ---------------------------------------------------------------
        # INPUT — voice input card & chat input
        # ---------------------------------------------------------------
        audio_value = st.audio_input(
            "🎙️ మీ ప్రశ్నను రికార్డ్ చేయండి (Record your question)",
            key="voice_input_widget",
        )

        user_question = st.chat_input("Ask about PM-JAY or Aarogyasri (Telugu or English)…")

        if st.session_state.pending_question:
            user_question = st.session_state.pending_question
            st.session_state.pending_question = None

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

        # ---------------------------------------------------------------
        # PROCESS THE QUESTION
        # ---------------------------------------------------------------
        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

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

                structured = {"documents": [], "next_steps": [], "missing_info": [], "preliminary_only": True}
                if full_answer and len(full_answer) > 50 and "క్షమించండి" not in full_answer:
                    try:
                        structured = extract_structured_data(user_question, full_answer, chunks)
                    except Exception as e:
                        st.caption(f"(Structured extraction skipped: {e})")

                _render_structured_cards(structured)

                _render_audio_player(full_answer, key_suffix=f"live_{abs(hash(user_question)) % 100000}")

                download_text = _build_download_text(user_question, full_answer, structured, chunks)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M")
                st.download_button(
                    label="📥 సమాధానాన్ని డౌన్‌లోడ్ చేయండి (Download explanation)",
                    data=download_text.encode("utf-8"),
                    file_name=f"arogya-vaani-{timestamp}.txt",
                    mime="text/plain",
                    key=f"download_{timestamp}",
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": chunks,
                    "structured": structured,
                    "question": user_question,
                    "timestamp": datetime.now().strftime("%Y%m%d-%H%M"),
                })

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

    else:
        # ---------------------------------------------------------------
        # PRELIMINARY ELIGIBILITY CHECK
        # ---------------------------------------------------------------
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
                    st.session_state.precheck_submitted = True
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
                    st.session_state.precheck_submitted = True
                    rules = check_aarogyasri(
                        has_aadhaar=has_aadhaar2,
                        has_white_ration_card=has_white_ration_card,
                        is_telangana_resident=is_telangana_resident,
                    )
                    _render_precheck_results(rules, scheme="Aarogyasri")

    # Clear chat button
    if st.session_state.messages or st.session_state.get("precheck_submitted"):
        st.markdown("---")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.precheck_submitted = False
            st.session_state.started = False
            st.session_state.active_tab = "chat"
            st.rerun()

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
