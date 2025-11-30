import os
import json
import re
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import yaml
import altair as alt

# --- Optional PDF / OCR libraries ---
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# --- LLM client libraries ---
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

# --- xAI (Grok) SDK (per official sample) ---
from xai_sdk import Client as XaiClient
from xai_sdk.chat import user as xai_user, system as xai_system
# from xai_sdk.chat import image as xai_image  # for future image OCR use

# -----------------------------------------------------------
# Nordic Theme + Flower Styles Configuration
# -----------------------------------------------------------

# Light/dark base palettes in a clean Nordic style
FDA_THEMES = {
    "light": {
        "primary": "#2F4F4F",   # dark slate
        "secondary": "#7EA7A6", # muted teal
        "background": "#F5F7F8",# very light grey
        "text": "#1E2A32",      # deep grey-blue
        "accent": "#FF7F50",    # coral for key highlights
    },
    "dark": {
        "primary": "#E0F2F1",   # light teal
        "secondary": "#90A4AE", # blue-grey
        "background": "#111827",# near-black blue-grey
        "text": "#E5E7EB",      # light grey
        "accent": "#FF7F50",    # coral for key highlights
    }
}

# 20 flower-based styles – used as an extra accent layer via "magic wheel"
FLOWER_STYLES = {
    "Lily":       {"icon": "🤍", "color": "#F6E9E9", "description": "Calm white lily, minimal and pure."},
    "Rose":       {"icon": "🌹", "color": "#F28B82", "description": "Soft rose warmth with subtle contrast."},
    "Tulip":      {"icon": "🌷", "color": "#F9B4AB", "description": "Gentle tulip gradients, modern yet cozy."},
    "Lotus":      {"icon": "🪷", "color": "#C1E3E1", "description": "Lotus serenity, aqua pastel balance."},
    "Lavender":   {"icon": "💜", "color": "#C4B5FD", "description": "Lavender mist, calm regulatory focus."},
    "Peony":      {"icon": "🌸", "color": "#FAD4E1", "description": "Peony blush, soft and inviting."},
    "Sunflower":  {"icon": "🌻", "color": "#FDE68A", "description": "Bright sunflower, optimistic review mood."},
    "Camellia":   {"icon": "🌺", "color": "#FBB6CE", "description": "Camellia pink, elegant and clear."},
    "Daisy":      {"icon": "🌼", "color": "#FFF7CC", "description": "Daisy light, high readability."},
    "Hydrangea":  {"icon": "🩵", "color": "#BFDBFE", "description": "Hydrangea blue, cool and analytical."},
    "Orchid":     {"icon": "🪻", "color": "#E9D5FF", "description": "Orchid lilac, refined clinical tone."},
    "Magnolia":   {"icon": "🤍", "color": "#F3E8FF", "description": "Magnolia white-violet, calm authority."},
    "Iris":       {"icon": "🪻", "color": "#A5B4FC", "description": "Iris blue-violet, sharp and focused."},
    "Poppy":      {"icon": "🌺", "color": "#FDBA74", "description": "Poppy orange, clear highlights."},
    "Anemone":    {"icon": "🌸", "color": "#FDE2E4", "description": "Anemone blush, soft contrast."},
    "Cornflower": {"icon": "💠", "color": "#BFDBFE", "description": "Cornflower blue, structured clarity."},
    "Heather":    {"icon": "💜", "color": "#E5E7EB", "description": "Heather grey-lilac, understated calm."},
    "Edelweiss":  {"icon": "🤍", "color": "#E5F0FF", "description": "Edelweiss alpine white-blue, Nordic crisp."},
    "Marigold":   {"icon": "🧡", "color": "#FED7AA", "description": "Marigold apricot, gentle emphasis."},
    "Bluebell":   {"icon": "🔔", "color": "#C7D2FE", "description": "Bluebell periwinkle, quiet confidence."},
}

REVIEW_CONTEXT_STYLES = {
    "General 510(k)": {
        "icon": "📁",
        "description": "一般 510(k) 傳統醫療器材審查情境",
        "color": "#2B6CB0",
    },
    "Orthopedic": {
        "icon": "🦴",
        "description": "骨科植入物與器材審查情境",
        "color": "#805AD5",
    },
    "Cardiovascular": {
        "icon": "❤️",
        "description": "心血管裝置與支架審查情境",
        "color": "#E53E3E",
    },
    "Radiology": {
        "icon": "🩻",
        "description": "影像診斷設備與 AI 讀片輔助審查情境",
        "color": "#3182CE",
    },
    "In Vitro Diagnostic": {
        "icon": "🧪",
        "description": "體外診斷 (IVD) 試劑與儀器審查情境",
        "color": "#38A169",
    },
    "Digital Health": {
        "icon": "📱",
        "description": "數位健康、SaMD 與遠距監測系統審查情境",
        "color": "#D53F8C",
    },
    "Surgical": {
        "icon": "🔪",
        "description": "手術器械與能量設備審查情境",
        "color": "#DD6B20",
    },
    "Dental": {
        "icon": "🦷",
        "description": "牙科裝置與材料審查情境",
        "color": "#319795",
    },
    "Anesthesiology": {
        "icon": "💤",
        "description": "麻醉與呼吸治療設備審查情境",
        "color": "#4A5568",
    },
    "Combination Product": {
        "icon": "💊",
        "description": "藥械組合產品與邊界產品審查情境",
        "color": "#B83280",
    },
}

TRANSLATIONS = {
    "en": {
        "title": "FDA 510(k) Multi-Agent Review Studio",
        "subtitle": "Role: Professional Regulatory AI Orchestrator",
        "theme": "UI Theme",
        "language": "Language",
        "art_style": "Review Context Style",
        "health": "Compliance Health",
        "mana": "AI Resource Capacity",
        "experience": "Case Experience",
        "api_keys": "API Keys",
        "input": "Case Inputs",
        "pipeline": "Review Pipelines",
        "smart_replace": "Smart Editing",
        "notes": "AI Note Keeper",
        "dashboard": "Dashboard",
        "run": "Run Pipeline",
        "level": "Maturity Level",
        "quest_log": "Case Log",
        "achievements": "Milestones",
        "ocr": "Submission OCR Studio",
    },
    "zh": {
        "title": "FDA 510(k) 多代理審查工作室",
        "subtitle": "專業角色：FDA 醫療器材 510(k) 審查協作代理系統",
        "theme": "介面主題",
        "language": "語言",
        "art_style": "審查情境風格",
        "health": "合規健康度",
        "mana": "AI 資源容量",
        "experience": "案件經驗值",
        "api_keys": "API 金鑰",
        "input": "案件輸入",
        "pipeline": "審查流程",
        "smart_replace": "智能編輯",
        "notes": "AI 筆記助手",
        "dashboard": "儀表板",
        "run": "執行流程",
        "level": "審查成熟度等級",
        "quest_log": "案件紀錄",
        "achievements": "重要里程碑",
        "ocr": "送件 OCR 工作室",
    }
}

# -----------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "theme": "dark",
        "language": "zh",
        "art_style": "General 510(k)",
        "flower_style": "Edelweiss",
        "player_level": 1,
        "health": 100,
        "mana": 100,
        "experience": 0,
        "quests_completed": 0,
        "achievements": [],
        "combat_log": [],
        "template": "## 案件模板\n\n在此撰寫或貼上 510(k) 案件相關模板內容...",
        "observations": "在此新增臨床、風險或技術觀察備註...",
        "pipeline_history": [],
        "note_raw_text": "",
        "note_markdown": "",
        "note_formatted": "",
        "note_keywords_output": "",
        "note_entities_json_data": [],
        "note_mindmap_json_text": "",
        "note_wordgraph_json_text": "",
        "note_chat_history": [],
        # OCR Studio state
        "ocr_files": [],              # list of per-file dicts
        "ocr_global_keywords": "510(k), substantial equivalence, risk, performance testing, adverse event, indication, predicate device, 臨床, 風險, 性能測試, 適應症",
        "combined_markdown": "",
        "combined_entities": [],
        "combined_qa_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------

@st.cache_data
def load_agents_config(path: str = "agents.yaml") -> Dict[str, Any]:
    """Load agents configuration from YAML file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"agents": [], "pipelines": []}

def get_translation(key: str) -> str:
    """Get translated text based on current language"""
    lang = st.session_state.get("language", "zh")
    return TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)

def apply_custom_css():
    """Apply Nordic-style custom CSS with flower accents"""
    theme_key = st.session_state.get("theme", "dark")
    style_key = st.session_state.get("art_style", "General 510(k)")
    flower_key = st.session_state.get("flower_style", "Edelweiss")

    colors = FDA_THEMES[theme_key]
    context_color = REVIEW_CONTEXT_STYLES.get(
        style_key, REVIEW_CONTEXT_STYLES["General 510(k)"]
    )["color"]
    flower_style = FLOWER_STYLES.get(flower_key, FLOWER_STYLES["Edelweiss"])
    flower_color = flower_style["color"]

    # Accent is based on flower + coral highlight
    accent_color = flower_color
    coral = colors["accent"]

    css = f"""
    <style>
    /* Main app container */
    .stApp {{
        background: radial-gradient(circle at top left, {flower_color} 0, {colors['background']} 50%);
        color: {colors['text']};
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif;
    }}

    /* Headers with subtle underline accent */
    h1, h2, h3 {{
        color: {colors['primary']};
        border-bottom: 2px solid rgba(148, 163, 184, 0.35);
        padding-bottom: 4px;
        letter-spacing: 0.02em;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {accent_color}, {context_color});
        color: #111827;
        border-radius: 999px;
        padding: 0.5rem 1.2rem;
        border: 1px solid rgba(15, 23, 42, 0.12);
        font-weight: 600;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.12);
        transition: all 0.18s ease-out;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.22);
        opacity: 0.96;
    }}

    /* Status bar container */
    .status-bar {{
        background: linear-gradient(90deg, rgba(15,23,42,0.06), transparent);
        border-radius: 999px;
        padding: 0.25rem 0.6rem;
        margin: 0.25rem 0;
    }}

    /* Card style */
    .review-card {{
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 14px 18px;
        margin: 6px 0;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.16);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.25rem;
        background-color: rgba(15,23,42,0.04);
        border-radius: 999px;
        padding: 0.2rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 999px;
        font-weight: 600;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {accent_color}, {context_color});
        color: #111827;
    }}

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: rgba(15,23,42,0.02);
        border-radius: 0.75rem;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.98));
        color: #E5E7EB !important;
        border-right: 1px solid rgba(148, 163, 184, 0.4);
    }}
    section[data-testid="stSidebar"] * {{
        color: #E5E7EB !important;
    }}

    /* Progress bars */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {accent_color}, {context_color});
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background: rgba(15,23,42,0.2);
        color: #E5E7EB;
        border-radius: 999px;
        font-weight: 600;
    }}

    /* Coral keyword highlight demo */
    .coral-keyword {{
        color: {coral};
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def update_player_stats(action: str):
    """
    Update abstracted 'player' stats, re-interpreted as review metrics:
    - level: 審查成熟度等級
    - health: 合規健康度
    - mana: AI 資源容量
    """
    if action == "quest_complete":
        st.session_state.experience += 10
        st.session_state.quests_completed += 1
        if st.session_state.experience >= st.session_state.player_level * 50:
            st.session_state.player_level += 1
            st.session_state.experience = 0
            st.toast(f"🎯 審查成熟度提升！目前等級：{st.session_state.player_level}")
    elif action == "use_mana":
        st.session_state.mana = max(0, st.session_state.mana - 20)
    elif action == "regenerate":
        st.session_state.mana = min(100, st.session_state.mana + 10)
        st.session_state.health = min(100, st.session_state.health + 5)

def add_combat_log(message: str, message_type: str = "info"):
    """Add entry to review activity log"""
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "spell": "🧠",
    }
    log_entry = {
        "icon": icons.get(message_type, "ℹ️"),
        "message": message,
        "timestamp": st.session_state.get("quests_completed", 0),
    }
    if "combat_log" not in st.session_state:
        st.session_state.combat_log = []
    st.session_state.combat_log.append(log_entry)
    if len(st.session_state.combat_log) > 200:
        st.session_state.combat_log.pop(0)

# -----------------------------------------------------------
# API Key Management
# -----------------------------------------------------------

def get_api_key_from_env_or_ui(
    provider_name: str,
    env_var: str,
    session_key: str,
    label: str,
) -> Optional[str]:
    """Get API key from environment or user input (do not echo env key)"""
    env_val = os.getenv(env_var)
    if env_val:
        st.caption(f"🔑 {label}: 已從環境變數載入")
        st.session_state[session_key] = env_val
        return env_val

    key = st.text_input(
        label,
        value=st.session_state.get(session_key, ""),
        type="password",
    )
    if key:
        st.session_state[session_key] = key
        st.caption(f"🔑 {label} 已暫存於工作階段")
        return key
    return None

# -----------------------------------------------------------
# LLM Call Router (OpenAI, Gemini, Grok via xai_sdk, Anthropic)
# -----------------------------------------------------------

def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Route LLM calls to appropriate provider"""
    provider = provider.lower().strip()

    add_combat_log(f"呼叫 {provider} 模型：{model}", "spell")
    update_player_stats("use_mana")

    if provider == "openai":
        api_key = st.session_state.get("openai_api_key")
        if not api_key:
            raise RuntimeError("OpenAI API key is not set.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        api_key = st.session_state.get("gemini_api_key")
        if not api_key:
            raise RuntimeError("Gemini API key is not set.")
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(
            system_prompt + "\n\nUSER MESSAGE:\n" + user_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return resp.text

    elif provider == "xai":
        # Grok via xai_sdk (per official sample)
        api_key = st.session_state.get("xai_api_key")
        if not api_key:
            raise RuntimeError("xAI (Grok) API key is not set.")
        client = XaiClient(api_key=api_key, timeout=3600)
        chat = client.chat.create(model=model)
        chat.append(xai_system(system_prompt))
        chat.append(xai_user(user_prompt))
        response = chat.sample()
        # response.content is typically a string
        return getattr(response, "content", str(response))

    elif provider == "anthropic":
        api_key = st.session_state.get("anthropic_api_key")
        if not api_key:
            raise RuntimeError("Anthropic API key is not set.")
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if resp.content and len(resp.content) > 0:
            block = resp.content[0]
            if hasattr(block, "text"):
                return block.text
        return json.dumps(resp.model_dump(), indent=2)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

def run_agent(
    agent_cfg: Dict[str, Any],
    user_prompt: str,
    override_provider: Optional[str] = None,
    override_model: Optional[str] = None,
    override_system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Run a single configured agent"""
    provider = override_provider or agent_cfg.get("provider", "openai")
    model = override_model or agent_cfg.get("default_model", "gpt-4o-mini")
    system_prompt = override_system_prompt or agent_cfg.get("system_prompt", "")
    return call_llm(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

# -----------------------------------------------------------
# Status Indicators
# -----------------------------------------------------------

def render_status_indicators():
    """Render review status indicators (WOW gauges)"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"**{get_translation('level')}**")
        st.metric(label="", value=st.session_state.player_level)

    with col2:
        st.markdown(f"**{get_translation('health')}**")
        st.progress(st.session_state.health / 100)
        st.caption(f"{st.session_state.health}/100")

    with col3:
        st.markdown(f"**{get_translation('mana')}**")
        st.progress(st.session_state.mana / 100)
        st.caption(f"{st.session_state.mana}/100")

    with col4:
        st.markdown(f"**{get_translation('experience')}**")
        max_xp = st.session_state.player_level * 50
        st.progress(st.session_state.experience / max_xp)
        st.caption(f"{st.session_state.experience}/{max_xp}")

def render_activity_log():
    """Render review activity log"""
    st.markdown("### 📑 活動紀錄")
    with st.expander("檢視近期動作", expanded=False):
        if st.session_state.combat_log:
            for entry in reversed(st.session_state.combat_log[-40:]):
                st.markdown(f"{entry['icon']} {entry['message']}")
        else:
            st.info("目前尚無活動紀錄")

# -----------------------------------------------------------
# Review Context Selector
# -----------------------------------------------------------

def render_review_context_selector():
    """Render interactive review context selector"""
    st.markdown("### 🏥 審查情境選擇器")

    cols = st.columns(5)
    styles = list(REVIEW_CONTEXT_STYLES.keys())

    for idx, style in enumerate(styles):
        with cols[idx % 5]:
            style_data = REVIEW_CONTEXT_STYLES[style]
            button_label = f"{style_data['icon']} {style}"
            if st.button(
                button_label,
                key=f"style_{style}",
                help=style_data["description"],
                use_container_width=True
            ):
                st.session_state.art_style = style
                add_combat_log(f"切換審查情境為：{style}", "success")
                st.rerun()

    current_style = st.session_state.get("art_style", "General 510(k)")
    style_data = REVIEW_CONTEXT_STYLES[current_style]
    st.markdown(
        f"<div class='review-card' style='text-align: center; "
        f"background: linear-gradient(135deg, {style_data['color']}33, transparent);'>"
        f"<h3>{style_data['icon']} 目前情境：{current_style}</h3>"
        f"<p>{style_data['description']}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------
# Enhanced Sidebar (incl. Magic Flower Wheel)
# -----------------------------------------------------------

def render_enhanced_sidebar(config: Dict[str, Any]):
    """Render Nordic-themed sidebar with controls"""
    st.sidebar.markdown(f"# {get_translation('title')}")
    st.sidebar.markdown(f"*{get_translation('subtitle')}*")
    st.sidebar.markdown("---")

    # Theme and Language Selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        theme = st.selectbox(
            get_translation("theme"),
            ["light", "dark"],
            index=1 if st.session_state.theme == "dark" else 0,
            key="theme_selector"
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()

    with col2:
        lang = st.selectbox(
            get_translation("language"),
            ["zh", "en"],
            index=0 if st.session_state.language == "zh" else 1,
            key="lang_selector"
        )
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()

    # Magic Flower Wheel
    st.sidebar.markdown("### 🌸 Magic Flower Wheel")
    flower_names = list(FLOWER_STYLES.keys())
    flower_labels = [
        f"{FLOWER_STYLES[name]['icon']} {name}" for name in flower_names
    ]
    current_index = flower_names.index(st.session_state.get("flower_style", "Edelweiss"))
    idx = st.sidebar.select_slider(
        "選擇 UI 花卉風格",
        options=list(range(len(flower_names))),
        value=current_index,
        format_func=lambda i: flower_labels[i],
        key="flower_wheel",
    )
    chosen_flower = flower_names[idx]
    if chosen_flower != st.session_state.flower_style:
        st.session_state.flower_style = chosen_flower
        add_combat_log(f"切換花卉風格為：{chosen_flower}", "info")
        st.rerun()

    st.sidebar.markdown("---")

    # Review Status
    st.sidebar.markdown("### 📊 審查狀態總覽")
    render_status_indicators()
    st.sidebar.markdown("---")

    # API Keys
    st.sidebar.markdown(f"### 🔑 {get_translation('api_keys')}")
    with st.sidebar.expander("設定 API 金鑰"):
        get_api_key_from_env_or_ui(
            "OpenAI", "OPENAI_API_KEY", "openai_api_key", "OpenAI API Key"
        )
        get_api_key_from_env_or_ui(
            "Gemini", "GEMINI_API_KEY", "gemini_api_key", "Gemini API Key"
        )
        get_api_key_from_env_or_ui(
            "xAI", "XAI_API_KEY", "xai_api_key", "xAI (Grok) API Key"
        )
        get_api_key_from_env_or_ui(
            "Anthropic", "ANTHROPIC_API_KEY", "anthropic_api_key", "Anthropic API Key"
        )

    st.sidebar.markdown("---")

    # Model Settings
    st.sidebar.markdown("### ⚙️ 模型呼叫設定")

    provider = st.sidebar.selectbox(
        "模型供應商",
        ["openai", "gemini", "xai", "anthropic"],
        key="default_provider",
    )

    provider_models = {
        "openai": ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"],
        "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
        "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
    }

    st.sidebar.selectbox(
        "模型版本",
        provider_models[provider],
        key="default_model",
    )

    st.sidebar.slider(
        "最大輸出 Token 數",
        64, 4096, 1024, 64,
        key="default_max_tokens",
    )

    st.sidebar.slider(
        "溫度（隨機性）",
        0.0, 1.0, 0.7, 0.05,
        key="default_temperature",
    )

    st.sidebar.markdown("---")

    # Case Log
    st.sidebar.markdown(f"### 📁 {get_translation('quest_log')}")
    st.sidebar.metric("已完成案件數", st.session_state.quests_completed)

    if st.sidebar.button("🔄 恢復資源"):
        update_player_stats("regenerate")
        add_combat_log("AI 資源與合規健康度已適度恢復", "success")
        st.rerun()

# -----------------------------------------------------------
# Input Tab
# -----------------------------------------------------------

def render_input_tab():
    """Render case input tab"""
    st.markdown(f"## 📝 {get_translation('input')}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.text_area(
            "📄 510(k) 案件模板 / 主要內容",
            key="template",
            height=260,
            help="例如：設備描述、適應症說明、實質等同性比較、風險管理摘要等"
        )

        st.text_area(
            "🔍 審查觀察與備註",
            key="observations",
            height=260,
            help="記錄審查歷程中的疑問、風險點、需追問之資料等"
        )

    with col2:
        render_activity_log()

        st.markdown("### ⚡ 快速動作")
        if st.button("💾 儲存當前輸入", use_container_width=True):
            add_combat_log("目前案件輸入已儲存（暫存於 session）", "success")
            st.success("已暫存目前內容。")

        if st.button("🧹 清空欄位", use_container_width=True):
            st.session_state.template = ""
            st.session_state.observations = ""
            add_combat_log("案件輸入欄位已清空", "info")
            st.rerun()

# -----------------------------------------------------------
# Pipeline Tab
# -----------------------------------------------------------

def render_pipeline_tab(config: Dict[str, Any]):
    """Render multi-agent 510(k) review pipeline tab"""
    st.markdown(f"## 🔄 {get_translation('pipeline')}")

    if not config or "pipelines" not in config:
        st.warning("⚠️ agents.yaml 中未找到任何審查流程 (pipelines) 設定。")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        pipeline_options = {p["name"]: p for p in config["pipelines"]}
        selected_name = st.selectbox("🔎 選擇審查流程", list(pipeline_options.keys()))
        pipeline = pipeline_options[selected_name]

        st.markdown(f"**流程 ID：** `{pipeline['id']}`")
        st.markdown(f"**說明：** {pipeline.get('description', '')}")

        st.markdown("### 📂 流程步驟")
        for idx, step in enumerate(pipeline["steps"], start=1):
            st.markdown(f"- 第 {idx} 步：`{step['agent_id']}`")

        st.markdown("---")

        override_prompt = st.text_area(
            "📌 其他補充說明 / 特別指示",
            "例如：此案件風險偏高，請提高風險評估與法規比對的嚴謹度。",
            height=120,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            provider = st.selectbox(
                "模型供應商覆寫（選填）",
                ["(使用預設)", "openai", "gemini", "xai", "anthropic"],
            )
        with col_b:
            model_override = st.text_input("模型名稱覆寫（選填）", "")

        if st.button(f"▶️ {get_translation('run')}", use_container_width=True):
            if st.session_state.mana < 20:
                st.error("❌ AI 資源不足，請先按左側『恢復資源』。")
                return

            template = st.session_state.get("template", "")
            observations = st.session_state.get("observations", "")
            current_input = (
                "【510(k) 案件輸入】\n"
                f"{template}\n\n"
                "【審查觀察與備註】\n"
                f"{observations}\n\n"
                "【額外指示】\n"
                f"{override_prompt}"
            )

            outputs = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, step in enumerate(pipeline["steps"]):
                agent_id = step["agent_id"]
                agent_cfg = next((a for a in config["agents"] if a["id"] == agent_id), None)

                if not agent_cfg:
                    st.error(f"❌ 找不到代理設定：{agent_id}")
                    return

                progress = (idx + 1) / len(pipeline["steps"])
                progress_bar.progress(progress)
                status_text.text(f"執行代理：{agent_cfg['name']} ...")

                try:
                    result = run_agent(
                        agent_cfg=agent_cfg,
                        user_prompt=current_input,
                        override_provider=None if provider.startswith("(") else provider,
                        override_model=model_override or None,
                        max_tokens=st.session_state.get("default_max_tokens", 1024),
                        temperature=st.session_state.get("default_temperature", 0.7),
                    )
                    outputs.append({"agent_id": agent_id, "output": result})
                    current_input = result
                    update_player_stats("regenerate")
                except Exception as e:
                    st.error(f"❌ 模型呼叫失敗：{e}")
                    add_combat_log(f"審查流程在代理 {agent_id} 中斷。", "error")
                    return

            progress_bar.progress(1.0)
            status_text.text("✅ 審查流程完成。")

            st.success("🎉 審查流程已成功完成並產出結果。")
            update_player_stats("quest_complete")
            add_combat_log(f"已完成審查流程：{selected_name}", "success")

            st.session_state.pipeline_history.append(outputs)

            st.markdown("### 📘 流程輸出結果")
            for idx, item in enumerate(outputs, start=1):
                with st.expander(f"步驟 {idx} – 代理 `{item['agent_id']}`", expanded=(idx == len(outputs))):
                    st.markdown(item["output"])

    with col2:
        render_activity_log()
        st.markdown("### 📊 流程統計")
        st.metric("已執行流程次數", len(st.session_state.pipeline_history))

# -----------------------------------------------------------
# Smart Replace Tab (placeholder, original feature kept)
# -----------------------------------------------------------

def render_smart_replace_tab():
    """Placeholder for smart editing (original feature kept)"""
    st.markdown(f"## ✨ {get_translation('smart_replace')}")
    st.info("此區可整合既有文字改寫與比對工具（保留原始設計空間）。")

# -----------------------------------------------------------
# AI Note Keeper: helpers
# -----------------------------------------------------------

def highlight_keywords_in_text(text: str, keywords: List[str], color: str) -> str:
    """Highlight given keywords in text using HTML span with specified color"""
    if not text or not keywords:
        return text
    result = text
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        result = pattern.sub(
            lambda m: f"<span style='color:{color}'>{m.group(0)}</span>",
            result,
        )
    return result

# -----------------------------------------------------------
# AI Note Keeper Tab
# -----------------------------------------------------------

def render_notes_tab():
    """Render AI Note Keeper with multiple AI tools"""
    st.markdown(f"## 📔 {get_translation('notes')}")
    st.info(
        "將 510(k) 或醫療器材相關文字貼上，利用多代理 AI 進行 **Markdown 結構化、格式優化、關鍵字標示、實體抽取、心智圖與詞彙關聯圖**。"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.text_area(
            "🧾 原始文本貼上區",
            key="note_raw_text",
            height=260,
            help="例如：510(k) 摘要、風險管理報告片段、技術說明、回覆 FDA 問答等",
        )
        if st.button("📄 轉換為 Markdown 結構", use_container_width=True):
            if not st.session_state.note_raw_text.strip():
                st.warning("請先貼上原始文本。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "You are a professional FDA 510(k) regulatory note architect.\n"
                        "Goal: Convert the raw text into a **lossless, well-structured Markdown document**.\n"
                        "- Preserve all original factual content (no deletions, no hallucinations).\n"
                        "- You MAY:\n"
                        "  - Split or merge paragraphs for readability.\n"
                        "  - Introduce hierarchical headings (##, ###) that reflect regulatory logic (device, indications, SE, testing, risk, clinical, labeling, etc.).\n"
                        "  - Use bullet / numbered lists where appropriate.\n"
                        "- You MUST NOT:\n"
                        "  - Omit meaningful information.\n"
                        "  - Add new data not present in the source.\n"
                        "Output: Markdown only. Do not add any explanation outside the Markdown content."
                    )
                    user_prompt = st.session_state.note_raw_text
                    md = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=st.session_state.get("default_max_tokens", 1024),
                        temperature=0.1,
                    )
                    st.session_state.note_markdown = md
                    add_combat_log("完成原始文本的 Markdown 結構化。", "success")
                except Exception as e:
                    st.error(f"轉換為 Markdown 時發生錯誤：{e}")

    with col2:
        st.markdown("### 📑 Markdown 預覽")
        if st.session_state.note_markdown:
            st.markdown(st.session_state.note_markdown)
        else:
            st.caption("尚未產生 Markdown，請先於左側貼上文字並按下「轉換為 Markdown」。")

    st.markdown("---")

    tab_fmt, tab_kw, tab_ent, tab_mind, tab_word = st.tabs(
        ["AI 格式優化", "AI 關鍵字標示", "AI 實體抽取", "AI 心智圖", "AI 詞彙關聯圖"]
    )

    # --- AI Formatting ---
    with tab_fmt:
        st.markdown("### 🧹 AI 格式優化（保留原文，強化結構與重點）")
        st.caption(
            "說明：在**不刪除任何原文句子**的前提下，重新編排段落與標題，並用珊瑚色標註重要術語。"
        )
        if st.button("⚙️ 執行 AI 格式優化", use_container_width=True, key="btn_ai_format"):
            base_text = st.session_state.note_markdown or st.session_state.note_raw_text
            if not base_text.strip():
                st.warning("請先貼上文字並至少完成一次 Markdown 轉換。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "You are an expert editor for FDA 510(k) submissions.\n"
                        "Task: Reformat the provided content while **preserving every original sentence**.\n"
                        "You MUST:\n"
                        "- Keep all sentences and technical terms intact (no deletion, no paraphrasing).\n"
                        "- Re-group paragraphs logically (device description, indications, SE, testing, risk, clinical, labeling, etc.).\n"
                        "- Add meaningful Markdown headings (##, ###) and lists.\n"
                        "- Wrap HIGH-VALUE regulatory/technical/clinical keywords with HTML spans:\n"
                        "  <span style=\"color:coral\">keyword</span>\n"
                        "Output: Markdown + inline HTML only. No extra commentary."
                    )
                    user_prompt = base_text
                    formatted = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=st.session_state.get("default_max_tokens", 2048),
                        temperature=0.4,
                    )
                    st.session_state.note_formatted = formatted
                    add_combat_log("完成 AI 格式優化與重點標示。", "success")
                except Exception as e:
                    st.error(f"AI 格式優化失敗：{e}")

        if st.session_state.note_formatted:
            st.markdown("#### 格式優化結果")
            st.markdown(st.session_state.note_formatted, unsafe_allow_html=True)

    # --- AI Keywords ---
    with tab_kw:
        st.markdown("### 🎯 AI 關鍵字標示")
        st.caption("可自訂欲強調的關鍵詞與顏色，在 Markdown 內容中自動高亮。")

        kw_text = st.text_input(
            "輸入欲標示的關鍵字（以逗號分隔）",
            value="510(k), substantial equivalence, risk management, performance testing, FDA",
        )
        kw_color = st.color_picker("關鍵字顏色", value="#FF7F50")

        if st.button("🔍 標示關鍵字", use_container_width=True):
            base_text = (
                st.session_state.note_formatted
                or st.session_state.note_markdown
                or st.session_state.note_raw_text
            )
            if not base_text.strip():
                st.warning("尚無可處理的文本，請先產生 Markdown 或貼上文字。")
            else:
                keywords = [k for k in kw_text.split(",") if k.strip()]
                highlighted = highlight_keywords_in_text(base_text, keywords, kw_color)
                st.session_state.note_keywords_output = highlighted
                add_combat_log("完成自訂關鍵字標示。", "success")

        if st.session_state.note_keywords_output:
            st.markdown("#### 關鍵字標示結果")
            st.markdown(st.session_state.note_keywords_output, unsafe_allow_html=True)

    # --- AI Entities ---
    with tab_ent:
        st.markdown("### 🧬 AI 實體抽取（最多 20 個）")
        st.caption(
            "從文本中抽取最重要的法規、技術、臨床與風險相關實體，並產生結構化表格與 JSON。"
        )
        if st.button("📊 抽取 20 個關鍵實體", use_container_width=True):
            base_text = st.session_state.note_markdown or st.session_state.note_raw_text
            if not base_text.strip():
                st.warning("請先貼上文字並至少完成一次 Markdown 轉換。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "You are an information extraction specialist for FDA 510(k) dossiers.\n"
                        "From the provided text, identify up to 20 highest-value entities. Entities may be:\n"
                        "- regulations or standards\n"
                        "- submission sections (e.g., Indications for Use, Device Description)\n"
                        "- device modules or components\n"
                        "- risk types or hazards\n"
                        "- performance tests\n"
                        "- clinical endpoints or outcomes\n"
                        "Return **JSON only** in the form:\n"
                        "[\n"
                        "  {\"id\": 1, \"name\": \"...\", \"type\": \"regulation|section|risk|test|clinical|other\", "
                        "\"description\": \"short explanation\", \"source_snippet\": \"representative phrase from text\"},\n"
                        "  ... up to 20 entities\n"
                        "]"
                    )
                    user_prompt = base_text
                    raw = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=1024,
                        temperature=0.2,
                    )
                    raw_str = raw.strip().strip("```json").strip("```").strip()
                    entities = json.loads(raw_str)
                    if not isinstance(entities, list):
                        raise ValueError("回傳內容並非 JSON 陣列。")
                    st.session_state.note_entities_json_data = entities
                    add_combat_log("完成文本實體抽取（最多 20 個）。", "success")
                except Exception as e:
                    st.error(f"實體抽取與 JSON 解析失敗：{e}")

        if st.session_state.note_entities_json_data:
            st.markdown("#### 實體表格")
            table_md = "| id | name | type | description | source_snippet |\n"
            table_md += "|---|------|------|-------------|----------------|\n"
            for ent in st.session_state.note_entities_json_data:
                table_md += (
                    f"| {ent.get('id','')} "
                    f"| {ent.get('name','')} "
                    f"| {ent.get('type','')} "
                    f"| {ent.get('description','').replace('|','/')} "
                    f"| {ent.get('source_snippet','').replace('|','/')} |\n"
                )
            st.markdown(table_md)

            st.markdown("#### JSON 檢視")
            st.json(st.session_state.note_entities_json_data)

    # --- AI Mind-Map ---
    with tab_mind:
        st.markdown("### 🧠 AI 心智圖")
        st.caption(
            "根據文本內容自動產生節點與關係的 JSON，您可手動調整後，即時視覺化為心智圖。"
        )
        if st.button("🧠 產生心智圖 JSON", use_container_width=True):
            base_text = st.session_state.note_markdown or st.session_state.note_raw_text
            if not base_text.strip():
                st.warning("請先貼上文字並至少完成一次 Markdown 轉換。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "You are a knowledge graph designer.\n"
                        "Create a compact **mind-map JSON** from the text:\n"
                        "{\n"
                        "  \"nodes\": [\n"
                        "    {\"id\": \"NodeID\", \"label\": \"display name\", \"type\": \"device|risk|test|regulation|clinical|other\"},\n"
                        "    ... 8–15 nodes\n"
                        "  ],\n"
                        "  \"edges\": [\n"
                        "    {\"source\": \"NodeID\", \"target\": \"NodeID\", \"relation\": \"short description\"},\n"
                        "    ... 10–25 edges\n"
                        "  ]\n"
                        "}\n"
                        "Output JSON only."
                    )
                    user_prompt = base_text
                    raw = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=1024,
                        temperature=0.3,
                    )
                    raw_str = raw.strip().strip("```json").strip("```").strip()
                    st.session_state.note_mindmap_json_text = raw_str
                    add_combat_log("已產生心智圖 JSON 結構。", "success")
                except Exception as e:
                    st.error(f"心智圖 JSON 產生失敗：{e}")

        mindmap_text = st.text_area(
            "心智圖 JSON 可於此調整後重新繪製",
            value=st.session_state.note_mindmap_json_text,
            height=220,
        )
        if st.button("📈 根據 JSON 顯示心智圖", use_container_width=True):
            try:
                data = json.loads(mindmap_text)
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])
                dot = "digraph G {\nrankdir=LR;\n"
                for n in nodes:
                    nid = n.get("id", "")
                    label = n.get("label", nid)
                    dot += f"  \"{nid}\" [label=\"{label}\"];\n"
                for e in edges:
                    src = e.get("source", "")
                    tgt = e.get("target", "")
                    rel = e.get("relation", "")
                    dot += f"  \"{src}\" -> \"{tgt}\" [label=\"{rel}\"];\n"
                dot += "}"
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"解析或繪製心智圖時發生錯誤：{e}")

    # --- AI Wordgraph ---
    with tab_word:
        st.markdown("### 📚 AI 詞彙關聯圖 (Wordgraph)")
        st.caption(
            "根據文本自動分析重要術語之間的關聯，產生詞彙關聯圖 JSON 並視覺化。"
        )
        if st.button("📚 產生詞彙關聯 JSON", use_container_width=True):
            base_text = st.session_state.note_markdown or st.session_state.note_raw_text
            if not base_text.strip():
                st.warning("請先貼上文字並至少完成一次 Markdown 轉換。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "You are a text mining and terminology network expert.\n"
                        "From the text, select 10–15 key technical/regulatory/clinical terms and "
                        "build a wordgraph JSON:\n"
                        "{\n"
                        "  \"nodes\": [\n"
                        "    {\"id\": \"TermID\", \"label\": \"display name\", \"frequency\": number},\n"
                        "    ...\n"
                        "  ],\n"
                        "  \"edges\": [\n"
                        "    {\"source\": \"TermID\", \"target\": \"TermID\", \"weight\": 1-5, \"note\": \"link explanation\"},\n"
                        "    ...\n"
                        "  ]\n"
                        "}\n"
                        "Output JSON only."
                    )
                    user_prompt = base_text
                    raw = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=1024,
                        temperature=0.4,
                    )
                    raw_str = raw.strip().strip("```json").strip("```").strip()
                    st.session_state.note_wordgraph_json_text = raw_str
                    add_combat_log("已產生詞彙關聯圖 JSON 結構。", "success")
                except Exception as e:
                    st.error(f"詞彙關聯 JSON 產生失敗：{e}")

        wordgraph_text = st.text_area(
            "詞彙關聯圖 JSON 可於此調整後重新繪製",
            value=st.session_state.note_wordgraph_json_text,
            height=220,
        )
        if st.button("📊 根據 JSON 顯示詞彙關聯圖", use_container_width=True):
            try:
                data = json.loads(wordgraph_text)
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])
                dot = "graph G {\n"
                for n in nodes:
                    nid = n.get("id", "")
                    label = n.get("label", nid)
                    freq = n.get("frequency", 1)
                    size = 10 + freq * 2
                    dot += f"  \"{nid}\" [label=\"{label}\", fontsize={size}];\n"
                for e in edges:
                    src = e.get("source", "")
                    tgt = e.get("target", "")
                    w = e.get("weight", 1)
                    note = e.get("note", "")
                    penwidth = 1 + w
                    dot += (
                        f"  \"{src}\" -- \"{tgt}\" "
                        f"[label=\"{note}\", penwidth={penwidth}];\n"
                    )
                dot += "}"
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"解析或繪製詞彙關聯圖時發生錯誤：{e}")

# -----------------------------------------------------------
# Submission OCR Studio – helpers
# -----------------------------------------------------------

def parse_page_selection(pages_str: str, max_pages: int) -> List[int]:
    """
    Parse a page selection string like "1-3,5" into a sorted list of 1-based page numbers.
    """
    if not pages_str:
        return list(range(1, max_pages + 1))
    pages_str = pages_str.replace(" ", "")
    pages: List[int] = []
    for part in pages_str.split(","):
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            if not start_s.isdigit() or not end_s.isdigit():
                continue
            start, end = int(start_s), int(end_s)
            if start <= 0 or end <= 0:
                continue
            for p in range(start, end + 1):
                if 1 <= p <= max_pages:
                    pages.append(p)
        else:
            if part.isdigit():
                p = int(part)
                if 1 <= p <= max_pages:
                    pages.append(p)
    pages = sorted(set(pages))
    if not pages:
        pages = [1]
    return pages

def ensure_pdf_reader():
    if PdfReader is None:
        raise RuntimeError("PyPDF2 未安裝，無法讀取 PDF。請在環境中安裝 PyPDF2。")

def ensure_tesseract():
    if pytesseract is None or convert_from_bytes is None:
        raise RuntimeError("pytesseract 或 pdf2image 未安裝，無法執行 Python OCR。")

def get_pdf_page_count(pdf_bytes: bytes) -> int:
    ensure_pdf_reader()
    reader = PdfReader(BytesIO(pdf_bytes))
    return len(reader.pages)

def extract_pdf_text(pdf_bytes: bytes, pages: List[int]) -> str:
    """Extract textual content from specified 1-based pages using PyPDF2"""
    ensure_pdf_reader()
    reader = PdfReader(BytesIO(pdf_bytes))
    texts: List[str] = []
    for p in pages:
        if 1 <= p <= len(reader.pages):
            page = reader.pages[p - 1]
            txt = page.extract_text() or ""
            texts.append(f"\n\n--- Page {p} ---\n\n{txt}")
    return "\n".join(texts).strip()

def ocr_pdf_tesseract(pdf_bytes: bytes, pages: List[int], lang: str) -> str:
    """OCR selected pages using Tesseract (english / traditional chinese)"""
    ensure_tesseract()
    first_page, last_page = min(pages), max(pages)
    images = convert_from_bytes(pdf_bytes, first_page=first_page, last_page=last_page)
    result_chunks: List[str] = []
    for idx, img in enumerate(images, start=first_page):
        if idx in pages:
            text = pytesseract.image_to_string(img, lang=lang)
            result_chunks.append(f"\n\n--- Page {idx} ---\n\n{text}")
    return "\n".join(result_chunks).strip()

def pdf_to_base64_iframe(pdf_bytes: bytes, width: str = "100%", height: str = "600") -> str:
    """Generate an HTML iframe to preview a PDF from bytes."""
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"""
    <iframe
        src="data:application/pdf;base64,{b64}#toolbar=0"
        width="{width}"
        height="{height}"
        style="border-radius: 12px; border: 1px solid rgba(148,163,184,0.5);"
    ></iframe>
    """

# -----------------------------------------------------------
# Submission OCR Studio Tab
# -----------------------------------------------------------

ADVANCED_OCR_SYSTEM_PROMPT = """
You are an elite OCR + document reconstruction assistant specialized in FDA 510(k) submissions.
Input: noisy text extracted from PDF pages (including possible encoding issues, line breaks, hyphenations).
Your tasks:
1. Denoise and normalize:
   - Fix broken words and hyphenation at line breaks.
   - Remove obvious OCR noise (random symbols, page headers/footers if clearly repetitive).
   - Preserve all substantive regulatory, clinical, risk and technical information.
2. Reconstruct structure as Markdown:
   - Introduce clear headings (##, ###) for sections like: Device Description, Indications for Use, Substantial Equivalence, Performance Testing, Risk Management, Clinical, Labeling, etc., when they are present or inferable.
   - Use bullet/numbered lists to improve readability.
3. Coral keyword highlighting:
   - Wrap high-value domain keywords with: <span style="color:coral">keyword</span>.
   - Focus on: device name, key parameters, standards, risk terms, clinical endpoints, important performance metrics, critical regulatory references.
Constraints:
- Do NOT invent new facts.
- Do NOT omit meaningful content.
Output:
- Return **Markdown + inline HTML only**, ready to render in a viewer.
"""

def render_submission_ocr_tab():
    """Render multi-file Submission OCR Studio with PDF/TXT upload + OCR + summaries + combined QA"""
    st.markdown(f"## 📂 {get_translation('ocr')}")

    st.info(
        "此分頁可處理多個 PDF / TXT 送件資料：\n"
        "1️⃣ 選擇欲上傳檔案數量 → 2️⃣ 上傳 PDF/TXT → 3️⃣ 為每份檔案選擇頁碼與 OCR 方式\n"
        "4️⃣ 為每份檔產生 **Markdown（含珊瑚色關鍵字）** 與摘要 → 5️⃣ 對所有 OCR 文件整合抽取 20 個實體並進行提問。"
    )

    # Step 0 – global keyword highlight config
    st.markdown("### 🎯 全域關鍵字設定（適用於 Python OCR 產物）")
    st.text_input(
        "在 OCR 結果中欲以珊瑚色標示的關鍵字（逗號分隔，可中英混合）",
        key="ocr_global_keywords",
    )

    # Step 1 – user-estimated number of files
    num_files = st.number_input("預計處理的檔案數量", min_value=1, max_value=20, value=1, step=1)

    # Step 2 – upload files
    uploaded_files = st.file_uploader(
        "上傳 PDF / TXT 檔案（可多選）",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) != num_files:
            st.warning(f"目前已上傳 {len(uploaded_files)} 個檔案，與預計數量 {num_files} 不同，可視需要調整。")

        # Rebuild or update state for ocr_files
        existing_by_name = {f["filename"]: f for f in st.session_state.ocr_files}
        new_state_files: List[Dict[str, Any]] = []

        for uf in uploaded_files:
            name = uf.name
            ext = "pdf" if name.lower().endswith(".pdf") else "txt"
            content = uf.getvalue()

            prev = existing_by_name.get(name, {})
            entry = {
                "filename": name,
                "ext": ext,
                "bytes": content,
                "num_pages": prev.get("num_pages"),
                "markdown": prev.get("markdown", ""),
                "summary": prev.get("summary", ""),
            }
            if ext == "pdf" and entry["num_pages"] is None:
                try:
                    entry["num_pages"] = get_pdf_page_count(content)
                except Exception as e:
                    st.error(f"無法讀取 PDF 頁數：{name} - {e}")
                    entry["num_pages"] = 0

            new_state_files.append(entry)

        st.session_state.ocr_files = new_state_files

        st.markdown("### 📚 檔案設定與 OCR 選項")

        for idx, file_info in enumerate(st.session_state.ocr_files):
            fname = file_info["filename"]
            ext = file_info["ext"]
            key_prefix = f"ocr_{idx}"

            with st.expander(f"{idx+1}. {fname}", expanded=True):
                if ext == "pdf" and file_info.get("bytes"):
                    st.markdown("#### 📖 PDF 預覽")
                    try:
                        iframe_html = pdf_to_base64_iframe(file_info["bytes"], height="480")
                        st.components.v1.html(iframe_html, height=500, scrolling=True)
                    except Exception:
                        st.info("瀏覽器或環境限制，PDF 內嵌預覽失敗，可改用下載檢視。")
                        st.download_button("下載 PDF", data=file_info["bytes"], file_name=fname)

                    num_pages = file_info.get("num_pages", 0)
                    st.markdown(f"- 總頁數：**{num_pages}**")

                    pages_default = st.session_state.get(f"{key_prefix}_pages_str", "1-3" if num_pages >= 3 else "1")
                    pages_str = st.text_input(
                        "欲 OCR 的頁碼（例如：1-3,5）",
                        value=pages_default,
                        key=f"{key_prefix}_pages_str",
                    )

                    ocr_backend = st.radio(
                        "OCR 方式",
                        ["Python OCR (Tesseract)", "LLM-based OCR (多模型支援)"],
                        key=f"{key_prefix}_backend",
                    )

                    if ocr_backend.startswith("Python"):
                        lang_label = st.selectbox(
                            "Tesseract 語言",
                            ["English", "Traditional Chinese", "English + Traditional Chinese"],
                            key=f"{key_prefix}_lang",
                        )
                        if lang_label == "English":
                            lang_code = "eng"
                        elif lang_label == "Traditional Chinese":
                            lang_code = "chi_tra"
                        else:
                            lang_code = "eng+chi_tra"
                    else:
                        lang_code = None  # not used

                        st.markdown("##### LLM OCR 設定")
                        col_l1, col_l2 = st.columns(2)
                        with col_l1:
                            llm_provider = st.selectbox(
                                "供應商",
                                ["openai", "gemini", "xai", "anthropic"],
                                key=f"{key_prefix}_llm_provider",
                            )
                        with col_l2:
                            provider_models = {
                                "openai": ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"],
                                "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
                                "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
                                "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
                            }
                            llm_model = st.selectbox(
                                "模型",
                                provider_models[llm_provider],
                                key=f"{key_prefix}_llm_model",
                            )

                        llm_max_tokens = st.number_input(
                            "最大輸出 tokens（OCR/清理用）",
                            min_value=128, max_value=4096, value=1500, step=64,
                            key=f"{key_prefix}_llm_max_tokens",
                        )

                        llm_temp = st.slider(
                            "溫度（OCR/清理）",
                            0.0, 1.0, 0.2, 0.05,
                            key=f"{key_prefix}_llm_temp",
                        )

                        default_ocr_prompt = ADVANCED_OCR_SYSTEM_PROMPT.strip()
                        llm_system_prompt = st.text_area(
                            "進階 OCR 系統提示（可微調）",
                            value=default_ocr_prompt,
                            height=180,
                            key=f"{key_prefix}_llm_system_prompt",
                        )

                    if st.button("▶️ 執行此檔 OCR（轉 Markdown＋珊瑚色關鍵字）", key=f"{key_prefix}_run"):
                        if num_pages <= 0:
                            st.error("無法取得 PDF 頁數，請確認檔案是否損毀。")
                        else:
                            try:
                                pages = parse_page_selection(pages_str, num_pages)

                                if ocr_backend.startswith("Python"):
                                    # Python OCR path
                                    if "+" in (lang_code or ""):
                                        langs = lang_code.split("+")
                                        text_agg = ""
                                        for l in langs:
                                            text_agg += ocr_pdf_tesseract(file_info["bytes"], pages, l)
                                        raw_text = text_agg
                                    else:
                                        raw_text = ocr_pdf_tesseract(file_info["bytes"], pages, lang_code)

                                    # Simple Markdown wrap + keyword highlight
                                    markdown_raw = raw_text or ""
                                    kw_str = st.session_state.get("ocr_global_keywords", "")
                                    keywords = [k for k in kw_str.split(",") if k.strip()]
                                    markdown = highlight_keywords_in_text(
                                        markdown_raw, keywords, "#FF7F50"
                                    )
                                    file_info["markdown"] = markdown
                                    st.session_state.ocr_files[idx] = file_info
                                    add_combat_log(f"{fname} 已完成 Python OCR。", "success")

                                else:
                                    # LLM-based OCR / cleanup
                                    text_extracted = extract_pdf_text(file_info["bytes"], pages)
                                    llm_provider = st.session_state.get(f"{key_prefix}_llm_provider", "openai")
                                    llm_model = st.session_state.get(f"{key_prefix}_llm_model", "gpt-4o-mini")
                                    llm_max_tokens = st.session_state.get(f"{key_prefix}_llm_max_tokens", 1500)
                                    llm_temp = st.session_state.get(f"{key_prefix}_llm_temp", 0.2)
                                    llm_system = st.session_state.get(
                                        f"{key_prefix}_llm_system_prompt",
                                        ADVANCED_OCR_SYSTEM_PROMPT.strip(),
                                    )
                                    markdown = call_llm(
                                        provider=llm_provider,
                                        model=llm_model,
                                        system_prompt=llm_system,
                                        user_prompt=text_extracted,
                                        max_tokens=int(llm_max_tokens),
                                        temperature=float(llm_temp),
                                    )
                                    file_info["markdown"] = markdown
                                    st.session_state.ocr_files[idx] = file_info
                                    add_combat_log(f"{fname} 已完成 LLM OCR / 清理。", "success")

                                st.success("✅ OCR 完成，已轉換為 Markdown。")
                                st.markdown("##### OCR Markdown 預覽")
                                st.markdown(file_info["markdown"], unsafe_allow_html=True)

                            except Exception as e:
                                st.error(f"OCR 過程發生錯誤：{e}")

                else:
                    # TXT file
                    text_content = file_info["bytes"].decode("utf-8", errors="ignore")
                    st.markdown("#### 📄 TXT 內容預覽（前 800 字）")
                    st.code(text_content[:800] + ("..." if len(text_content) > 800 else ""))

                    if st.button("▶️ 將 TXT 轉為 Markdown（含珊瑚色關鍵字）", key=f"{key_prefix}_txt_to_md"):
                        try:
                            # Use a light LLM formatting for TXT
                            provider = st.session_state.get("default_provider", "openai")
                            model = st.session_state.get("default_model", "gpt-4o-mini")
                            system_prompt = ADVANCED_OCR_SYSTEM_PROMPT.strip()
                            markdown = call_llm(
                                provider=provider,
                                model=model,
                                system_prompt=system_prompt,
                                user_prompt=text_content,
                                max_tokens=2000,
                                temperature=0.2,
                            )
                            file_info["markdown"] = markdown
                            st.session_state.ocr_files[idx] = file_info
                            add_combat_log(f"{fname} TXT 已轉換為結構化 Markdown。", "success")
                            st.success("✅ TXT 已轉為 Markdown。")
                            st.markdown("##### Markdown 預覽")
                            st.markdown(markdown, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"TXT 轉 Markdown 失敗：{e}")

                # Per-file summary, if markdown ready
                if file_info.get("markdown"):
                    st.markdown("#### 🧾 檔案摘要（可自訂提示與模型）")
                    sum_provider = st.selectbox(
                        "摘要用模型供應商",
                        ["openai", "gemini", "xai", "anthropic"],
                        key=f"{key_prefix}_sum_provider",
                    )
                    provider_models = {
                        "openai": ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"],
                        "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
                        "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
                        "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
                    }
                    sum_model = st.selectbox(
                        "摘要模型",
                        provider_models[sum_provider],
                        key=f"{key_prefix}_sum_model",
                    )
                    sum_tokens = st.number_input(
                        "最大摘要 tokens",
                        min_value=128, max_value=4096, value=800, step=64,
                        key=f"{key_prefix}_sum_tokens",
                    )

                    default_sum_prompt = (
                        "You are a senior FDA 510(k) reviewer.\n"
                        "Summarize this single document into a **concise yet comprehensive regulatory briefing**.\n"
                        "Include:\n"
                        "- Device overview and intended use\n"
                        "- Indications for Use (if present)\n"
                        "- Key technological characteristics\n"
                        "- Substantial equivalence argument highlight\n"
                        "- Major performance tests (bench, biocompatibility, EMC, software, etc.)\n"
                        "- Main risks and mitigations\n"
                        "- Any clinical data or rationale\n"
                        "Return Markdown with clear headings and bullet lists. Do not hallucinate."
                    )
                    custom_sum_prompt = st.text_area(
                        "進階摘要系統提示（可調整）",
                        value=default_sum_prompt,
                        height=160,
                        key=f"{key_prefix}_sum_prompt",
                    )

                    if st.button("🧾 產生此檔的專業摘要", key=f"{key_prefix}_run_summary"):
                        try:
                            summary = call_llm(
                                provider=sum_provider,
                                model=sum_model,
                                system_prompt=custom_sum_prompt,
                                user_prompt=file_info["markdown"],
                                max_tokens=int(sum_tokens),
                                temperature=0.3,
                            )
                            file_info["summary"] = summary
                            st.session_state.ocr_files[idx] = file_info
                            add_combat_log(f"{fname} 已產生摘要。", "success")
                            st.success("✅ 已產生摘要。")
                            st.markdown("##### 摘要預覽")
                            st.markdown(summary, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"產生摘要失敗：{e}")

                    if file_info.get("summary"):
                        with st.expander("🔎 目前儲存的摘要", expanded=False):
                            st.markdown(file_info["summary"], unsafe_allow_html=True)

    # Combined analysis for all OCR documents
    st.markdown("---")
    st.markdown("### 🔗 整合所有 OCR 文件並執行跨文件分析")

    all_markdowns = [
        f"## File {i+1}: {f['filename']}\n\n{f.get('markdown','')}"
        for i, f in enumerate(st.session_state.ocr_files)
        if f.get("markdown")
    ]
    if all_markdowns:
        combined_markdown = "\n\n---\n\n".join(all_markdowns)
        st.session_state.combined_markdown = combined_markdown

        with st.expander("📚 合併後 Markdown 預覽", expanded=False):
            st.markdown(combined_markdown, unsafe_allow_html=True)

        # Entity extraction across all files
        if st.button("🧬 從所有文件中抽取 20 個跨文件關鍵實體", key="combined_entities_run"):
            try:
                provider = st.session_state.get("default_provider", "openai")
                model = st.session_state.get("default_model", "gpt-4o-mini")
                system_prompt = (
                    "You are a cross-document knowledge extraction specialist for FDA 510(k) dossiers.\n"
                    "You will receive multiple OCR'd documents merged into one Markdown corpus.\n"
                    "Task: Identify up to 20 **cross-document entities** that are most important, such as:\n"
                    "- Device or component names\n"
                    "- Key clinical endpoints / indications\n"
                    "- Critical risks / hazards\n"
                    "- Pivotal performance tests or validation activities\n"
                    "- Referenced standards / guidance documents\n"
                    "For each entity, construct:\n"
                    "{\n"
                    "  \"id\": number,\n"
                    "  \"name\": string,\n"
                    "  \"type\": \"device|risk|test|clinical|regulation|other\",\n"
                    "  \"description\": \"short explanation in 1-3 sentences\",\n"
                    "  \"source_files\": [\"filename1.pdf\", \"filename2.pdf\", ...],\n"
                    "  \"context_snippet\": \"representative excerpt from one or more files\"\n"
                    "}\n"
                    "Output: JSON array only, with at most 20 entities."
                )
                user_prompt = combined_markdown
                raw = call_llm(
                    provider=provider,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=2000,
                    temperature=0.2,
                )
                raw_str = raw.strip().strip("```json").strip("```").strip()
                entities = json.loads(raw_str)
                if not isinstance(entities, list):
                    raise ValueError("回傳內容並非 JSON 陣列。")
                st.session_state.combined_entities = entities
                add_combat_log("完成跨文件 20 個關鍵實體抽取。", "success")
            except Exception as e:
                st.error(f"跨文件實體抽取失敗：{e}")

        if st.session_state.combined_entities:
            st.markdown("#### 🧬 跨文件關鍵實體表格")
            table_md = "| id | name | type | description | source_files | context_snippet |\n"
            table_md += "|---|------|------|-------------|--------------|-----------------|\n"
            for ent in st.session_state.combined_entities:
                table_md += (
                    f"| {ent.get('id','')} "
                    f"| {ent.get('name','')} "
                    f"| {ent.get('type','')} "
                    f"| {ent.get('description','').replace('|','/')} "
                    f"| {', '.join(ent.get('source_files', []))} "
                    f"| {ent.get('context_snippet','').replace('|','/')} |\n"
                )
            st.markdown(table_md)

            with st.expander("JSON 檢視", expanded=False):
                st.json(st.session_state.combined_entities)

        # Prompting on combined document
        st.markdown("### 💬 對合併後 OCR 文檔進行提問")

        qa_prompt = st.text_area(
            "請輸入對整體文件的提問或分析指令（例如：整體風險輪廓、SE 論證是否一致、哪份檔案風險較高）",
            height=140,
            key="combined_qa_prompt",
        )

        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            qa_provider = st.selectbox(
                "供應商",
                ["openai", "gemini", "xai", "anthropic"],
                key="combined_qa_provider",
            )
        with col_q2:
            provider_models = {
                "openai": ["gpt-5-nano", "gpt-4o-mini", "gpt-4.1-mini"],
                "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
                "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
                "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
            }
            qa_model = st.selectbox(
                "模型",
                provider_models[qa_provider],
                key="combined_qa_model",
            )
        with col_q3:
            qa_max_tokens = st.number_input(
                "最大回答 tokens",
                min_value=128, max_value=4096, value=1200, step=64,
                key="combined_qa_max_tokens",
            )
        with col_q4:
            qa_temp = st.slider(
                "回答溫度",
                0.0, 1.0, 0.3, 0.05,
                key="combined_qa_temp",
            )

        if st.button("💬 針對合併文件執行提問", key="combined_qa_run"):
            if not qa_prompt.strip():
                st.warning("請先輸入提問內容。")
            else:
                try:
                    system_prompt = (
                        "You are a senior FDA 510(k) reviewer analyzing multiple OCR'd documents.\n"
                        "You will receive a combined Markdown corpus representing all documents, "
                        "followed by a user question.\n"
                        "You MUST:\n"
                        "- Ground all reasoning strictly in the provided corpus.\n"
                        "- Cross-reference documents when needed (e.g., identify which file supports which point).\n"
                        "- Clearly distinguish hypotheses from explicit evidence.\n"
                        "Output: A structured Markdown answer (with headings and bullet lists) aimed at regulatory reviewers."
                    )
                    user_prompt = (
                        "=== COMBINED OCR DOCUMENTS START ===\n"
                        f"{st.session_state.combined_markdown}\n"
                        "=== COMBINED OCR DOCUMENTS END ===\n\n"
                        f"User question:\n{qa_prompt}"
                    )
                    answer = call_llm(
                        provider=qa_provider,
                        model=qa_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=int(qa_max_tokens),
                        temperature=float(qa_temp),
                    )
                    st.session_state.combined_qa_history.append(
                        {"question": qa_prompt, "answer": answer}
                    )
                    st.success("✅ 已根據合併文件完成回答。")
                    st.markdown("#### 回答")
                    st.markdown(answer, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"合併文件提問失敗：{e}")

        if st.session_state.combined_qa_history:
            with st.expander("🧾 歷史 Q&A", expanded=False):
                for i, qa in enumerate(reversed(st.session_state.combined_qa_history), start=1):
                    st.markdown(f"**Q{i}:** {qa['question']}")
                    st.markdown(qa["answer"], unsafe_allow_html=True)
                    st.markdown("---")

    else:
        st.info("請先於上方上傳至少一個 PDF 或 TXT 檔案。")

# -----------------------------------------------------------
# Dashboard Tab – enhanced with simple interactive chart
# -----------------------------------------------------------

def render_dashboard_tab():
    """Render interactive dashboard"""
    st.markdown(f"## 📊 {get_translation('dashboard')}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("審查成熟度等級", st.session_state.player_level)
    with col2:
        st.metric("已完成案件數", st.session_state.quests_completed)
    with col3:
        st.metric("LLM 呼叫次數", len(st.session_state.combat_log))
    with col4:
        st.metric("已執行流程數", len(st.session_state.pipeline_history))

    st.markdown("---")

    dash_tab1, dash_tab2, dash_tab3, dash_tab4 = st.tabs(
        ["案件歷程", "活動紀錄", "里程碑", "互動分析圖"]
    )

    with dash_tab1:
        st.markdown("### 📁 案件 / 流程歷程")
        history = st.session_state.get("pipeline_history", [])
        if not history:
            st.info("尚未執行任何審查流程。")
        else:
            for run_idx, run in enumerate(reversed(history), start=1):
                with st.expander(f"案件流程 #{len(history) - run_idx + 1}"):
                    for step_idx, item in enumerate(run, start=1):
                        st.markdown(f"**步驟 {step_idx}** – 代理 `{item['agent_id']}`")
                        st.markdown(item["output"][:300] + "...")

    with dash_tab2:
        st.markdown("### 📑 完整活動紀錄")
        if st.session_state.combat_log:
            for entry in reversed(st.session_state.combat_log):
                st.markdown(f"{entry['icon']} {entry['message']}")
        else:
            st.info("尚無活動紀錄。")

    with dash_tab3:
        st.markdown("### 🏅 審查里程碑")

        achievements = []
        if st.session_state.player_level >= 5:
            achievements.append("🎖️ 進階審查官：審查成熟度等級達 5。")
        if st.session_state.quests_completed >= 10:
            achievements.append("📜 案件達人：完成 10 件以上案件流程。")
        if len(st.session_state.combat_log) >= 50:
            achievements.append("📈 高度互動：已執行超過 50 次模型呼叫或操作。")
        if st.session_state.player_level >= 10:
            achievements.append("👑 資深審查架構師：審查成熟度等級達 10。")

        if achievements:
            for ach in achievements:
                st.success(ach)
        else:
            st.info("持續累積案件與流程，可解鎖更多審查里程碑。")

    with dash_tab4:
        st.markdown("### 📈 互動分析圖（代理使用分佈）")
        history = st.session_state.get("pipeline_history", [])
        if not history:
            st.info("尚無流程執行記錄，無法繪製統計。")
        else:
            # Count how many times each agent_id appears
            from collections import Counter
            counter = Counter()
            for run in history:
                for step in run:
                    counter[step["agent_id"]] += 1
            data = [{"agent_id": k, "count": v} for k, v in counter.items()]
            chart = (
                alt.Chart(alt.Data(values=data))
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("agent_id:N", title="Agent ID"),
                    y=alt.Y("count:Q", title="使用次數"),
                    tooltip=["agent_id", "count"],
                    color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

# -----------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="FDA 510(k) Multi-Agent Review Studio",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    apply_custom_css()
    config = load_agents_config()
    render_enhanced_sidebar(config)

    st.markdown(f"# 🏥 {get_translation('title')}")
    st.markdown(f"_{get_translation('subtitle')}_")

    render_review_context_selector()

    st.markdown("---")

    tab_input, tab_pipeline, tab_smart, tab_notes, tab_ocr, tab_dashboard = st.tabs([
        f"📝 {get_translation('input')}",
        f"🔄 {get_translation('pipeline')}",
        f"✨ {get_translation('smart_replace')}",
        f"📔 {get_translation('notes')}",
        f"📂 {get_translation('ocr')}",
        f"📊 {get_translation('dashboard')}",
    ])

    with tab_input:
        render_input_tab()

    with tab_pipeline:
        render_pipeline_tab(config)

    with tab_smart:
        render_smart_replace_tab()

    with tab_notes:
        render_notes_tab()

    with tab_ocr:
        render_submission_ocr_tab()

    with tab_dashboard:
        render_dashboard_tab()


if __name__ == "__main__":
    main()