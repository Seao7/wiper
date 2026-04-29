import streamlit as st
import os
import numpy as np
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="Hyperspectral Residue Analysis",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Root palette ── */
    :root {
        --bg-primary:    #0b0f14;
        --bg-surface:    #111720;
        --bg-card:       #161d28;
        --border:        #1f2d3d;
        --accent:        #00d4ff;
        --accent-dim:    rgba(0, 212, 255, 0.12);
        --accent-green:  #00e5a0;
        --accent-warm:   #ff8c42;
        --text-primary:  #e8edf3;
        --text-secondary:#7a8fa6;
        --text-muted:    #4a5d72;
        --danger:        #ff4d6a;
    }

    /* ── Global resets ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    section[data-testid="stSidebar"] .stRadio > label { color: var(--text-secondary) !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        margin-bottom: 4px;
        font-size: 0.88rem;
        transition: all 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        border-color: var(--accent);
        color: var(--accent) !important;
    }

    /* ── App header ── */
    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 1.2rem;
        margin-bottom: 2rem;
    }
    .app-header .brand {
        font-family: 'Space Mono', monospace;
        font-size: 1.05rem;
        color: var(--accent);
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .app-header .title {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    .app-header .badge {
        margin-left: auto;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent-green);
        border: 1px solid var(--accent-green);
        border-radius: 4px;
        padding: 2px 8px;
        letter-spacing: 0.1em;
    }

    /* ── Tab bar (the group selector) ── */
    .tab-bar {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.8rem;
        flex-wrap: wrap;
    }
    .tab-btn {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.55rem 1.2rem;
        border-radius: 6px;
        border: 1px solid var(--border);
        background: var(--bg-card);
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s;
    }
    .tab-btn:hover  { border-color: var(--accent); color: var(--accent); }
    .tab-btn.active {
        background: var(--accent-dim);
        border-color: var(--accent);
        color: var(--accent);
    }

    /* ── Summary card ── */
    .summary-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 2rem;
        font-size: 0.93rem;
        line-height: 1.65;
        color: var(--text-secondary);
    }
    .summary-card .label {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.4rem;
    }

    /* ── Baseline reference card ── */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    /* ── Sample block ── */
    .sample-header {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 1.6rem 0 0.8rem;
    }
    .sample-index {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: var(--accent);
        border: 1px solid var(--accent);
        border-radius: 4px;
        padding: 2px 6px;
        min-width: 28px;
        text-align: center;
    }
    .sample-title {
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-primary);
    }
    .sample-id {
        margin-left: auto;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-muted);
    }

    /* ── Column headers ── */
    .col-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid var(--border);
    }

    /* ── Image containers ── */
    .stImage img {
        border-radius: 6px;
        border: 1px solid var(--border);
        transition: border-color 0.2s;
    }
    .stImage img:hover { border-color: var(--accent); }

    /* ── Missing data ── */
    .missing-data {
        background: rgba(255, 77, 106, 0.07);
        border: 1px dashed var(--danger);
        border-radius: 6px;
        padding: 1.2rem;
        text-align: center;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        color: var(--danger);
        letter-spacing: 0.06em;
    }

    /* ── Divider ── */
    .sample-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.2rem 0 0;
    }

    /* ── Streamlit widget fixes ── */
    .stSelectbox label, .stRadio label, .stSidebar label { color: var(--text-secondary) !important; }
    div[data-baseweb="select"] { background: var(--bg-card) !important; border-color: var(--border) !important; }
    .stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# --- DATA DEFINITIONS ---
background_mapped_groups = {
    "B_out": {
        "background": "kido5000.gain200.B.background",
        "samples": {
            "kido5000.gain200.B.all":   "No Cleaning — Initial State",
            "kido5000.gain200.B.1time": "Wiped 1×",
            "kido5000.gain200.B.2time": "Wiped 2×",
            "kido5000.gain200.B.3time": "Wiped 3×",
            "kido5000.gain200.B.4time": "Wiped 4×",
        }
    },
    "C_out": {
        "background": "kido5000.gain200.C.background",
        "samples": {
            "kido5000.gain200.C.all":   "No Cleaning — Initial State",
            "kido5000.gain200.C.1time": "Wiped 1×",
            "kido5000.gain200.C.2time": "Wiped 2×",
            "kido5000.gain200.C.3time": "Wiped 3×",
            "kido5000.gain200.C.4time": "Wiped 4×",
        }
    },
    "B_C_in": {
        "background": "kido5000.gain255.C.background.in.light",
        "samples": {
            "kido5000.gain255.B.all.in.light":   "Sample B — No Cleaning",
            "kido5000.gain255.B.1time.in.light":  "Sample B — Wiped 1×",
            "kido5000.gain255.B.4time.in.light":  "Sample B — Wiped 4×",
            "kido5000.gain255.C.4time.in.light":  "Sample C — Wiped 4×",
        }
    }
}

# --- TRANSLATIONS ---
translations = {
    "English": {
        "title": "Hyperspectral Residue Analysis",
        "badge": "LIVE ANALYSIS",
        "story_label": "Executive Summary",
        "bg_header": "Baseline Reference — Clean State",
        "col1_title": "RGB — Standard Vision",
        "col2_title": "Spectral Variance",
        "col3_title": "Residue Mask",
        "missing": "DATA UNAVAILABLE",
        "logo_placeholder": "YOUR LOGO HERE",
        "copyright": "© Invisible World. All rights reserved.",
        "b_out_story": "Analysis indicates initial contaminants are highly concentrated on the driver's side, likely distributed by the wiper mechanism. Across sequential cleaning cycles, the driver's side clears effectively, but particulate migration and settling on the passenger side is observed.",
        "c_out_story": "While exhibiting a similar migration pattern to Sample B, Dirt C demonstrates a distinct hyperspectral signature, suggesting variances in material density and surface adhesion properties during wipe cycles.",
        "in_story": "Internal cabin angles reveal that residue presence actively intensifies light scattering. Contaminants nearly invisible to the naked eye appear as highly active 'glow' regions in spectral variance maps.",
        "tab_labels": {"B_out": "Sample B — External", "C_out": "Sample C — External", "B_C_in": "B & C — Internal"},
    },
    "日本語": {
        "title": "ハイパースペクトル残留物分析",
        "badge": "分析中",
        "story_label": "エグゼクティブ・サマリー",
        "bg_header": "ベースライン参照 — 清掃完了状態",
        "col1_title": "RGB — 標準カメラ映像",
        "col2_title": "分光分散マップ",
        "col3_title": "残留物マスク",
        "missing": "データ利用不可",
        "logo_placeholder": "ロゴをここに",
        "copyright": "著作権 © Invisible World 無断での複写・転載を禁じます。",
        "b_out_story": "初期の汚れは運転席側に集中しており、ワイパーの作動によって拡散した可能性が高いことが示されました。清掃サイクルを重ねるごとに運転席側は効果的に清浄化されますが、微粒子が助手席側へ移動・定着するマイグレーション現象が確認されています。",
        "c_out_story": "サンプルBと同様の移動パターンを示していますが、汚れCは異なるハイパースペクトル・シグネチャを示しており、拭き取り時の物質密度や表面付着力の差異が示唆されます。",
        "in_story": "車内からの分析では、残留物が光の散乱を積極的に強めていることが判明しました。肉眼ではほぼ見えない汚れも、分光分散マップ上では高活性の発光領域として検出されます。",
        "tab_labels": {"B_out": "サンプルB — 外部", "C_out": "サンプルC — 外部", "B_C_in": "B & C — 内部"},
    }
}

# --- SIDEBAR ---

# Sidebar logo placeholder
with st.sidebar:
    st.image("logo.png", use_container_width=True)

selected_lang = st.sidebar.radio("Language / 言語", ["English", "日本語"], label_visibility="collapsed")
t = translations[selected_lang]

# Fixed directories (not shown to user)
rgb_dir     = "rgb"
heatmap_dir = "variance/heatmaps"
mask_dir    = "refined_masks"

# Sidebar footer copyright — pushed to bottom via flex spacer trick
st.sidebar.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style='
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #1f2d3d;
    font-family: DM Sans, sans-serif;
    font-size: 0.68rem;
    color: #2a3d52;
    line-height: 1.7;
'>
    {t['copyright']}
</div>
""", unsafe_allow_html=True)

# --- GROUP SELECTOR via Streamlit tabs (acts as tab bar) ---
story_map = {
    "B_out":  t["b_out_story"],
    "C_out":  t["c_out_story"],
    "B_C_in": t["in_story"],
}

# --- APP HEADER ---
import base64

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = img_to_base64("logo.png")

# Then in the header HTML string:
st.markdown(f"""
<div class="app-header">
    <span class="title">{t['title']}</span>
    <span class="badge">{t['badge']}</span>
    <img src="data:image/png;base64,{logo_b64}"
         style="margin-left:1.5rem; height:44px; width:auto; object-fit:contain;" />
</div>
""", unsafe_allow_html=True)

# --- TABS as group selector ---
tab_labels = [t["tab_labels"][k] for k in ["B_out", "C_out", "B_C_in"]]
tabs = st.tabs(tab_labels)

# Apply dark tab styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 4px;
        border-bottom: 1px solid #1f2d3d;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4a5d72 !important;
        background: #161d28;
        border: 1px solid #1f2d3d;
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1.1rem;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,212,255,0.1) !important;
        border-color: #00d4ff !important;
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
    .stTabs [data-baseweb="tab-border"] { background: #1f2d3d !important; }
</style>
""", unsafe_allow_html=True)

group_keys = ["B_out", "C_out", "B_C_in"]

def render_group(group_key, t, rgb_dir, heatmap_dir, mask_dir):
    group_data = background_mapped_groups[group_key]

    # Summary card
    st.markdown(f"""
    <div class="summary-card">
        <div class="label">◈ {t['story_label']}</div>
        {story_map[group_key]}
    </div>
    """, unsafe_allow_html=True)

    # Baseline reference
    st.markdown(f'<div class="section-label">◈ {t["bg_header"]}</div>', unsafe_allow_html=True)
    bg_key  = group_data["background"]
    bg_path = os.path.join(rgb_dir, f"{bg_key}.jpg")
    col_bg, _ = st.columns([1, 3])
    with col_bg:
        if os.path.exists(bg_path):
            st.image(bg_path, use_container_width=True,
                     caption=bg_key)
        else:
            st.markdown(f'<div class="missing-data">{t["missing"]}<br><small>{bg_key}</small></div>',
                        unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1f2d3d;margin:1.5rem 0;'>", unsafe_allow_html=True)

    # Column headers (once, above all samples)
    hdr1, hdr2, hdr3 = st.columns(3)
    with hdr1:
        st.markdown(f'<div class="col-header">① {t["col1_title"]}</div>', unsafe_allow_html=True)
    with hdr2:
        st.markdown(f'<div class="col-header">② {t["col2_title"]}</div>', unsafe_allow_html=True)
    with hdr3:
        st.markdown(f'<div class="col-header">③ {t["col3_title"]}</div>', unsafe_allow_html=True)

    # Samples
    for idx, (sample_key, desc) in enumerate(group_data["samples"].items(), 1):
        st.markdown(f"""
        <div class="sample-header">
            <span class="sample-index">S{idx:02d}</span>
            <span class="sample-title">{desc}</span>
            <span class="sample-id">{sample_key}</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        rgb_path     = os.path.join(rgb_dir,     f"{sample_key}.jpg")
        heatmap_path = os.path.join(heatmap_dir, f"{sample_key}_heatmap.png")
        mask_path    = os.path.join(mask_dir,    f"{sample_key}_refined.npy")

        def missing_box(label): 
            return f'<div class="missing-data">{label}</div>'

        with col1:
            if os.path.exists(rgb_path):
                st.image(Image.open(rgb_path), use_container_width=True)
            else:
                st.markdown(missing_box(t["missing"]), unsafe_allow_html=True)

        with col2:
            if os.path.exists(heatmap_path):
                st.image(heatmap_path, use_container_width=True)
            else:
                st.markdown(missing_box(t["missing"]), unsafe_allow_html=True)

        with col3:
            if os.path.exists(rgb_path) and os.path.exists(mask_path):
                rgb_array  = np.array(Image.open(rgb_path).convert("RGB"))
                mask_array = np.load(mask_path)
                rgb_array[mask_array] = [255, 0, 0]
                st.image(Image.fromarray(rgb_array), use_container_width=True)
            else:
                st.markdown(missing_box(t["missing"]), unsafe_allow_html=True)

        st.markdown('<hr class="sample-divider">', unsafe_allow_html=True)

# Render each tab
for tab_widget, group_key in zip(tabs, group_keys):
    with tab_widget:
        render_group(group_key, t, rgb_dir, heatmap_dir, mask_dir)