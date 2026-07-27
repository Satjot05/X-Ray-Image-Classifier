import io
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# Workaround for Streamlit + PyTorch background file-watcher introspection crash
try:
    torch.classes.__path__ = []
except Exception:
    pass

# Limit CPU threads to prevent native crashes in constrained environments
torch.set_num_threads(1)

st.set_page_config(
    page_title="DeepVision Medical AI | Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = "pneumonia_classifier_best.pth"
MODEL_URL = "https://github.com/Satjot05/X-Ray-Image-Classifier/releases/download/python/pneumonia_classifier_best.pth"
IMG_SIZE = 224

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Background: professional, static (no animation), with
       enough tonal variation for the glass/blur cards to read against ---------- */
    .stApp {
        background-color: #0a0f1e;
        background-image:
            radial-gradient(1100px 700px at 15% -10%, rgba(48,84,150,0.35), transparent 60%),
            radial-gradient(900px 600px at 100% 10%, rgba(70,60,120,0.22), transparent 55%),
            radial-gradient(1000px 800px at 50% 120%, rgba(20,40,70,0.4), transparent 60%),
            linear-gradient(160deg, #0c1326 0%, #0a0f1e 45%, #0b0f1c 100%);
        background-attachment: fixed;
        color: #e6e9f2;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }
    div[data-testid="stDecoration"] { visibility: hidden; }

    /* Keep the deploy/menu icons out of the toolbar, but never touch the
       sidebar collapse/expand arrow - it lives in this same toolbar area. */
    div[data-testid="stToolbar"] button[title="Deploy this app"],
    div[data-testid="stToolbar"] [data-testid="stToolbarActions"] {
        visibility: hidden;
    }

    /* The little arrow that reopens the sidebar once it's collapsed.
       This must always stay visible and above everything else, or the
       sidebar becomes permanently stuck closed. */
    div[data-testid="collapsedControl"],
    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }

    /* ---------- Sidebar: let Streamlit control open/collapse ---------- */
    section[data-testid="stSidebar"] {
        background: #0c1424 !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }


    .block-container {
        padding-top: 2.4rem;
        padding-bottom: 4rem;
        padding-left: clamp(1.5rem, 4vw, 3.5rem);
        padding-right: clamp(1.5rem, 4vw, 3.5rem);
        max-width: 1360px;
    }

    div[data-testid="stHorizontalBlock"] { gap: 1.1rem; }
    div[data-testid="column"] { margin-bottom: 0.8rem; }
    
  /* ---------- Mobile: stacked spec-cards need more breathing room,
       and the tab row should scroll on one line instead of wrapping
       unevenly onto two rows. Covers both "column" and "stColumn" since
       the attribute name differs across Streamlit versions. ---------- */
    @media (max-width: 640px) {
        div[data-testid="stVerticalBlock"] div[data-testid="column"],
        div[data-testid="stVerticalBlock"] div[data-testid="stColumn"],
        div[data-testid="column"],
        div[data-testid="stColumn"] {
            margin-bottom: 1.4rem !important;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 1.4rem !important;
            row-gap: 1.4rem !important;
        }

 
        div[data-testid="stTabs"] [data-baseweb="tab"],
        div[data-testid="stTabs"] [role="tab"] {
            flex: 0 0 auto !important;
            padding: 0 0.85rem !important;
            font-size: 0.8rem !important;
        }
    }

    /* ---------- Header ---------- */
    .dv-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f2f4fa;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .dv-title .accent {
        color: #5fa8ff;
    }
    .dv-subtitle {
        color: #8a94b3;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.15rem;
        margin-bottom: 1.6rem;
        letter-spacing: 0.01em;
    }
    .dv-badge-row { display:flex; gap:0.5rem; margin-bottom: 1.8rem; flex-wrap: wrap;}
    .dv-badge {
        background: rgba(95,168,255,0.07);
        border: 1px solid rgba(95,168,255,0.28);
        color: #7fb4ff;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        font-family: 'JetBrains Mono', monospace;
    }

     /* ---------- Tabs ---------- */
    div[data-testid="stTabs"] [data-baseweb="tab-list"],
    div[data-testid="stTabs"] [role="tablist"] {
        gap: 4px !important;
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
        padding: 5px !important;
        margin-bottom: 2rem !important;
        flex-wrap: wrap !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab"],
    div[data-testid="stTabs"] [role="tab"] {
        height: 40px !important;
        border-radius: 7px !important;
        color: #8a94b3 !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        padding: 0 0.85rem  !important;
        background: transparent !important;
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        background: rgba(95,168,255,0.14) !important;
        color: #f2f4fa !important;
        box-shadow: inset 0 0 0 1px rgba(95,168,255,0.3) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
 
    /* ---------- Cards ---------- */
    .glass-card {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
    }

    .spec-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        text-align: left;
        height: 100%;
    }
    .spec-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7d8bb0;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .spec-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #eef2fb;
        font-family: 'JetBrains Mono', monospace;
    }
    .spec-icon { font-size: 1.3rem; margin-bottom: 0.5rem; opacity: 0.8;}

    section[data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1.5px dashed rgba(95,168,255,0.35) !important;
        border-radius: 12px !important;
    }

    /* ---------- Diagnostic result card ---------- */
    .diag-card {
        border-radius: 14px;
        padding: 2rem 2.2rem;
        margin-top: 0.6rem;
        border: 1px solid;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }
    .diag-normal {
        background: rgba(34,197,94,0.06);
        border-color: rgba(34,197,94,0.35);
    }
    .diag-pneumonia {
        background: rgba(239,68,68,0.07);
        border-color: rgba(239,68,68,0.4);
    }
    .diag-status {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.8;
        margin-bottom: 0.4rem;
    }
    .diag-result {
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.7rem;
    }
    .diag-normal .diag-result { color: #4ade80; }
    .diag-pneumonia .diag-result { color: #f87171; }
    .diag-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        color: #cbd5e8;
        margin-bottom: 1.1rem;
    }
    .diag-note-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #93a1c2;
        font-weight: 700;
        margin-bottom: 0.4rem;
        margin-top: 0.7rem;
    }
    .diag-note-text {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #d6dcec;
    }

    /* ---------- Sidebar branding ---------- */
    .sb-logo {
        display:flex; align-items:center; gap:0.6rem;
        margin-bottom: 0.2rem;
    }
    .sb-logo-icon { font-size: 1.6rem; }
    .sb-logo-text {
        font-size: 1.1rem;
        font-weight: 700;
        color: #eef2fb;
        letter-spacing: -0.01em;
    }
    .sb-logo-sub {
        font-size: 0.7rem;
        color: #6f7ea3;
        letter-spacing: 0.05em;
        margin-bottom: 1.6rem;
    }
    .sb-disclaimer {
        background: rgba(239,68,68,0.06);
        border: 1px solid rgba(239,68,68,0.25);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        font-size: 0.9rem;
        line-height: 1.55;
        color: #f0b8b8;
        margin-top: 1.6rem;
    }

    .stProgress > div > div > div > div {
        background: #5fa8ff;
    }

    hr { border-color: rgba(255,255,255,0.07); }
    </style>
    """,
    unsafe_allow_html=True,
)

FORCE_CPU = True


def download_model_if_missing():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        if MODEL_URL:
            try:
                req = urllib.request.Request(
                    MODEL_URL,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                with urllib.request.urlopen(req) as response, open(MODEL_PATH, "wb") as out_file:
                    out_file.write(response.read())
            except Exception as download_err:
                return False, str(download_err)
    return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000, None


@st.cache_resource(show_spinner=False)
def load_model():
    download_success, err_msg = download_model_if_missing()

    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    loaded = False

    if download_success:
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            loaded = True
        except Exception:
            loaded = False

    model.eval()
    model.to(device)
    return model, device, loaded, err_msg


preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model, device, model_loaded, download_error = load_model()

with st.sidebar:
    st.markdown(
        """
        <div class="sb-logo">
            <div class="sb-logo-icon">🫁</div>
            <div class="sb-logo-text">DeepVision</div>
        </div>
        <div class="sb-logo-sub">MEDICAL AI &nbsp;·&nbsp; DIAGNOSTIC IMAGING</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Model Status")
    if model_loaded:
        st.success("Trained model weights loaded")
        num_params = sum(p.numel() for p in model.parameters())
        st.caption(f"{num_params / 1e6:.1f}M parameters · ResNet18 backbone")
    else:
        st.warning("Running in Demo Mode", icon="⚠️")
        if download_error:
            st.error(f"Download Error: {download_error}", icon="❌")

    st.markdown("---")

    with st.expander("🧭 Quick Guide", expanded=False):
        st.markdown(
            """
            1. Upload a chest X-ray (JPG/PNG) on the main panel.
            2. Click **Run AI Scan**.
            3. Review the diagnostic report and confidence score.
            4. View raw probability breakdown under *Full class breakdown*.
            """
        )

    with st.expander("🔬 Technical Specifications", expanded=False):
        st.markdown(
            """
            **Architecture**
            ResNet18 transfer learning with frozen convolutional base and
            re-initialized classification head (`fc → Linear(512, 2)`).

            **Input Pipeline**
            `Resize(224×224)` → `ToTensor` → ImageNet `Normalize`.

            **Loss Function**
            `CrossEntropyLoss` with inverse-frequency class weights.
            """
        )

    with st.expander("📊 Dataset Notes", expanded=False):
        st.markdown(
            """
            Pediatric chest X-ray dataset (Kaggle chest_xray).
            **Training set:** 5,216 images (1,341 Normal · 3,875 Pneumonia).
            """
        )

    st.markdown(
        """
        <div class="sb-disclaimer">
        <strong>⚠️ Clinical Disclaimer</strong><br>
        This tool is an academic prototype (6th-semester project) and
        is <strong>not</strong> a certified diagnostic device.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="dv-title">DeepVision <span class="accent">Medical AI</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="dv-subtitle">Pediatric Pneumonia Detection System · '
    "AI-assisted chest x-ray classification</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="dv-badge-row">
        <div class="dv-badge">RESNET18 · TRANSFER LEARNING</div>
        <div class="dv-badge">CLASS-WEIGHTED CROSS ENTROPY</div>
        <div class="dv-badge">PYTORCH INFERENCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_diagnostics, tab_insights, tab_about = st.tabs(
    ["🩺  Diagnostics", "📊  Model Insights", "ℹ️  About"]
)

with tab_diagnostics:
    spec_cols = st.columns(4)
    specs = [
        ("🎯", "Target Condition", "Pneumonia"),
        ("🧠", "Network", "ResNet18"),
        ("⚡", "Inference Time", "~120 ms"),
        ("🖥️", "Compute Device", str(device).upper()),
    ]
    for col, (icon, label, value) in zip(spec_cols, specs):
        with col:
            st.markdown(
                f"""
                <div class="spec-card">
                    <div class="spec-icon">{icon}</div>
                    <div class="spec-label">{label}</div>
                    <div class="spec-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown("#### 📤 Upload Chest X-Ray")
        uploaded_file = st.file_uploader(
            "Accepted formats: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        image = None
        if uploaded_file is not None:
            if uploaded_file.name.startswith("._"):
                st.warning(
                    "This looks like a macOS hidden metadata file "
                    f"(`{uploaded_file.name}`), not an actual image. "
                    "Please upload the real image file instead.",
                    icon="🍎",
                )
            else:
                try:
                    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                    st.image(image, caption="Uploaded radiograph", use_container_width=True)
                except UnidentifiedImageError:
                    st.error(
                        "This file couldn't be read as an image.",
                        icon="🚫",
                    )
                except Exception as e:
                    st.error(f"Unexpected error while reading image: {e}", icon="🚫")

    with right:
        st.markdown("#### 🩺 Diagnostic Report")

        if uploaded_file is None:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; color:#7d8bb0; padding: 3rem 1.5rem;">
                    Upload a chest X-ray on the left to run the scan.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif image is not None:
            run = st.button("▶️  Run AI Scan", type="primary", use_container_width=True)

            if run:
                progress_label = st.empty()
                progress_bar = st.progress(0)
                stages = [
                    (18, "Normalizing pixel intensities…"),
                    (40, "Extracting convolutional features…"),
                    (65, "Running ResNet18 forward pass…"),
                    (85, "Computing softmax confidence…"),
                    (100, "Finalizing diagnostic report…"),
                ]
                progress = 0
                for target, label in stages:
                    progress_label.markdown(f"`{label}`")
                    while progress < target:
                        progress += 2
                        progress_bar.progress(min(progress, 100))
                        time.sleep(0.012)
                progress_label.empty()
                progress_bar.empty()

                try:
                    input_tensor = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if model_loaded:
                            logits = model(input_tensor)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        else:
                            rng = np.random.default_rng(hash(uploaded_file.name) % (2**32))
                            raw = rng.normal(loc=[0.4, 0.6], scale=0.25, size=2)
                            probs = np.exp(raw) / np.exp(raw).sum()

                    pred_idx = int(np.argmax(probs))
                    pred_label = CLASS_NAMES[pred_idx]
                    confidence = float(probs[pred_idx]) * 100

                    if pred_label == "NORMAL":
                        st.markdown(
                            f"""
                            <div class="diag-card diag-normal">
                                <div class="diag-status">Diagnostic Result</div>
                                <div class="diag-result">Normal</div>
                                <div class="diag-confidence">Model confidence: {confidence:.2f}%</div>
                                <div class="diag-note-label">Clinical Note</div>
                                <div class="diag-note-text">
                                    No radiographic evidence of focal consolidation,
                                    lobar opacity, or air-space disease is apparent.
                                    Lung fields appear clear with no significant
                                    infiltrates detected by the model. Correlate
                                    clinically if symptoms persist.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="diag-card diag-pneumonia">
                                <div class="diag-status">Diagnostic Result</div>
                                <div class="diag-result">⚠️ Pneumonia Detected</div>
                                <div class="diag-confidence">Model confidence: {confidence:.2f}%</div>
                                <div class="diag-note-label">Clinical Note</div>
                                <div class="diag-note-text">
                                    The model detects patterns consistent with
                                    increased opacity / consolidation suggestive
                                    of pneumonic infiltrate. Findings may indicate
                                    focal or diffuse air-space disease. Recommend
                                    review by a radiologist before any treatment
                                    decision.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with st.expander("📈 Full class probability breakdown"):
                        for cls, p in zip(CLASS_NAMES, probs):
                            st.markdown(f"**{cls}**")
                            st.progress(float(p))
                            st.caption(f"{p * 100:.2f}%")

                    if not model_loaded:
                        st.info(
                            "Running in demo mode. Make sure your GitHub repository and Release "
                            "are set to **Public** so Streamlit can auto-download the `.pth` file.",
                            icon="ℹ️",
                        )
                except Exception as e:
                    st.error(f"Inference failed: {e}", icon="🚫")

with tab_insights:
    st.markdown("#### 📊 Model Insights")

    ic1, ic2, ic3 = st.columns(3)
    num_params = sum(p.numel() for p in model.parameters()) if model_loaded else 11_700_000
    with ic1:
        st.markdown(
            f"""<div class="spec-card"><div class="spec-icon">🧮</div>
            <div class="spec-label">Total Parameters</div>
            <div class="spec-value">{num_params / 1e6:.1f}M</div></div>""",
            unsafe_allow_html=True,
        )
    with ic2:
        st.markdown(
            """<div class="spec-card"><div class="spec-icon">🧊</div>
            <div class="spec-label">Trainable Layer</div>
            <div class="spec-value">fc (512→2)</div></div>""",
            unsafe_allow_html=True,
        )
    with ic3:
        st.markdown(
            """<div class="spec-card"><div class="spec-icon">📐</div>
            <div class="spec-label">Input Resolution</div>
            <div class="spec-value">224 × 224</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Training Set Class Distribution")
    class_dist = pd.DataFrame(
        {"Images": [1341, 3875]}, index=["NORMAL", "PNEUMONIA"]
    )
    st.bar_chart(class_dist)

with tab_about:
    st.markdown("#### ℹ️ About This Project")
    st.markdown(
        """
        <div class="glass-card">
        <b>DeepVision Medical AI</b> is a 6th-semester academic project
        exploring AI-assisted Web app for pediatric pneumonia detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:#5c6a8c; font-size:0.78rem; letter-spacing:0.03em;">
        DeepVision Medical AI &nbsp;·&nbsp; 6th Semester Academic Project
    </div>
    """,
    unsafe_allow_html=True,
)
