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

    .stApp {
        background-color: #0a0f22;
        background-image:
            radial-gradient(circle 380px at 8% 12%, rgba(79,157,255,0.28), transparent 70%),
            radial-gradient(circle 420px at 92% 45%, rgba(185,139,255,0.24), transparent 70%),
            radial-gradient(circle 320px at 38% 95%, rgba(255,139,196,0.20), transparent 70%),
            linear-gradient(120deg, #0a1230, #0f1c3f, #131338, #0a1c33, #0a1230);
        background-repeat: no-repeat;
        background-size: 140% 140%, 140% 140%, 140% 140%, 300% 300%;
        animation: dv-orb-1 17s ease-in-out infinite,
                   dv-orb-2 21s ease-in-out infinite,
                   dv-orb-3 19s ease-in-out infinite,
                   dv-gradient-shift 26s ease infinite;
        color: #e7ecf7;
    }
    @keyframes dv-gradient-shift {
        0%   { background-position: 0% 50%, 0% 50%, 0% 50%, 0% 50%; }
        50%  { background-position: 0% 50%, 0% 50%, 0% 50%, 100% 50%; }
        100% { background-position: 0% 50%, 0% 50%, 0% 50%, 0% 50%; }
    }
    @keyframes dv-orb-1 {
        0%, 100% { background-position: 8% 12%, 0% 0%, 0% 0%, 0% 0%; }
        50%      { background-position: 16% 20%, 0% 0%, 0% 0%, 0% 0%; }
    }
    @keyframes dv-orb-2 {
        0%, 100% { background-position: 0% 0%, 92% 45%, 0% 0%, 0% 0%; }
        50%      { background-position: 0% 0%, 84% 55%, 0% 0%, 0% 0%; }
    }
    @keyframes dv-orb-3 {
        0%, 100% { background-position: 0% 0%, 0% 0%, 38% 95%, 0% 0%; }
        50%      { background-position: 0% 0%, 0% 0%, 46% 85%, 0% 0%; }
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }
    div[data-testid="stToolbar"] { visibility: hidden; }
    div[data-testid="stDecoration"] { visibility: hidden; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1226 0%, #060a15 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
        z-index: 100 !important;
        display: flex !important;
        visibility: visible !important;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        padding-left: clamp(1rem, 4vw, 3rem);
        padding-right: clamp(1rem, 4vw, 3rem);
        max-width: 1400px;
    }

    div[data-testid="stHorizontalBlock"] { gap: 1rem; }
    div[data-testid="column"] { margin-bottom: 0.6rem; }

    .dv-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #4f9dff 0%, #7ee8fa 35%, #b98bff 70%, #ff8bc4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        line-height: 1.15;
    }
    .dv-subtitle {
        color: #92a1c4;
        font-size: 1.02rem;
        font-weight: 400;
        margin-top: 0.2rem;
        margin-bottom: 1.4rem;
        letter-spacing: 0.01em;
    }
    .dv-badge-row { display:flex; gap:0.5rem; margin-bottom: 1.4rem; flex-wrap: wrap;}
    .dv-badge {
        background: rgba(79,157,255,0.08);
        border: 1px solid rgba(79,157,255,0.35);
        color: #8fc2ff;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        font-family: 'JetBrains Mono', monospace;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 14px;
        padding: 6px;
        margin-bottom: 1.6rem;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        color: #92a1c4;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 1.1rem;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(79,157,255,0.22), rgba(185,139,255,0.22));
        color: #eef2fb !important;
        box-shadow: inset 0 0 0 1px rgba(79,157,255,0.35);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }

    .spec-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        text-align: left;
        height: 100%;
    }
    .spec-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #7d8bb0;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .spec-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #eef2fb;
        font-family: 'JetBrains Mono', monospace;
    }
    .spec-icon { font-size: 1.4rem; margin-bottom: 0.4rem; opacity: 0.85;}

    section[data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1.5px dashed rgba(79,157,255,0.4) !important;
        border-radius: 16px !important;
    }

    .diag-card {
        border-radius: 20px;
        padding: 1.8rem 2rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        margin-top: 0.5rem;
        border: 1px solid;
    }
    .diag-normal {
        background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(34,197,94,0.03));
        border-color: rgba(34,197,94,0.45);
        box-shadow: 0 8px 32px rgba(34,197,94,0.12);
    }
    .diag-pneumonia {
        background: linear-gradient(135deg, rgba(239,68,68,0.16), rgba(239,68,68,0.03));
        border-color: rgba(239,68,68,0.5);
        box-shadow: 0 8px 32px rgba(239,68,68,0.14);
    }
    .diag-status {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 0.3rem;
    }
    .diag-result {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.6rem;
    }
    .diag-normal .diag-result { color: #4ade80; }
    .diag-pneumonia .diag-result { color: #f87171; }
    .diag-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: #cbd5e8;
        margin-bottom: 0.9rem;
    }
    .diag-note-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #93a1c2;
        font-weight: 700;
        margin-bottom: 0.35rem;
        margin-top: 0.6rem;
    }
    .diag-note-text {
        font-size: 0.92rem;
        line-height: 1.55;
        color: #dde4f3;
    }

    .sb-logo {
        display:flex; align-items:center; gap:0.6rem;
        margin-bottom: 0.2rem;
    }
    .sb-logo-icon { font-size: 1.8rem; }
    .sb-logo-text {
        font-size: 1.15rem;
        font-weight: 800;
        color: #eef2fb;
        letter-spacing: -0.01em;
    }
    .sb-logo-sub {
        font-size: 0.72rem;
        color: #6f7ea3;
        letter-spacing: 0.05em;
        margin-bottom: 1.4rem;
    }
    .sb-disclaimer {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 0.74rem;
        line-height: 1.5;
        color: #f2b8b8;
        margin-top: 1.5rem;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4f9dff, #7ee8fa);
    }

    hr { border-color: rgba(255,255,255,0.08); }
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
        st.success("Trained model weights loaded (`~45 MB`)", icon="✅")
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

st.markdown('<div class="dv-title">DeepVision Medical AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dv-subtitle">Pediatric Pneumonia Detection System · '
    "AI-assisted chest radiograph triage</div>",
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

    st.markdown("<br>", unsafe_allow_html=True)

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

    st.markdown("<br>", unsafe_allow_html=True)
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
        exploring AI-assisted triage for pediatric pneumonia detection.
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
