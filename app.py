import streamlit as st
import numpy as np
from PIL import Image
import io

# ---------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="PneumoScan AI — Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}

:root {
    --bg: #050d1a;
    --card: #0f1e35;
    --border: rgba(56,139,255,0.18);
    --accent: #388bff;
    --accent2: #6366f1;
    --danger: #ff4d6d;
    --safe: #00e5a0;
    --warn: #f59e0b;
    --text: #e8f0ff;
    --muted: #6b82a8;
    --muted2: #8fa3c4;
}

.stApp {
    background-color: #050d1a;
    font-family: 'DM Sans', sans-serif;
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #050d1a 0%, #0b1628 100%);
    border: 1px solid rgba(56,139,255,0.18);
    border-radius: 20px;
    padding: 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    text-align: center;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #388bff, transparent);
}

.hero-badge {
    display: inline-block;
    background: rgba(56,139,255,0.1);
    border: 1px solid rgba(56,139,255,0.3);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #388bff;
    margin-bottom: 20px;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: #e8f0ff;
    margin-bottom: 16px;
}

.hero-title span {
    background: linear-gradient(135deg, #388bff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-desc {
    color: #8fa3c4;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 600px;
    margin: 0 auto 32px;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 48px;
    flex-wrap: wrap;
}

.stat-item {
    text-align: center;
}

.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #388bff;
}

.stat-label {
    font-size: 0.72rem;
    color: #6b82a8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Section Headers */
.section-tag {
    display: inline-block;
    background: rgba(56,139,255,0.1);
    border: 1px solid rgba(56,139,255,0.2);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #388bff;
    margin-bottom: 12px;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e8f0ff;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}

.section-title span {
    background: linear-gradient(135deg, #388bff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.section-desc {
    color: #6b82a8;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-bottom: 32px;
}

/* Cards */
.info-card {
    background: #0f1e35;
    border: 1px solid rgba(56,139,255,0.18);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
}

.info-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e8f0ff;
    margin-bottom: 16px;
}

/* Stat Cards */
.stat-card {
    background: #0f1e35;
    border: 1px solid rgba(56,139,255,0.18);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin-bottom: 14px;
}

.stat-card-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #e8f0ff;
    margin-bottom: 4px;
}

.stat-card-label {
    font-size: 0.76rem;
    color: #6b82a8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Model Cards */
.model-card {
    background: #0f1e35;
    border: 1px solid rgba(56,139,255,0.18);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
}

.model-card.best {
    border-color: rgba(0,229,160,0.4);
}

.model-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #e8f0ff;
    margin-bottom: 4px;
}

.model-arch {
    font-size: 0.78rem;
    color: #388bff;
    margin-bottom: 10px;
}

.model-desc {
    font-size: 0.83rem;
    color: #8fa3c4;
    line-height: 1.6;
    margin-bottom: 14px;
}

/* Result Cards */
.result-card-pneumonia {
    background: rgba(255,77,109,0.08);
    border: 1px solid rgba(255,77,109,0.35);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.result-card-normal {
    background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.35);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.result-title-pneumonia {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #ff4d6d;
    margin-bottom: 8px;
}

.result-title-normal {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #00e5a0;
    margin-bottom: 8px;
}

.result-desc {
    color: #8fa3c4;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Performance cards */
.perf-card {
    background: #0f1e35;
    border: 1px solid rgba(56,139,255,0.18);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 14px;
}

.perf-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 4px;
}

.perf-label {
    font-size: 0.74rem;
    color: #6b82a8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Stack cards */
.stack-card {
    background: #0f1e35;
    border: 1px solid rgba(56,139,255,0.18);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 14px;
}

.stack-icon { font-size: 1.8rem; margin-bottom: 10px; }
.stack-name { font-family: 'Syne', sans-serif; font-size: 0.88rem; font-weight: 700; color: #e8f0ff; margin-bottom: 4px; }
.stack-role { font-size: 0.74rem; color: #6b82a8; }

/* Achievement list */
.achieve-item {
    background: #0f1e35;
    border: 1px solid rgba(56,139,255,0.18);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.87rem;
    color: #8fa3c4;
    line-height: 1.55;
}

/* Disclaimer */
.disclaimer {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: #8fa3c4;
    line-height: 1.55;
    margin-top: 16px;
}

/* Divider */
.divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,139,255,0.18), transparent);
    margin: 48px 0;
}

/* Footer */
.footer {
    background: #0f1e35;
    border: 1px solid rgba(56,139,255,0.18);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-top: 48px;
}

.footer-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #e8f0ff;
    margin-bottom: 8px;
}

.footer-brand span { color: #388bff; }
.footer-info { font-size: 0.85rem; color: #6b82a8; line-height: 1.8; }

/* Upload area styling */
.uploadedFile { display: none; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #388bff, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    width: 100% !important;
    transition: all 0.3s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(56,139,255,0.4) !important;
}

div[data-testid="stFileUploader"] {
    background: #0f1e35 !important;
    border: 2px dashed rgba(56,139,255,0.3) !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #388bff, #6366f1) !important;
}

table {
    width: 100%;
    border-collapse: collapse;
    background: #0f1e35;
    border-radius: 12px;
    overflow: hidden;
}

th {
    background: rgba(56,139,255,0.1);
    padding: 12px 16px;
    text-align: left;
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8fa3c4;
}

td {
    padding: 12px 16px;
    font-size: 0.86rem;
    border-top: 1px solid rgba(56,139,255,0.1);
    color: #8fa3c4;
}

tr.best-row td { color: #00e5a0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load("vgg19_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">🫁 VGG19 Deep Learning · 92% Accuracy</div>
    <div class="hero-title">AI-Powered <span>Pneumonia</span> Detection</div>
    <div class="hero-desc">
        Upload a chest X-ray and get an instant AI diagnosis powered by VGG19 —
        trained on 5,863 pediatric radiographs from Guangzhou Women & Children's Medical Center.
        Built by Aarthi A R, Anna Adarsh College for Women.
    </div>
    <div class="hero-stats">
        <div class="stat-item"><div class="stat-num">92%</div><div class="stat-label">Test Accuracy</div></div>
        <div class="stat-item"><div class="stat-num">5,863</div><div class="stat-label">X-Ray Images</div></div>
        <div class="stat-item"><div class="stat-num">0.92</div><div class="stat-label">F1-Score</div></div>
        <div class="stat-item"><div class="stat-num">4</div><div class="stat-label">CNN Models</div></div>
        <div class="stat-item"><div class="stat-num">0.98</div><div class="stat-label">Pneumonia Precision</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">How It <span>Works</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">From raw chest X-ray to AI diagnosis in seconds — a 5-stage deep learning pipeline.</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
steps = [
    ("01", "Upload X-Ray", "Drag & drop or browse a JPEG/PNG chest radiograph."),
    ("02", "Preprocess", "Resized to 224×224 px and pixel-normalized to 0–1."),
    ("03", "VGG19 Inference", "16 convolutional layers extract deep radiographic features."),
    ("04", "Classification", "FC layers output Normal vs Pneumonia probabilities."),
    ("05", "Result", "Diagnosis & confidence score displayed instantly."),
]
for col, (num, title, desc) in zip([c1,c2,c3,c4,c5], steps):
    with col:
        st.markdown(f"""
        <div style="text-align:center;padding:10px;">
            <div style="width:60px;height:60px;background:#0f1e35;border:1px solid rgba(56,139,255,0.18);
                border-radius:50%;display:flex;align-items:center;justify-content:center;
                margin:0 auto 14px;font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;color:#388bff;">
                {num}
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;color:#e8f0ff;margin-bottom:6px;">{title}</div>
            <div style="font-size:0.76rem;color:#6b82a8;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# ANALYZER SECTION
# ---------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Live Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Chest X-Ray <span>Analyzer</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Upload a chest X-ray image to get an instant AI-powered pneumonia prediction.</div>', unsafe_allow_html=True)

col_upload, col_result = st.columns(2)

with col_upload:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your X-ray here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image (JPG, JPEG, PNG)"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Uploaded Chest X-Ray")
        analyze = st.button("🔍 Analyze X-Ray")
    else:
        analyze = False
    st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    if uploaded_file and analyze:
        with st.spinner("Analyzing X-ray with VGG19..."):
            model = load_model()
            if model:
                img_array = preprocess_image(image)
                preds = model.predict(img_array)

                if preds.shape[-1] == 1:
                    confidence = float(preds[0][0])
                    prediction = "PNEUMONIA" if confidence >= 0.5 else "NORMAL"
                    if prediction == "NORMAL":
                        confidence = 1 - confidence
                else:
                    class_idx = int(np.argmax(preds[0]))
                    confidence = float(preds[0][class_idx])
                    prediction = "PNEUMONIA" if class_idx == 1 else "NORMAL"

                pct = round(confidence * 100)

                if prediction == "PNEUMONIA":
                    st.markdown(f"""
                    <div class="result-card-pneumonia">
                        <div style="font-size:2.5rem;margin-bottom:12px;">⚠️</div>
                        <div class="result-title-pneumonia">Pneumonia Detected</div>
                        <div class="result-desc">The VGG19 model identified patterns consistent with pneumonia.
                        Please seek immediate medical evaluation.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {pct}%")
                    st.progress(confidence)
                else:
                    st.markdown(f"""
                    <div class="result-card-normal">
                        <div style="font-size:2.5rem;margin-bottom:12px;">✅</div>
                        <div class="result-title-normal">No Pneumonia Found</div>
                        <div class="result-desc">The VGG19 model found no significant indicators of pneumonia.
                        Lung patterns appear normal. Always confirm with a doctor.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {pct}%")
                    st.progress(confidence)

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f"""
                    <div class="perf-card">
                        <div class="perf-val" style="color:#388bff">{prediction}</div>
                        <div class="perf-label">Predicted Class</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class="perf-card">
                        <div class="perf-val" style="color:#388bff">{pct}%</div>
                        <div class="perf-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer">
                    ⚠️ For research & educational use only. Always consult a qualified
                    medical professional for diagnosis and treatment.
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card" style="min-height:300px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;text-align:center;">
            <div style="font-size:3rem;margin-bottom:16px;opacity:0.3;">🔬</div>
            <div style="color:#6b82a8;font-size:0.9rem;">
                Upload a chest X-ray and click<br/><strong style="color:#8fa3c4;">Analyze X-Ray</strong>
                to see the AI result here.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# DATASET SECTION
# ---------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Training Data</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Dataset & <span>Preprocessing</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Sourced from Kaggle\'s Chest X-Ray Images dataset — real-world pediatric radiographs, expert-graded and carefully preprocessed.</div>', unsafe_allow_html=True)

col_ds1, col_ds2 = st.columns(2)
with col_ds1:
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown('<div class="stat-card"><div class="stat-card-num">5,863</div><div class="stat-card-label">Total Images</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-card"><div class="stat-card-num">624</div><div class="stat-card-label">Test Set</div></div>', unsafe_allow_html=True)
    with sc2:
        st.markdown('<div class="stat-card"><div class="stat-card-num">5,216</div><div class="stat-card-label">Training Set</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-card"><div class="stat-card-num">2</div><div class="stat-card-label">Classes</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Dataset Details</h4>
        <div style="display:flex;gap:10px;margin-bottom:10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;">Source</span>
            <span style="font-size:0.8rem;color:#8fa3c4;">Kaggle — Chest X-Ray Images (Pneumonia)</span>
        </div>
        <div style="display:flex;gap:10px;margin-bottom:10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;">Patients</span>
            <span style="font-size:0.8rem;color:#8fa3c4;">Pediatric patients aged 1–5 years</span>
        </div>
        <div style="display:flex;gap:10px;margin-bottom:10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;">Origin</span>
            <span style="font-size:0.8rem;color:#8fa3c4;">Guangzhou Women & Children's Medical Center</span>
        </div>
        <div style="display:flex;gap:10px;margin-bottom:10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;">Format</span>
            <span style="font-size:0.8rem;color:#8fa3c4;">JPEG greyscale chest radiographs</span>
        </div>
        <div style="display:flex;gap:10px;margin-bottom:10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;">Labels</span>
            <span style="font-size:0.8rem;color:#8fa3c4;">NORMAL vs PNEUMONIA (binary classification)</span>
        </div>
        <div style="display:flex;gap:10px;">
            <span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;">Quality</span>
            <span style="font-size:0.8rem;color:#8fa3c4;">Expert physician grading; poor-quality scans removed</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_ds2:
    st.markdown("""
    <div class="info-card">
        <h4>Preprocessing Pipeline</h4>
        <div style="display:flex;gap:12px;margin-bottom:12px;">
            <div style="width:26px;height:26px;background:linear-gradient(135deg,#388bff,#6366f1);border-radius:50%;
                display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;
                color:white;flex-shrink:0;">1</div>
            <div style="font-size:0.83rem;color:#8fa3c4;padding-top:3px;">Resize all images to <strong style="color:#e8f0ff;">224×224 pixels</strong> for model input uniformity.</div>
        </div>
        <div style="display:flex;gap:12px;margin-bottom:12px;">
            <div style="width:26px;height:26px;background:linear-gradient(135deg,#388bff,#6366f1);border-radius:50%;
                display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;
                color:white;flex-shrink:0;">2</div>
            <div style="font-size:0.83rem;color:#8fa3c4;padding-top:3px;"><strong style="color:#e8f0ff;">Pixel normalization</strong> — scale values to 0–1 range for stable training.</div>
        </div>
        <div style="display:flex;gap:12px;margin-bottom:12px;">
            <div style="width:26px;height:26px;background:linear-gradient(135deg,#388bff,#6366f1);border-radius:50%;
                display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;
                color:white;flex-shrink:0;">3</div>
            <div style="font-size:0.83rem;color:#8fa3c4;padding-top:3px;"><strong style="color:#e8f0ff;">Oversampling</strong> to balance Normal & Pneumonia classes to 50/50 split.</div>
        </div>
        <div style="display:flex;gap:12px;margin-bottom:12px;">
            <div style="width:26px;height:26px;background:linear-gradient(135deg,#388bff,#6366f1);border-radius:50%;
                display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;
                color:white;flex-shrink:0;">4</div>
            <div style="font-size:0.83rem;color:#8fa3c4;padding-top:3px;"><strong style="color:#e8f0ff;">Data Augmentation</strong> — rotations, flips, zoom, shifts to improve generalization.</div>
        </div>
        <div style="display:flex;gap:12px;">
            <div style="width:26px;height:26px;background:linear-gradient(135deg,#388bff,#6366f1);border-radius:50%;
                display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;
                color:white;flex-shrink:0;">5</div>
            <div style="font-size:0.83rem;color:#8fa3c4;padding-top:3px;"><strong style="color:#e8f0ff;">Train / Validation / Test</strong> split preserved — 5,216 / 8 / 624 images.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# MODELS SECTION
# ---------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Architecture Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">CNN <span>Models</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Four deep learning architectures trained and evaluated. VGG19 achieved the best test accuracy of 92%.</div>', unsafe_allow_html=True)

mc1, mc2 = st.columns(2)
models = [
    ("Basic CNN", "Custom 4-Convolutional Layer Architecture", "Conv → ReLU → MaxPool → Dropout → FC → Sigmoid. Baseline architecture built from scratch as a performance benchmark.", 88, "#388bff", False),
    ("Xception", "Entry / Middle / Exit Flow Architecture", "Depthwise separable convolutions for efficient feature extraction with fewer parameters. Transfer learning from ImageNet weights.", 82, "#a78bfa", False),
    ("VGG19 ★", "16 Convolutional + 3 Fully Connected Layers", "Small 3×3 filters with uniform deep architecture. Transfer learning + fine-tuned top layers. Best accuracy: 92%, F1-Score: 0.92.", 92, "#00e5a0", True),
    ("ResNet-50", "50-Layer Residual Network", "Skip connections solve the vanishing gradient problem. Residual learning enables deep feature extraction from complex X-ray patterns.", 85, "#f59e0b", False),
]

for i, (name, arch, desc, acc, color, best) in enumerate(models):
    col = mc1 if i % 2 == 0 else mc2
    with col:
        border = "rgba(0,229,160,0.4)" if best else "rgba(56,139,255,0.18)"
        badge = '<span style="background:rgba(0,229,160,0.15);border:1px solid rgba(0,229,160,0.3);color:#00e5a0;font-size:0.68rem;font-weight:700;padding:3px 10px;border-radius:100px;float:right;">★ Best Model</span>' if best else ''
        st.markdown(f"""
        <div style="background:#0f1e35;border:1px solid {border};border-radius:16px;padding:24px;margin-bottom:16px;">
            {badge}
            <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:#e8f0ff;margin-bottom:4px;">{name}</div>
            <div style="font-size:0.76rem;color:#388bff;margin-bottom:8px;">{arch}</div>
            <div style="font-size:0.81rem;color:#8fa3c4;line-height:1.6;margin-bottom:16px;">{desc}</div>
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:0.72rem;color:#6b82a8;text-transform:uppercase;width:76px;flex-shrink:0;">Test Accuracy</span>
                <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:100px;height:7px;overflow:hidden;">
                    <div style="width:{acc}%;height:100%;background:{color};border-radius:100px;"></div>
                </div>
                <span style="font-family:'Syne',sans-serif;font-size:0.88rem;font-weight:700;color:{color};width:36px;text-align:right;">{acc}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# RESULTS SECTION
# ---------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Performance Metrics</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Results & <span>Metrics</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">VGG19 outperformed all architectures across every metric on the 624-image held-out test set.</div>', unsafe_allow_html=True)

p1,p2,p3,p4,p5,p6 = st.columns(6)
perfs = [
    ("92%","#00e5a0","Test Accuracy"),("0.92","#388bff","F1-Score"),
    ("0.98","#a78bfa","Pneumonia Precision"),("0.97","#f59e0b","Normal Recall"),
    ("347","#00e5a0","Pneumonia Correct"),("227","#388bff","Normal Correct"),
]
for col, (val, color, label) in zip([p1,p2,p3,p4,p5,p6], perfs):
    with col:
        st.markdown(f'<div class="perf-card"><div class="perf-val" style="color:{color}">{val}</div><div class="perf-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:16px;overflow:hidden;margin-top:8px;">
<table>
<thead><tr><th>Model</th><th>Train Acc.</th><th>Val Acc.</th><th>Test Acc.</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>
<tbody>
<tr><td>Basic CNN</td><td>91%</td><td>92%</td><td>88%</td><td>0.88</td><td>0.88</td><td>0.88</td></tr>
<tr><td>Xception</td><td>91%</td><td>92%</td><td>82%</td><td>0.85</td><td>0.82</td><td>0.83</td></tr>
<tr class="best-row"><td>VGG19 ★</td><td>94%</td><td>94%</td><td>92%</td><td>0.93</td><td>0.92</td><td>0.92</td></tr>
<tr><td>ResNet-50</td><td>87%</td><td>91%</td><td>85%</td><td>0.87</td><td>0.85</td><td>0.85</td></tr>
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# TECH STACK
# ---------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Technology</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Software <span>Stack</span></div>', unsafe_allow_html=True)

sk1,sk2,sk3,sk4,sk5,sk6,sk7,sk8 = st.columns(8)
stacks = [
    ("🐍","Python 3.x","Core Language"),("🧠","TensorFlow","Deep Learning"),
    ("🖼️","OpenCV+Pillow","Image Processing"),("🔢","NumPy/Pandas","Data Analysis"),
    ("📊","Matplotlib","Visualization"),("🔬","Scikit-learn","Metrics"),
    ("🌐","Streamlit","Web App"),("📓","Jupyter","Dev Environment"),
]
for col, (icon, name, role) in zip([sk1,sk2,sk3,sk4,sk5,sk6,sk7,sk8], stacks):
    with col:
        st.markdown(f'<div class="stack-card"><div class="stack-icon">{icon}</div><div class="stack-name">{name}</div><div class="stack-role">{role}</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# CONCLUSION
# ---------------------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-tag">Conclusion</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Key Findings & <span>Future Work</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Deep Learning × Medical Imaging = Faster, Accessible Pneumonia Diagnosis.</div>', unsafe_allow_html=True)

cc1, cc2 = st.columns(2)
with cc1:
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:700;color:#00e5a0;margin-bottom:16px;font-size:1.05rem;">✓ Key Achievements</div>', unsafe_allow_html=True)
    achievements = [
        "Dataset balanced using oversampling — achieving 50/50 class split.",
        "4 CNN architectures trained and evaluated on 5,863 chest X-rays.",
        "VGG19 achieved best test accuracy of 92% with F1-Score 0.92.",
        "Transfer learning significantly improved model robustness.",
        "Real-time web app deployed for instant pneumonia detection.",
        "Prediction probabilities provide transparent, interpretable diagnostics.",
    ]
    for a in achievements:
        st.markdown(f'<div class="achieve-item">✓ &nbsp;{a}</div>', unsafe_allow_html=True)

with cc2:
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:700;color:#388bff;margin-bottom:16px;font-size:1.05rem;">→ Future Directions</div>', unsafe_allow_html=True)
    futures = [
        "Distinguish Bacterial vs Viral pneumonia subtypes for finer diagnosis.",
        "Explore InceptionV3, MobileNet, and ShuffleNet architectures.",
        "Hyperparameter optimization to push accuracy beyond 92%.",
        "Real-time optimization for clinical deployment at scale.",
        "Cloud deployment for broader hospital accessibility.",
        "Integrate with EHR systems as a clinical decision support tool.",
    ]
    for f in futures:
        st.markdown(f'<div class="achieve-item">→ &nbsp;{f}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("""
<div class="footer">
    <div class="footer-brand">Pneumo<span>Scan</span> AI</div>
    <div class="footer-info">
        Approach for Pneumonia Detection via Image Classification<br/>
        <strong style="color:#a78bfa;">Aarthi A R</strong> &nbsp;|&nbsp;
        Anna Adarsh College for Women (Autonomous)<br/>
        Guide: <strong style="color:#a78bfa;">Dr. Hannah Vijaykumar</strong><br/>
        <span style="font-size:0.76rem;color:#4a5f80;">Review II · February 2026 · Design & Implementation Stage</span>
    </div>
</div>
""", unsafe_allow_html=True)
