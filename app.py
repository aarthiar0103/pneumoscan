import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="PneumoScan AI — Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;} .stDeployButton {display:none;}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.stApp{background-color:#050d1a;font-family:'Inter',sans-serif!important;}
.stTabs [data-baseweb="tab-list"]{background:#0b1628;border-radius:12px;padding:4px;gap:4px;border:1px solid rgba(56,139,255,0.15);margin-bottom:32px;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:#6b82a8!important;font-family:'Inter',sans-serif!important;font-weight:500;font-size:0.88rem;padding:8px 20px;border:none;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#388bff,#6366f1)!important;color:white!important;font-weight:600!important;}
.stTabs [data-baseweb="tab-highlight"]{display:none;}
.hero-section{background:linear-gradient(135deg,#0b1628 0%,#0f1e35 100%);border:1px solid rgba(56,139,255,0.18);border-radius:20px;padding:52px 48px;margin-bottom:32px;position:relative;overflow:hidden;text-align:center;}
.hero-section::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#388bff,transparent);}
.hero-badge{display:inline-block;background:rgba(56,139,255,0.1);border:1px solid rgba(56,139,255,0.3);border-radius:100px;padding:6px 18px;font-size:11px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#388bff;margin-bottom:20px;font-family:'Inter',sans-serif;}
.hero-title{font-family:'Inter',sans-serif;font-size:3rem;font-weight:800;line-height:1.1;letter-spacing:-0.03em;color:#e8f0ff;margin-bottom:32px;}
.hero-title span{background:linear-gradient(135deg,#388bff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-stats{display:flex;justify-content:center;gap:56px;flex-wrap:wrap;}
.stat-item{text-align:center;}
.stat-num{font-family:'Inter',sans-serif;font-size:2rem;font-weight:800;color:#388bff;}
.stat-label{font-size:0.72rem;color:#6b82a8;text-transform:uppercase;letter-spacing:0.1em;}
.section-tag{display:inline-block;background:rgba(56,139,255,0.1);border:1px solid rgba(56,139,255,0.2);border-radius:100px;padding:4px 14px;font-size:11px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:#388bff;margin-bottom:10px;font-family:'Inter',sans-serif;}
.section-title{font-family:'Inter',sans-serif;font-size:1.9rem;font-weight:800;color:#e8f0ff;margin-bottom:6px;letter-spacing:-0.02em;}
.section-title span{background:linear-gradient(135deg,#388bff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.section-desc{color:#6b82a8;font-size:0.93rem;line-height:1.7;margin-bottom:32px;font-family:'Inter',sans-serif;}
.info-card{background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:16px;padding:26px;margin-bottom:16px;}
.info-card h4{font-family:'Inter',sans-serif;font-size:0.97rem;font-weight:700;color:#e8f0ff;margin-bottom:16px;}
.stat-card{background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:14px;padding:22px;text-align:center;margin-bottom:14px;}
.stat-card-num{font-family:'Inter',sans-serif;font-size:2rem;font-weight:800;color:#e8f0ff;margin-bottom:4px;}
.stat-card-label{font-size:0.76rem;color:#6b82a8;text-transform:uppercase;letter-spacing:0.1em;}
.result-card-pneumonia{background:rgba(255,77,109,0.08);border:1px solid rgba(255,77,109,0.35);border-radius:16px;padding:28px;text-align:center;}
.result-card-normal{background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.35);border-radius:16px;padding:28px;text-align:center;}
.result-title-pneumonia{font-family:'Inter',sans-serif;font-size:1.7rem;font-weight:800;color:#ff4d6d;margin-bottom:8px;}
.result-title-normal{font-family:'Inter',sans-serif;font-size:1.7rem;font-weight:800;color:#00e5a0;margin-bottom:8px;}
.result-desc{color:#8fa3c4;font-size:0.88rem;line-height:1.6;}
.perf-card{background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:14px;padding:20px;text-align:center;margin-bottom:14px;}
.perf-val{font-family:'Inter',sans-serif;font-size:1.7rem;font-weight:800;margin-bottom:4px;}
.perf-label{font-size:0.72rem;color:#6b82a8;text-transform:uppercase;letter-spacing:0.1em;}
.stack-card{background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:14px;padding:20px;text-align:center;margin-bottom:14px;}
.stack-icon{font-size:1.8rem;margin-bottom:10px;}
.stack-name{font-family:'Inter',sans-serif;font-size:0.87rem;font-weight:700;color:#e8f0ff;margin-bottom:4px;}
.stack-role{font-size:0.73rem;color:#6b82a8;}
.achieve-item{background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:10px;padding:12px 16px;margin-bottom:10px;font-size:0.86rem;color:#8fa3c4;line-height:1.55;}
.disclaimer{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:12px 16px;font-size:0.78rem;color:#8fa3c4;line-height:1.55;margin-top:16px;}
.footer{background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:16px;padding:32px;text-align:center;margin-top:40px;}
.footer-brand{font-family:'Inter',sans-serif;font-size:1.4rem;font-weight:800;color:#e8f0ff;margin-bottom:8px;}
.footer-brand span{color:#388bff;}
.footer-info{font-size:0.84rem;color:#6b82a8;line-height:1.9;}
.stButton>button{background:linear-gradient(135deg,#388bff,#6366f1)!important;color:white!important;border:none!important;border-radius:12px!important;font-family:'Inter',sans-serif!important;font-weight:600!important;font-size:0.95rem!important;padding:12px 32px!important;width:100%!important;}
div[data-testid="stFileUploader"]{background:#0f1e35!important;border:2px dashed rgba(56,139,255,0.3)!important;border-radius:14px!important;padding:20px!important;}
table{width:100%;border-collapse:collapse;background:#0f1e35;border-radius:12px;overflow:hidden;font-family:'Inter',sans-serif;}
th{background:rgba(56,139,255,0.1);padding:12px 16px;text-align:left;font-size:0.77rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#8fa3c4;}
td{padding:12px 16px;font-size:0.85rem;border-top:1px solid rgba(56,139,255,0.1);color:#8fa3c4;}
tr.best-row td{color:#00e5a0;font-weight:600;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        from tensorflow.keras.models import load_model as keras_load
        return keras_load("vgg19_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# HERO
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">🫁 VGG19 Deep Learning</div>
    <div class="hero-title">AI-Powered <span>Pneumonia</span> Detection</div>
</div>
""", unsafe_allow_html=True)

# TABS
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "🔬 Analyzer","ℹ️ How It Works","📊 Dataset","🧠 Models","📈 Results","🛠 Tech Stack","✅ Conclusion"
])

# TAB 1 — ANALYZER
with tab1:
    st.markdown('<div class="section-tag">Live Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Chest X-Ray <span>Analyzer</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Upload a chest X-ray image to get an instant AI-powered pneumonia prediction using our VGG19 model.</div>', unsafe_allow_html=True)

    col_u, col_r = st.columns(2)
    with col_u:
        uploaded_file = st.file_uploader("Drop your X-ray here or click to browse", type=["jpg","jpeg","png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded Chest X-Ray")
            analyze = st.button("🔍 Analyze X-Ray")
        else:
            analyze = False

    with col_r:
        if uploaded_file and analyze:
            with st.spinner("Analyzing with VGG19..."):
                model = load_model()
                if model:
                    preds = model.predict(preprocess_image(image))
                    if preds.shape[-1] == 1:
                        confidence = float(preds[0][0])
                        prediction = "PNEUMONIA" if confidence >= 0.5 else "NORMAL"
                        if prediction == "NORMAL": confidence = 1 - confidence
                    else:
                        class_idx = int(np.argmax(preds[0]))
                        confidence = float(preds[0][class_idx])
                        prediction = "PNEUMONIA" if class_idx == 1 else "NORMAL"
                    pct = round(confidence * 100)
                    if prediction == "PNEUMONIA":
                        st.markdown(f'<div class="result-card-pneumonia"><div style="font-size:2.5rem;margin-bottom:12px;">⚠️</div><div class="result-title-pneumonia">Pneumonia Detected</div><div class="result-desc">The VGG19 model identified patterns consistent with pneumonia. Please seek immediate medical evaluation.</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-card-normal"><div style="font-size:2.5rem;margin-bottom:12px;">✅</div><div class="result-title-normal">No Pneumonia Found</div><div class="result-desc">No significant indicators of pneumonia found. Lung patterns appear normal. Always confirm with a doctor.</div></div>', unsafe_allow_html=True)
                    st.markdown('<div class="disclaimer">⚠️ For research & educational use only. Always consult a qualified medical professional for diagnosis and treatment.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card" style="min-height:320px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;"><div style="font-size:3rem;margin-bottom:16px;opacity:0.25;">🔬</div><div style="color:#6b82a8;font-size:0.9rem;line-height:1.7;">Upload a chest X-ray on the left<br/>and click <strong style="color:#8fa3c4;">Analyze X-Ray</strong> to see the AI result here.</div></div>', unsafe_allow_html=True)

# TAB 2 — HOW IT WORKS
with tab2:
    st.markdown('<div class="section-tag">Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How It <span>Works</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">A step-by-step overview of how the AI analyzes your chest X-ray image for pneumonia detection.</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(num,title,desc) in zip([c1,c2,c3,c4,c5],[
        ("01","Upload X-Ray","Drop a JPEG/PNG chest radiograph."),
        ("02","Preprocess","Resized to 224×224 px, normalized."),
        ("03","VGG19","16 conv layers extract features."),
        ("04","Classify","FC layers output probabilities."),
        ("05","Result","Diagnosis shown instantly."),
    ]):
        with col:
            st.markdown(f"""<div style="text-align:center;padding:12px;background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:12px;margin-bottom:24px;">
                <div style="width:46px;height:46px;background:linear-gradient(135deg,#388bff,#6366f1);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 10px;font-family:'Inter',sans-serif;font-size:0.82rem;font-weight:800;color:white;">{num}</div>
                <div style="font-family:'Inter',sans-serif;font-size:0.81rem;font-weight:700;color:#e8f0ff;margin-bottom:4px;">{title}</div>
                <div style="font-size:0.72rem;color:#6b82a8;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    hw1,hw2,hw3 = st.columns(3)
    for col,(val,color,label) in zip([hw1,hw2,hw3],[("92%","#00e5a0","Test Accuracy"),("0.92","#388bff","F1-Score"),("0.98","#a78bfa","Pneumonia Precision")]):
        with col: st.markdown(f'<div class="perf-card"><div class="perf-val" style="color:{color}">{val}</div><div class="perf-label">{label}</div></div>', unsafe_allow_html=True)

# TAB 3 — DATASET
with tab3:
    st.markdown('<div class="section-tag">Training Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset & <span>Preprocessing</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Sourced from Kaggle\'s Chest X-Ray Images dataset — real-world pediatric radiographs, expert-graded and carefully preprocessed.</div>', unsafe_allow_html=True)
    d1,d2 = st.columns(2)
    with d1:
        s1,s2 = st.columns(2)
        with s1:
            st.markdown('<div class="stat-card"><div class="stat-card-num">5,863</div><div class="stat-card-label">Total Images</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-card"><div class="stat-card-num">624</div><div class="stat-card-label">Test Set</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="stat-card"><div class="stat-card-num">5,216</div><div class="stat-card-label">Training Set</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stat-card"><div class="stat-card-num">2</div><div class="stat-card-label">Classes</div></div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card"><h4>Dataset Details</h4>
            <div style="display:flex;gap:10px;margin-bottom:10px;"><span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;font-family:'Inter',sans-serif;">Source</span><span style="font-size:0.8rem;color:#8fa3c4;font-family:'Inter',sans-serif;">Kaggle — Chest X-Ray Images (Pneumonia)</span></div>
            <div style="display:flex;gap:10px;margin-bottom:10px;"><span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;font-family:'Inter',sans-serif;">Patients</span><span style="font-size:0.8rem;color:#8fa3c4;font-family:'Inter',sans-serif;">Pediatric patients aged 1–5 years</span></div>
            <div style="display:flex;gap:10px;margin-bottom:10px;"><span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;font-family:'Inter',sans-serif;">Origin</span><span style="font-size:0.8rem;color:#8fa3c4;font-family:'Inter',sans-serif;">Guangzhou Women & Children's Medical Center</span></div>
            <div style="display:flex;gap:10px;margin-bottom:10px;"><span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;font-family:'Inter',sans-serif;">Format</span><span style="font-size:0.8rem;color:#8fa3c4;font-family:'Inter',sans-serif;">JPEG greyscale chest radiographs</span></div>
            <div style="display:flex;gap:10px;margin-bottom:10px;"><span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;font-family:'Inter',sans-serif;">Labels</span><span style="font-size:0.8rem;color:#8fa3c4;font-family:'Inter',sans-serif;">NORMAL vs PNEUMONIA (binary classification)</span></div>
            <div style="display:flex;gap:10px;"><span style="font-size:0.78rem;font-weight:600;color:#388bff;min-width:76px;font-family:'Inter',sans-serif;">Quality</span><span style="font-size:0.8rem;color:#8fa3c4;font-family:'Inter',sans-serif;">Expert physician grading; poor-quality scans removed</span></div>
        </div>""", unsafe_allow_html=True)
    with d2:
        steps_pipe = [("1","Resize all images to <strong style='color:#e8f0ff;'>224×224 pixels</strong> for model input uniformity."),("2","<strong style='color:#e8f0ff;'>Pixel normalization</strong> — scale values to 0–1 range for stable training."),("3","<strong style='color:#e8f0ff;'>Oversampling</strong> to balance Normal & Pneumonia classes to 50/50 split."),("4","<strong style='color:#e8f0ff;'>Data Augmentation</strong> — rotations, flips, zoom, shifts to improve generalization."),("5","<strong style='color:#e8f0ff;'>Train / Validation / Test</strong> split preserved — 5,216 / 8 / 624 images.")]
        pipe_html = '<div class="info-card"><h4>Preprocessing Pipeline</h4>'
        for n,t in steps_pipe:
            pipe_html += f'<div style="display:flex;gap:12px;margin-bottom:14px;"><div style="width:26px;height:26px;background:linear-gradient(135deg,#388bff,#6366f1);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;color:white;flex-shrink:0;font-family:Inter,sans-serif;">{n}</div><div style="font-size:0.83rem;color:#8fa3c4;padding-top:3px;font-family:Inter,sans-serif;">{t}</div></div>'
        pipe_html += '</div>'
        st.markdown(pipe_html, unsafe_allow_html=True)

# TAB 4 — MODELS
with tab4:
    st.markdown('<div class="section-tag">Architecture Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">CNN <span>Models</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Four deep learning architectures trained and evaluated. VGG19 achieved the best test accuracy of 92%.</div>', unsafe_allow_html=True)
    mc1,mc2 = st.columns(2)
    for i,(name,arch,desc,acc,color,best) in enumerate([
        ("Basic CNN","Custom 4-Convolutional Layer Architecture","Conv → ReLU → MaxPool → Dropout → FC → Sigmoid. Baseline architecture built from scratch.",88,"#388bff",False),
        ("Xception","Entry / Middle / Exit Flow Architecture","Depthwise separable convolutions. Better performance with fewer parameters. Transfer learning applied.",82,"#a78bfa",False),
        ("VGG19 ★","16 Convolutional + 3 Fully Connected Layers","Small 3×3 filters, uniform architecture. Transfer learning + fine-tuned top layers. Best: 92% accuracy.",92,"#00e5a0",True),
        ("ResNet-50","50-Layer Residual Network","Skip connections solve vanishing gradient. Residual learning for deep feature extraction.",85,"#f59e0b",False),
    ]):
        col = mc1 if i%2==0 else mc2
        with col:
            border = "rgba(0,229,160,0.4)" if best else "rgba(56,139,255,0.18)"
            badge = f'<span style="background:rgba(0,229,160,0.15);border:1px solid rgba(0,229,160,0.3);color:#00e5a0;font-size:0.68rem;font-weight:700;padding:3px 10px;border-radius:100px;float:right;font-family:Inter,sans-serif;">★ Best</span>' if best else ''
            st.markdown(f"""<div style="background:#0f1e35;border:1px solid {border};border-radius:16px;padding:24px;margin-bottom:16px;">
                {badge}
                <div style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:800;color:#e8f0ff;margin-bottom:4px;">{name}</div>
                <div style="font-size:0.76rem;color:#388bff;margin-bottom:8px;font-family:'Inter',sans-serif;">{arch}</div>
                <div style="font-size:0.81rem;color:#8fa3c4;line-height:1.6;margin-bottom:16px;font-family:'Inter',sans-serif;">{desc}</div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:0.72rem;color:#6b82a8;text-transform:uppercase;width:76px;flex-shrink:0;font-family:'Inter',sans-serif;">Test Acc.</span>
                    <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:100px;height:7px;overflow:hidden;">
                        <div style="width:{acc}%;height:100%;background:{color};border-radius:100px;"></div>
                    </div>
                    <span style="font-family:'Inter',sans-serif;font-size:0.88rem;font-weight:700;color:{color};width:36px;text-align:right;">{acc}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

# TAB 5 — RESULTS
with tab5:
    st.markdown('<div class="section-tag">Performance Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Results & <span>Metrics</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">VGG19 outperformed all architectures across every metric on the 624-image held-out test set.</div>', unsafe_allow_html=True)
    p1,p2,p3,p4,p5,p6 = st.columns(6)
    for col,(val,color,label) in zip([p1,p2,p3,p4,p5,p6],[("92%","#00e5a0","Test Accuracy"),("0.92","#388bff","F1-Score"),("0.98","#a78bfa","Pneumonia Precision"),("0.97","#f59e0b","Normal Recall"),("347","#00e5a0","Pneumonia Correct"),("227","#388bff","Normal Correct")]):
        with col: st.markdown(f'<div class="perf-card"><div class="perf-val" style="color:{color}">{val}</div><div class="perf-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("""<div style="background:#0f1e35;border:1px solid rgba(56,139,255,0.18);border-radius:16px;overflow:hidden;margin-top:8px;">
    <table><thead><tr><th>Model</th><th>Train Acc.</th><th>Val Acc.</th><th>Test Acc.</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>
    <tbody>
    <tr><td>Basic CNN</td><td>91%</td><td>92%</td><td>88%</td><td>0.88</td><td>0.88</td><td>0.88</td></tr>
    <tr><td>Xception</td><td>91%</td><td>92%</td><td>82%</td><td>0.85</td><td>0.82</td><td>0.83</td></tr>
    <tr class="best-row"><td>VGG19 ★</td><td>94%</td><td>94%</td><td>92%</td><td>0.93</td><td>0.92</td><td>0.92</td></tr>
    <tr><td>ResNet-50</td><td>87%</td><td>91%</td><td>85%</td><td>0.87</td><td>0.85</td><td>0.85</td></tr>
    </tbody></table></div>""", unsafe_allow_html=True)

# TAB 6 — TECH STACK
with tab6:
    st.markdown('<div class="section-tag">Technology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Software <span>Stack</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Built with production-grade open-source tools for deep learning, image processing, and web deployment.</div>', unsafe_allow_html=True)
    r1 = st.columns(4)
    r2 = st.columns(4)
    for col,(icon,name,role) in zip(r1,[("🐍","Python 3.x","Core Language"),("🧠","TensorFlow / Keras","Deep Learning"),("🖼️","OpenCV + Pillow","Image Processing"),("🔢","NumPy / Pandas","Data Analysis")]):
        with col: st.markdown(f'<div class="stack-card"><div class="stack-icon">{icon}</div><div class="stack-name">{name}</div><div class="stack-role">{role}</div></div>', unsafe_allow_html=True)
    for col,(icon,name,role) in zip(r2,[("📊","Matplotlib / Seaborn","Visualization"),("🔬","Scikit-learn","Evaluation Metrics"),("🌐","Streamlit","Web Application"),("📓","Jupyter / Anaconda","Dev Environment")]):
        with col: st.markdown(f'<div class="stack-card"><div class="stack-icon">{icon}</div><div class="stack-name">{name}</div><div class="stack-role">{role}</div></div>', unsafe_allow_html=True)

# TAB 7 — CONCLUSION
with tab7:
    st.markdown('<div class="section-tag">Conclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Findings & <span>Future Work</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Deep Learning × Medical Imaging = Faster, Accessible Pneumonia Diagnosis.</div>', unsafe_allow_html=True)
    cc1,cc2 = st.columns(2)
    with cc1:
        st.markdown('<div style="font-family:\'Inter\',sans-serif;font-weight:700;color:#00e5a0;margin-bottom:16px;font-size:1rem;">✓ Key Achievements</div>', unsafe_allow_html=True)
        for a in ["Dataset balanced using oversampling — achieving 50/50 class split.","4 CNN architectures trained and evaluated on 5,863 chest X-rays.","VGG19 achieved best test accuracy of 92% with F1-Score 0.92.","Transfer learning significantly improved model robustness.","Real-time web app deployed for instant pneumonia detection.","Prediction probabilities provide transparent, interpretable diagnostics."]:
            st.markdown(f'<div class="achieve-item">✓ &nbsp;{a}</div>', unsafe_allow_html=True)
    with cc2:
        st.markdown('<div style="font-family:\'Inter\',sans-serif;font-weight:700;color:#388bff;margin-bottom:16px;font-size:1rem;">→ Future Directions</div>', unsafe_allow_html=True)
        for f in ["Distinguish Bacterial vs Viral pneumonia subtypes for finer diagnosis.","Explore InceptionV3, MobileNet, and ShuffleNet architectures.","Hyperparameter optimization to push accuracy beyond 92%.", "Real-time optimization for clinical deployment at scale.","Cloud deployment for broader hospital accessibility.","Integrate with EHR systems as a clinical decision support tool."]:
            st.markdown(f'<div class="achieve-item">→ &nbsp;{f}</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    <div class="footer-brand">Pneumo<span>Scan</span> AI</div>
    <div class="footer-info">
        AI-Powered Approach for Pneumonia Detection via Image Classification<br/>
        <span style="font-size:0.76rem;color:#4a5f80;">For research & educational use only · Always consult a qualified medical professional</span>
    </div>
</div>
""", unsafe_allow_html=True)
