from flask import Flask, request, jsonify, render_template_string
import numpy as np
import io
import os

app = Flask(__name__)


MODEL_PATH = "vgg19_model.h5"
model = None

def load_model():
    global model
    try:
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Load model on startup
load_model()

def preprocess_image(image_bytes):
    from PIL import Image
    from tensorflow.keras.preprocessing.image import img_to_array
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>PneumoScan AI — Chest X-Ray Pneumonia Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:#050d1a; --surface:#0b1628; --card:#0f1e35; --border:rgba(56,139,255,0.18);
  --accent:#388bff; --accent2:#6366f1; --danger:#ff4d6d; --safe:#00e5a0;
  --warn:#f59e0b; --text:#e8f0ff; --muted:#6b82a8; --muted2:#8fa3c4;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(56,139,255,0.035) 1px,transparent 1px),linear-gradient(90deg,rgba(56,139,255,0.035) 1px,transparent 1px);background-size:44px 44px;pointer-events:none;z-index:0;}
nav{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(5,13,26,0.88);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);padding:0 40px;}
.nav-inner{max-width:1100px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;height:64px;}
.nav-logo{font-family:'Syne',sans-serif;font-weight:800;font-size:1.25rem;}
.nav-logo span{color:var(--accent);}
.nav-links{display:flex;gap:28px;list-style:none;}
.nav-links a{color:var(--muted2);text-decoration:none;font-size:0.87rem;font-weight:500;transition:color 0.2s;}
.nav-links a:hover{color:var(--text);}
.nav-cta{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;font-size:0.8rem;font-weight:700;padding:8px 18px;border-radius:100px;text-decoration:none;font-family:'Syne',sans-serif;}
section{position:relative;z-index:1;}
.wrap{max-width:1100px;margin:0 auto;padding:100px 40px;}
.tag{display:inline-flex;align-items:center;gap:8px;background:rgba(56,139,255,0.1);border:1px solid var(--border);border-radius:100px;padding:5px 16px;font-size:11px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:var(--accent);margin-bottom:18px;}
.tag .dot{width:6px;height:6px;background:var(--accent);border-radius:50%;animation:blink 1.5s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}
h2.stitle{font-family:'Syne',sans-serif;font-size:clamp(1.8rem,4vw,2.6rem);font-weight:800;line-height:1.1;letter-spacing:-0.03em;margin-bottom:12px;}
h2.stitle span{background:linear-gradient(135deg,#388bff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.sdesc{color:var(--muted);font-size:0.97rem;line-height:1.7;max-width:600px;margin-bottom:52px;}
.divider{width:100%;height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);}
#hero{padding-top:64px;}
.hero-grid{max-width:1100px;margin:0 auto;padding:110px 40px 80px;display:grid;grid-template-columns:1fr 1fr;gap:60px;align-items:center;}
.hero-tag{display:inline-flex;align-items:center;gap:8px;background:rgba(56,139,255,0.1);border:1px solid var(--border);border-radius:100px;padding:6px 18px;font-size:11px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:var(--accent);margin-bottom:26px;}
.hero-tag .dot{width:7px;height:7px;background:var(--accent);border-radius:50%;animation:blink 1.5s ease-in-out infinite;}
h1{font-family:'Syne',sans-serif;font-size:clamp(2.4rem,5vw,3.8rem);font-weight:800;line-height:1.06;letter-spacing:-0.03em;margin-bottom:18px;}
h1 span{background:linear-gradient(135deg,#388bff 0%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-p{color:var(--muted2);font-size:1rem;line-height:1.75;margin-bottom:32px;}
.hero-stats{display:flex;gap:28px;margin-bottom:36px;}
.hs{text-align:center;}
.hs-num{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;color:var(--accent);}
.hs-label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;}
.hero-btns{display:flex;gap:12px;}
.btn-p{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border:none;border-radius:10px;padding:13px 26px;font-family:'Syne',sans-serif;font-size:0.92rem;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block;transition:all 0.3s;box-shadow:0 0 20px rgba(56,139,255,0.28);}
.btn-p:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(56,139,255,0.45);}
.btn-o{background:transparent;color:var(--text);border:1px solid var(--border);border-radius:10px;padding:13px 26px;font-family:'Syne',sans-serif;font-size:0.92rem;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block;transition:all 0.3s;}
.btn-o:hover{border-color:var(--accent);color:var(--accent);}
.hero-card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:26px;position:relative;overflow:hidden;}
.hero-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);}
.xray-box{background:#020810;border-radius:12px;aspect-ratio:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;border:1px dashed rgba(56,139,255,0.15);}
.xray-box svg{width:90px;height:90px;opacity:0.12;}
.xray-box p{color:var(--muted);font-size:0.8rem;}
.hc-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px;}
.hcs{background:rgba(56,139,255,0.06);border:1px solid var(--border);border-radius:10px;padding:12px;text-align:center;}
.hcs-n{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:var(--accent);}
.hcs-l{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;margin-top:2px;}
#how{background:var(--surface);}
.steps{display:grid;grid-template-columns:repeat(5,1fr);gap:0;position:relative;margin-top:8px;}
.steps::before{content:'';position:absolute;top:31px;left:10%;right:10%;height:1px;background:linear-gradient(90deg,transparent,var(--border),var(--border),var(--border),transparent);}
.step{text-align:center;padding:0 10px;}
.step-n{width:62px;height:62px;background:var(--card);border:1px solid var(--border);border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 18px;font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;color:var(--accent);position:relative;z-index:1;}
.step-t{font-family:'Syne',sans-serif;font-size:0.88rem;font-weight:700;margin-bottom:7px;}
.step-d{font-size:0.78rem;color:var(--muted);line-height:1.6;}
.az-grid{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start;}
.upload-card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:32px;position:relative;overflow:hidden;}
.upload-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:0.7;}
.drop-zone{border:2px dashed var(--border);border-radius:14px;padding:44px 20px;text-align:center;cursor:pointer;transition:all 0.3s;position:relative;overflow:hidden;}
.drop-zone:hover,.drop-zone.drag-over{border-color:var(--accent);background:rgba(56,139,255,0.05);}
.drop-zone input[type="file"]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
.up-icon{width:56px;height:56px;margin:0 auto 14px;background:rgba(56,139,255,0.12);border-radius:50%;display:flex;align-items:center;justify-content:center;}
.up-icon svg{width:24px;height:24px;stroke:var(--accent);}
.drop-label{font-family:'Syne',sans-serif;font-size:0.97rem;font-weight:700;margin-bottom:5px;}
.drop-sub{font-size:0.8rem;color:var(--muted);}
#preview-container{display:none;margin-top:20px;border-radius:12px;overflow:hidden;border:1px solid var(--border);}
#preview-img{width:100%;max-height:260px;object-fit:contain;background:#020810;display:block;}
.btn-analyze{display:flex;align-items:center;justify-content:center;gap:9px;width:100%;margin-top:18px;padding:14px 24px;background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;border-radius:12px;color:#fff;font-family:'Syne',sans-serif;font-size:0.93rem;font-weight:700;cursor:pointer;transition:all 0.3s;}
.btn-analyze:hover{transform:translateY(-2px);box-shadow:0 8px 26px rgba(56,139,255,0.4);}
.btn-analyze:disabled{opacity:0.45;cursor:not-allowed;transform:none;}
.spinner{width:17px;height:17px;border:2px solid rgba(255,255,255,0.25);border-top-color:#fff;border-radius:50%;animation:spin 0.7s linear infinite;display:none;}
@keyframes spin{to{transform:rotate(360deg)}}
.result-panel{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:32px;}
.result-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:300px;gap:14px;color:var(--muted);text-align:center;}
.result-empty svg{width:52px;height:52px;opacity:0.18;}
.result-empty p{font-size:0.87rem;line-height:1.6;}
#result-content{display:none;}
.res-badge{display:inline-flex;align-items:center;border-radius:100px;padding:7px 18px;font-family:'Syne',sans-serif;font-size:0.82rem;font-weight:700;margin-bottom:18px;}
.res-badge.pneumonia{background:rgba(255,77,109,0.15);border:1px solid rgba(255,77,109,0.35);color:var(--danger);}
.res-badge.normal{background:rgba(0,229,160,0.15);border:1px solid rgba(0,229,160,0.35);color:var(--safe);}
.res-hl{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;margin-bottom:9px;}
.res-hl.pneumonia{color:var(--danger);}
.res-hl.normal{color:var(--safe);}
.res-desc{color:var(--muted2);font-size:0.87rem;line-height:1.65;margin-bottom:24px;}
.conf-label{display:flex;justify-content:space-between;font-size:0.73rem;text-transform:uppercase;letter-spacing:0.09em;color:var(--muted);margin-bottom:7px;}
.conf-track{background:rgba(255,255,255,0.06);border-radius:100px;height:9px;overflow:hidden;margin-bottom:22px;}
.conf-fill{height:100%;border-radius:100px;transition:width 1.2s cubic-bezier(0.34,1.56,0.64,1);width:0%;}
.conf-fill.pneumonia{background:linear-gradient(90deg,#ff4d6d,#ff8fa3);}
.conf-fill.normal{background:linear-gradient(90deg,#00e5a0,#34d399);}
.res-metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:18px;}
.rm{background:rgba(56,139,255,0.06);border:1px solid var(--border);border-radius:10px;padding:13px;}
.rm-v{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;color:var(--accent);}
.rm-l{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;}
.disclaimer{display:flex;gap:9px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:11px 13px;font-size:0.77rem;color:var(--muted2);line-height:1.55;}
.disclaimer svg{flex-shrink:0;width:14px;height:14px;margin-top:1px;stroke:var(--warn);}
#dataset{background:var(--surface);}
.ds-grid{display:grid;grid-template-columns:1fr 1fr;gap:28px;}
.stat-cards{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px;}
.sc{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px;text-align:center;position:relative;overflow:hidden;}
.sc::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;}
.sc:nth-child(1)::after{background:var(--accent);}
.sc:nth-child(2)::after{background:var(--safe);}
.sc:nth-child(3)::after{background:var(--warn);}
.sc:nth-child(4)::after{background:#a78bfa;}
.sc-num{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;margin-bottom:4px;}
.sc-label{font-size:0.76rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;}
.info-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:26px;}
.info-card h4{font-family:'Syne',sans-serif;font-size:0.97rem;font-weight:700;margin-bottom:16px;}
.ir{display:flex;gap:10px;margin-bottom:11px;}
.ik{font-size:0.78rem;font-weight:600;color:var(--accent);min-width:76px;padding-top:1px;}
.iv{font-size:0.8rem;color:var(--muted2);line-height:1.55;}
.pipe-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:26px;}
.pipe-card h4{font-family:'Syne',sans-serif;font-size:0.97rem;font-weight:700;margin-bottom:18px;}
.ps{display:flex;gap:13px;margin-bottom:13px;align-items:flex-start;}
.ps-n{width:27px;height:27px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.73rem;font-weight:700;font-family:'Syne',sans-serif;flex-shrink:0;}
.ps-t{font-size:0.83rem;color:var(--muted2);line-height:1.55;padding-top:4px;}
.models-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;}
.mc{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:26px;position:relative;overflow:hidden;transition:transform 0.3s,box-shadow 0.3s;}
.mc:hover{transform:translateY(-4px);box-shadow:0 14px 36px rgba(0,0,0,0.3);}
.mc.best{border-color:rgba(0,229,160,0.4);}
.mc.best::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--safe),transparent);}
.best-badge{position:absolute;top:14px;right:14px;background:rgba(0,229,160,0.15);border:1px solid rgba(0,229,160,0.3);color:var(--safe);font-size:0.68rem;font-weight:700;padding:4px 10px;border-radius:100px;font-family:'Syne',sans-serif;}
.mc-name{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;margin-bottom:5px;}
.mc-arch{font-size:0.78rem;color:var(--accent);font-weight:500;margin-bottom:9px;}
.mc-desc{font-size:0.82rem;color:var(--muted2);line-height:1.6;margin-bottom:18px;}
.acc-row{display:flex;align-items:center;gap:9px;}
.acc-lbl{font-size:0.73rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.07em;width:76px;flex-shrink:0;}
.acc-bar{flex:1;background:rgba(255,255,255,0.06);border-radius:100px;height:7px;overflow:hidden;}
.acc-fill{height:100%;border-radius:100px;}
.f-cnn{background:linear-gradient(90deg,#388bff,#60a5fa);width:88%;}
.f-xce{background:linear-gradient(90deg,#a78bfa,#c4b5fd);width:82%;}
.f-vgg{background:linear-gradient(90deg,#00e5a0,#34d399);width:92%;}
.f-res{background:linear-gradient(90deg,#f59e0b,#fbbf24);width:85%;}
.acc-pct{font-family:'Syne',sans-serif;font-size:0.88rem;font-weight:700;width:36px;text-align:right;}
#results{background:var(--surface);}
.perf-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:36px;}
.pc{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px;text-align:center;}
.pc-val{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;margin-bottom:4px;}
.pc-lbl{font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;}
.table-wrap{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;}
table{width:100%;border-collapse:collapse;}
thead{background:rgba(56,139,255,0.1);}
th{padding:13px 18px;text-align:left;font-family:'Syne',sans-serif;font-size:0.77rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted2);}
td{padding:13px 18px;font-size:0.86rem;border-top:1px solid var(--border);color:var(--muted2);}
tr.hl td{color:var(--safe);font-weight:600;}
tr:hover td{background:rgba(56,139,255,0.04);}
td:first-child{font-weight:600;color:var(--text);}
.stack-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;}
.sk{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px;text-align:center;transition:transform 0.3s,border-color 0.3s;}
.sk:hover{transform:translateY(-4px);border-color:var(--accent);}
.sk-icon{font-size:1.9rem;margin-bottom:11px;}
.sk-name{font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;margin-bottom:5px;}
.sk-role{font-size:0.76rem;color:var(--muted);}
#conclusion{background:var(--surface);}
.conc-grid{display:grid;grid-template-columns:1fr 1fr;gap:28px;}
.conc-title{font-family:'Syne',sans-serif;font-weight:700;margin-bottom:18px;font-size:1.05rem;}
.ach-list,.fut-list{list-style:none;display:flex;flex-direction:column;gap:10px;}
.ach-list li,.fut-list li{display:flex;gap:11px;background:var(--card);border:1px solid var(--border);border-radius:10px;padding:13px 15px;font-size:0.86rem;color:var(--muted2);line-height:1.55;}
.ach-list li::before{content:'✓';color:var(--safe);font-weight:700;flex-shrink:0;margin-top:1px;}
.fut-list li::before{content:'→';color:var(--accent);font-weight:700;flex-shrink:0;margin-top:1px;}
footer{background:var(--card);border-top:1px solid var(--border);padding:44px 40px;}
.foot-inner{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:1fr auto;gap:40px;align-items:center;}
.foot-brand{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;}
.foot-brand span{color:var(--accent);}
.foot-tag{color:var(--muted);font-size:0.83rem;margin-top:5px;}
.foot-right{text-align:right;}
.foot-right p{font-size:0.81rem;color:var(--muted);line-height:1.9;}
.foot-right strong{color:var(--accent2);}
@media(max-width:900px){
  .hero-grid,.az-grid,.ds-grid,.conc-grid{grid-template-columns:1fr;}
  .steps{grid-template-columns:1fr 1fr;}
  .models-grid,.stack-grid{grid-template-columns:1fr 1fr;}
  .perf-grid{grid-template-columns:1fr 1fr;}
  .nav-links{display:none;}
}
@media(max-width:600px){
  .wrap{padding:80px 20px;}
  .hero-grid{padding:100px 20px 60px;}
  .stat-cards,.models-grid,.stack-grid,.perf-grid{grid-template-columns:1fr;}
}
</style>
</head>
<body>
<nav>
  <div class="nav-inner">
    <div class="nav-logo">Pneumo<span>Scan</span></div>
    <ul class="nav-links">
      <li><a href="#analyzer">Analyzer</a></li>
      <li><a href="#how">How It Works</a></li>
      <li><a href="#dataset">Dataset</a></li>
      <li><a href="#models">Models</a></li>
      <li><a href="#results">Results</a></li>
      <li><a href="#stack">Tech Stack</a></li>
      <li><a href="#conclusion">Conclusion</a></li>
    </ul>
    <a href="#analyzer" class="nav-cta">Try Now &rarr;</a>
  </div>
</nav>
<section id="hero">
  <div class="hero-grid">
    <div>
      <div class="hero-tag"><span class="dot"></span>VGG19 Deep Learning &middot; 92% Accuracy</div>
      <h1>AI-Powered<br/><span>Pneumonia</span><br/>Detection</h1>
      <p class="hero-p">Upload a chest X-ray and get an instant AI diagnosis powered by VGG19 — trained on 5,863 pediatric radiographs from Guangzhou Women &amp; Children's Medical Center. Built by Aarthi A R, Anna Adarsh College for Women.</p>
      <div class="hero-stats">
        <div class="hs"><div class="hs-num">92%</div><div class="hs-label">Test Accuracy</div></div>
        <div class="hs"><div class="hs-num">5,863</div><div class="hs-label">X-Ray Images</div></div>
        <div class="hs"><div class="hs-num">0.92</div><div class="hs-label">F1-Score</div></div>
        <div class="hs"><div class="hs-num">4</div><div class="hs-label">CNN Models</div></div>
      </div>
      <div class="hero-btns">
        <a href="#analyzer" class="btn-p">Analyze X-Ray &rarr;</a>
        <a href="#models" class="btn-o">View Models</a>
      </div>
    </div>
    <div class="hero-card">
      <div class="xray-box">
        <svg viewBox="0 0 100 120" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="5" y="5" width="90" height="110" rx="4" stroke="white" stroke-width="3"/>
          <ellipse cx="30" cy="62" rx="18" ry="30" stroke="white" stroke-width="2"/>
          <ellipse cx="70" cy="62" rx="18" ry="30" stroke="white" stroke-width="2"/>
          <line x1="50" y1="20" x2="50" y2="105" stroke="white" stroke-width="1.5"/>
          <line x1="18" y1="38" x2="82" y2="38" stroke="white" stroke-width="1.5"/>
        </svg>
        <p>Chest X-Ray Analyzer</p>
      </div>
      <div class="hc-stats">
        <div class="hcs"><div class="hcs-n">0.98</div><div class="hcs-l">Pneumonia Precision</div></div>
        <div class="hcs"><div class="hcs-n">0.97</div><div class="hcs-l">Normal Recall</div></div>
        <div class="hcs"><div class="hcs-n">224&sup2;</div><div class="hcs-l">Input Size px</div></div>
      </div>
    </div>
  </div>
</section>
<div class="divider"></div>
<section id="how">
  <div class="wrap">
    <div class="tag"><span class="dot"></span>Pipeline</div>
    <h2 class="stitle">How It <span>Works</span></h2>
    <p class="sdesc">From raw chest X-ray to AI diagnosis in seconds — a 5-stage deep learning pipeline.</p>
    <div class="steps">
      <div class="step"><div class="step-n">01</div><div class="step-t">Upload X-Ray</div><div class="step-d">Drag &amp; drop or browse a JPEG/PNG chest radiograph.</div></div>
      <div class="step"><div class="step-n">02</div><div class="step-t">Preprocess</div><div class="step-d">Resized to 224&times;224 px and pixel-normalized to 0-1.</div></div>
      <div class="step"><div class="step-n">03</div><div class="step-t">VGG19 Inference</div><div class="step-d">16 convolutional layers extract deep radiographic features.</div></div>
      <div class="step"><div class="step-n">04</div><div class="step-t">Classification</div><div class="step-d">FC layers output Normal vs Pneumonia probabilities.</div></div>
      <div class="step"><div class="step-n">05</div><div class="step-t">Result</div><div class="step-d">Diagnosis &amp; confidence score displayed instantly.</div></div>
    </div>
  </div>
</section>
<div class="divider"></div>
<section id="analyzer">
  <div class="wrap">
    <div class="tag"><span class="dot"></span>Live Demo</div>
    <h2 class="stitle">Chest X-Ray <span>Analyzer</span></h2>
    <p class="sdesc">Upload a chest X-ray image to get an instant AI-powered pneumonia prediction using our VGG19 model.</p>
    <div class="az-grid">
      <div class="upload-card">
        <div class="drop-zone" id="dropZone">
          <input type="file" id="fileInput" accept="image/*"/>
          <div class="up-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>
          <p class="drop-label">Drop your X-ray here</p>
          <p class="drop-sub">or click to browse &middot; JPG, PNG, JPEG</p>
        </div>
        <div id="preview-container"><img id="preview-img" src="" alt="X-ray preview"/></div>
        <button class="btn-analyze" id="analyzeBtn" disabled onclick="analyzeImage()">
          <svg id="btnIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:17px;height:17px;">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <div class="spinner" id="spinner"></div>
          <span id="btnText">Analyze X-Ray</span>
        </button>
      </div>
      <div class="result-panel">
        <div class="result-empty" id="resultEmpty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
            <circle cx="12" cy="12" r="10"/>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <p>Upload a chest X-ray and click<br/><strong>Analyze X-Ray</strong> to see the AI result here.</p>
        </div>
        <div id="result-content">
          <div class="res-badge" id="resBadge"></div>
          <div class="res-hl" id="resHl"></div>
          <div class="res-desc" id="resDesc"></div>
          <div class="conf-label"><span>Confidence Score</span><span id="confPct">0%</span></div>
          <div class="conf-track"><div class="conf-fill" id="confFill"></div></div>
          <div class="res-metrics">
            <div class="rm"><div class="rm-v" id="rmClass">-</div><div class="rm-l">Predicted Class</div></div>
            <div class="rm"><div class="rm-v" id="rmConf">-</div><div class="rm-l">Confidence</div></div>
          </div>
          <div class="disclaimer">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            For research &amp; educational use only. Always consult a qualified medical professional for diagnosis and treatment.
          </div>
        </div>
      </div>
    </div>
  </div>
</section>
<div class="divider"></div>
<section id="dataset">
  <div class="wrap">
    <div class="tag"><span class="dot"></span>Training Data</div>
    <h2 class="stitle">Dataset &amp; <span>Preprocessing</span></h2>
    <p class="sdesc">Sourced from Kaggle's Chest X-Ray Images dataset — real-world pediatric radiographs, expert-graded and carefully preprocessed.</p>
    <div class="ds-grid">
      <div>
        <div class="stat-cards">
          <div class="sc"><div class="sc-num">5,863</div><div class="sc-label">Total Images</div></div>
          <div class="sc"><div class="sc-num">5,216</div><div class="sc-label">Training Set</div></div>
          <div class="sc"><div class="sc-num">624</div><div class="sc-label">Test Set</div></div>
          <div class="sc"><div class="sc-num">2</div><div class="sc-label">Classes</div></div>
        </div>
        <div class="info-card">
          <h4>Dataset Details</h4>
          <div class="ir"><span class="ik">Source</span><span class="iv">Kaggle - Chest X-Ray Images (Pneumonia)</span></div>
          <div class="ir"><span class="ik">Patients</span><span class="iv">Pediatric patients aged 1-5 years</span></div>
          <div class="ir"><span class="ik">Origin</span><span class="iv">Guangzhou Women &amp; Children's Medical Center</span></div>
          <div class="ir"><span class="ik">Format</span><span class="iv">JPEG greyscale chest radiographs</span></div>
          <div class="ir"><span class="ik">Labels</span><span class="iv">NORMAL vs PNEUMONIA (binary classification)</span></div>
          <div class="ir"><span class="ik">Quality</span><span class="iv">Expert physician grading; poor-quality scans removed</span></div>
        </div>
      </div>
      <div class="pipe-card">
        <h4>Preprocessing Pipeline</h4>
        <div class="ps"><div class="ps-n">1</div><div class="ps-t">Resize all images to <strong>224x224 pixels</strong> for model input uniformity.</div></div>
        <div class="ps"><div class="ps-n">2</div><div class="ps-t"><strong>Pixel normalization</strong> - scale values to 0-1 range for stable training.</div></div>
        <div class="ps"><div class="ps-n">3</div><div class="ps-t"><strong>Oversampling</strong> to balance Normal &amp; Pneumonia classes to 50/50 split.</div></div>
        <div class="ps"><div class="ps-n">4</div><div class="ps-t"><strong>Data Augmentation</strong> - rotations, flips, zoom, shifts to improve generalization.</div></div>
        <div class="ps"><div class="ps-n">5</div><div class="ps-t"><strong>Train / Validation / Test</strong> split preserved - 5,216 / 8 / 624 images.</div></div>
      </div>
    </div>
  </div>
</section>
<div class="divider"></div>
<section id="models">
  <div class="wrap">
    <div class="tag"><span class="dot"></span>Architecture Comparison</div>
    <h2 class="stitle">CNN <span>Models</span></h2>
    <p class="sdesc">Four deep learning architectures trained and evaluated. VGG19 achieved the best test accuracy of 92% and was selected for deployment.</p>
    <div class="models-grid">
      <div class="mc"><div class="mc-name">Basic CNN</div><div class="mc-arch">Custom 4-Convolutional Layer Architecture</div><div class="mc-desc">Conv - ReLU - MaxPool - Dropout - FC - Sigmoid. Baseline architecture built from scratch as a performance benchmark.</div><div class="acc-row"><span class="acc-lbl">Test Accuracy</span><div class="acc-bar"><div class="acc-fill f-cnn"></div></div><span class="acc-pct" style="color:#388bff">88%</span></div></div>
      <div class="mc"><div class="mc-name">Xception</div><div class="mc-arch">Entry / Middle / Exit Flow Architecture</div><div class="mc-desc">Depthwise separable convolutions for efficient feature extraction with fewer parameters. Transfer learning from ImageNet weights.</div><div class="acc-row"><span class="acc-lbl">Test Accuracy</span><div class="acc-bar"><div class="acc-fill f-xce"></div></div><span class="acc-pct" style="color:#a78bfa">82%</span></div></div>
      <div class="mc best"><div class="best-badge">&#9733; Best Model</div><div class="mc-name">VGG19</div><div class="mc-arch">16 Convolutional + 3 Fully Connected Layers</div><div class="mc-desc">Small 3x3 filters with uniform deep architecture. Transfer learning + fine-tuned top layers. Best accuracy: 92%, F1-Score: 0.92.</div><div class="acc-row"><span class="acc-lbl">Test Accuracy</span><div class="acc-bar"><div class="acc-fill f-vgg"></div></div><span class="acc-pct" style="color:#00e5a0">92%</span></div></div>
      <div class="mc"><div class="mc-name">ResNet-50</div><div class="mc-arch">50-Layer Residual Network</div><div class="mc-desc">Skip connections solve the vanishing gradient problem. Residual learning enables deep feature extraction from complex X-ray patterns.</div><div class="acc-row"><span class="acc-lbl">Test Accuracy</span><div class="acc-bar"><div class="acc-fill f-res"></div></div><span class="acc-pct" style="color:#f59e0b">85%</span></div></div>
    </div>
  </div>
</section>
<div class="divider"></div>
<section id="results">
  <div class="wrap">
    <div class="tag"><span class="dot"></span>Performance Metrics</div>
    <h2 class="stitle">Results &amp; <span>Metrics</span></h2>
    <p class="sdesc">VGG19 outperformed all architectures across every metric on the 624-image held-out test set.</p>
    <div class="perf-grid">
      <div class="pc"><div class="pc-val" style="color:var(--safe)">92%</div><div class="pc-lbl">Test Accuracy</div></div>
      <div class="pc"><div class="pc-val" style="color:var(--accent)">0.92</div><div class="pc-lbl">F1-Score</div></div>
      <div class="pc"><div class="pc-val" style="color:#a78bfa">0.98</div><div class="pc-lbl">Pneumonia Precision</div></div>
      <div class="pc"><div class="pc-val" style="color:var(--warn)">0.97</div><div class="pc-lbl">Normal Recall</div></div>
      <div class="pc"><div class="pc-val" style="color:var(--safe)">347</div><div class="pc-lbl">Pneumonia Correct</div></div>
      <div class="pc"><div class="pc-val" style="color:var(--accent)">227</div><div class="pc-lbl">Normal Correct</div></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Train Acc.</th><th>Val Acc.</th><th>Test Acc.</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>
        <tbody>
          <tr><td>Basic CNN</td><td>91%</td><td>92%</td><td>88%</td><td>0.88</td><td>0.88</td><td>0.88</td></tr>
          <tr><td>Xception</td><td>91%</td><td>92%</td><td>82%</td><td>0.85</td><td>0.82</td><td>0.83</td></tr>
          <tr class="hl"><td>VGG19 &#9733;</td><td>94%</td><td>94%</td><td>92%</td><td>0.93</td><td>0.92</td><td>0.92</td></tr>
          <tr><td>ResNet-50</td><td>87%</td><td>91%</td><td>85%</td><td>0.87</td><td>0.85</td><td>0.85</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>
<div class="divider"></div>
<section id="stack">
  <div class="wrap">
    <div class="tag"><span class="dot"></span>Technology</div>
    <h2 class="stitle">Software <span>Stack</span></h2>
    <p class="sdesc">Built with production-grade open-source tools for deep learning, image processing, and web deployment.</p>
    <div class="stack-grid">
      <div class="sk"><div class="sk-icon">🐍</div><div class="sk-name">Python 3.x</div><div class="sk-role">Core Language</div></div>
      <div class="sk"><div class="sk-icon">🧠</div><div class="sk-name">TensorFlow / Keras</div><div class="sk-role">Deep Learning</div></div>
      <div class="sk"><div class="sk-icon">🖼️</div><div class="sk-name">OpenCV + Pillow</div><div class="sk-role">Image Processing</div></div>
      <div class="sk"><div class="sk-icon">🔢</div><div class="sk-name">NumPy / Pandas</div><div class="sk-role">Data Analysis</div></div>
      <div class="sk"><div class="sk-icon">📊</div><div class="sk-name">Matplotlib / Seaborn</div><div class="sk-role">Visualization</div></div>
      <div class="sk"><div class="sk-icon">🔬</div><div class="sk-name">Scikit-learn</div><div class="sk-role">Evaluation Metrics</div></div>
      <div class="sk"><div class="sk-icon">🌐</div><div class="sk-name">Flask</div><div class="sk-role">Web Application</div></div>
      <div class="sk"><div class="sk-icon">📓</div><div class="sk-name">Jupyter / Anaconda</div><div class="sk-role">Dev Environment</div></div>
    </div>
  </div>
</section>
<div class="divider"></div>
<section id="conclusion">
  <div class="wrap">
    <div class="tag"><span class="dot"></span>Conclusion</div>
    <h2 class="stitle">Key Findings &amp; <span>Future Work</span></h2>
    <p class="sdesc">Deep Learning x Medical Imaging = Faster, Accessible Pneumonia Diagnosis.</p>
    <div class="conc-grid">
      <div>
        <div class="conc-title" style="color:var(--safe)">✓ Key Achievements</div>
        <ul class="ach-list">
          <li>Dataset balanced using oversampling - achieving 50/50 class split.</li>
          <li>4 CNN architectures trained and evaluated on 5,863 chest X-rays.</li>
          <li>VGG19 achieved best test accuracy of 92% with F1-Score 0.92.</li>
          <li>Transfer learning significantly improved model robustness.</li>
          <li>Real-time Flask web app deployed for instant pneumonia detection.</li>
          <li>Prediction probabilities provide transparent, interpretable diagnostics.</li>
        </ul>
      </div>
      <div>
        <div class="conc-title" style="color:var(--accent)">→ Future Directions</div>
        <ul class="fut-list">
          <li>Distinguish Bacterial vs Viral pneumonia subtypes for finer diagnosis.</li>
          <li>Explore InceptionV3, MobileNet, and ShuffleNet architectures.</li>
          <li>Hyperparameter optimization to push accuracy beyond 92%.</li>
          <li>Real-time optimization for clinical deployment at scale.</li>
          <li>Cloud deployment for broader hospital accessibility.</li>
          <li>Integrate with EHR systems as a clinical decision support tool.</li>
        </ul>
      </div>
    </div>
  </div>
</section>
<div class="divider"></div>
<footer>
  <div class="foot-inner">
    <div>
      <div class="foot-brand">Pneumo<span>Scan</span> AI</div>
      <div class="foot-tag">Approach for Pneumonia Detection via Image Classification</div>
    </div>
    <div class="foot-right">
      <p><strong>Aarthi A R</strong></p>
      <p>Anna Adarsh College for Women (Autonomous)</p>
      <p>Guide: <strong>Dr. Hannah Vijaykumar</strong></p>
      <p style="margin-top:6px;color:var(--muted);font-size:0.76rem;">Review II · February 2026 · Design &amp; Implementation Stage</p>
    </div>
  </div>
</footer>
<script>
const fileInput = document.getElementById('fileInput');
const dropZone  = document.getElementById('dropZone');
const analyzeBtn = document.getElementById('analyzeBtn');
let selectedFile = null;
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('preview-img').src = ev.target.result;
    document.getElementById('preview-container').style.display = 'block';
    analyzeBtn.disabled = false;
    document.getElementById('result-content').style.display = 'none';
    document.getElementById('resultEmpty').style.display = 'flex';
  };
  reader.readAsDataURL(file);
}
async function analyzeImage() {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  document.getElementById('btnIcon').style.display = 'none';
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('btnText').textContent = 'Analyzing...';
  const formData = new FormData();
  formData.append('file', selectedFile);
  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    showResult(data.prediction, data.confidence);
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
    document.getElementById('btnIcon').style.display = '';
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('btnText').textContent = 'Analyze X-Ray';
  }
}
function showResult(prediction, confidence) {
  const isPneumonia = prediction === 'PNEUMONIA';
  const cls = isPneumonia ? 'pneumonia' : 'normal';
  const pct = Math.round(confidence * 100);
  document.getElementById('resultEmpty').style.display = 'none';
  document.getElementById('result-content').style.display = 'block';
  const badge = document.getElementById('resBadge');
  badge.textContent = isPneumonia ? '⚠ Pneumonia Detected' : '✓ Normal — No Pneumonia';
  badge.className = 'res-badge ' + cls;
  const hl = document.getElementById('resHl');
  hl.textContent = isPneumonia ? 'Pneumonia Detected' : 'No Pneumonia Found';
  hl.className = 'res-hl ' + cls;
  document.getElementById('resDesc').textContent = isPneumonia
    ? 'The VGG19 model identified patterns consistent with pneumonia. Please seek immediate medical evaluation.'
    : 'The VGG19 model found no significant indicators of pneumonia. Lung patterns appear normal. Always confirm with a doctor.';
  document.getElementById('confPct').textContent = pct + '%';
  const fill = document.getElementById('confFill');
  fill.className = 'conf-fill ' + cls;
  document.getElementById('rmClass').textContent = prediction;
  document.getElementById('rmConf').textContent = pct + '%';
  requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = pct + '%'; }));
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    try:
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)
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
        return jsonify({"prediction": prediction, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Load model on startup (works for both gunicorn and direct python)
load_model()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)