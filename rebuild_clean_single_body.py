# -*- coding: utf-8 -*-
"""
Perfect Single-Body Clean HTML Builder with Modern Tailwind Design System
"""

with open("/tmp/original_full.html", "r", encoding="utf-8") as f:
    orig = f.read()

# 1. Clean Modern Head with Tailwind & Inter font
head_content = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISBD Studio — Next-Gen AI Workspace</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚀</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<style>
:root {
  --bg-app: #f8fafc;
  --text-main: #0f172a;
  --text-sub: #475569;
  --text-muted: #64748b;
  --primary: #4f46e5;
  --cyan: #0284c7;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --radius-xl: 16px;
  --radius-lg: 12px;
  --radius-md: 8px;
  --radius-sm: 6px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', 'Noto Sans Bengali', system-ui, sans-serif;
  background-color: var(--bg-app);
  color: var(--text-main);
  min-height: 100vh;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

.workspace-layout {
  display: flex;
  min-height: 100vh;
}

.workspace-sidebar {
  width: 290px;
  background: #ffffff;
  border-right: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  position: sticky;
  top: 0;
  height: 100vh;
  z-index: 50;
  box-shadow: 1px 0 3px 0 rgba(0, 0, 0, 0.02);
}

.ws-brand {
  padding: 24px 20px 20px;
  border-bottom: 1px solid #f1f5f9;
}

.ws-nav {
  padding: 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  overflow-y: auto;
}

.ws-category {
  font-size: 11px;
  font-weight: 700;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 14px 12px 6px;
}

.ws-btn {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-radius: 10px;
  color: #475569;
  font-size: 13.5px;
  font-weight: 600;
  cursor: pointer;
  background: transparent;
  border: 1px solid transparent;
  transition: all 0.15s ease;
  width: 100%;
  text-align: left;
}

.ws-btn:hover {
  background: #f1f5f9;
  color: #0f172a;
}

.ws-btn.active {
  background: #eef2ff;
  border-color: #c7d2fe;
  color: #4338ca;
  font-weight: 700;
  box-shadow: 0 1px 2px 0 rgba(79, 70, 229, 0.05);
}

.ws-btn .icon {
  font-size: 18px;
  width: 24px;
  text-align: center;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.ws-footer {
  padding: 16px 20px;
  border-top: 1px solid #f1f5f9;
  background: #ffffff;
}

.workspace-content {
  flex: 1;
  padding: 28px 40px 80px;
  max-width: 1320px;
  overflow-y: auto;
}

.tab-pane {
  display: none;
  animation: paneFadeIn 0.2s ease forwards;
}

.tab-pane.active {
  display: block;
}

@keyframes paneFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ─ Modern Cards & Banners ─ */
.panel-box, .cv-lab-banner {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: var(--radius-xl);
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 18px;
}

.panel-header h2 {
  font-size: 16px;
  font-weight: 700;
  color: #0f172a;
  display: flex;
  align-items: center;
  gap: 10px;
}

.step-badge {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 700;
  background: #eef2ff;
  color: #4f46e5;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #c7d2fe;
}

/* ─ Modern Stats Grid ─ */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: var(--radius-lg);
  padding: 18px 20px;
  box-shadow: 0 1px 3px 0 rgba(0,0,0,0.03);
  transition: all 0.2s;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px -2px rgba(0, 0, 0, 0.06);
}

.stat-title {
  font-size: 11.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
}

.stat-val {
  font-size: 24px;
  font-weight: 800;
  color: #0f172a;
  font-family: 'JetBrains Mono', monospace;
  margin: 6px 0 2px;
}

.stat-footer {
  font-size: 12px;
  color: #64748b;
}

/* ─ Inputs & Buttons ─ */
.param-input, .param-select {
  width: 100%;
  background: #ffffff;
  border: 1px solid #cbd5e1;
  color: #0f172a;
  font-weight: 500;
  padding: 9px 14px;
  border-radius: 8px;
  font-size: 13px;
  font-family: inherit;
  outline: none;
  transition: all 0.15s;
}

.param-input:focus, .param-select:focus {
  border-color: #4f46e5;
  box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
}

.btn {
  padding: 9px 18px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: all 0.15s ease;
  border: 1px solid transparent;
  font-family: inherit;
}

.btn-main {
  background: #4f46e5;
  color: #ffffff;
  box-shadow: 0 1px 2px 0 rgba(79, 70, 229, 0.2);
}

.btn-main:hover {
  background: #4338ca;
  box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
  transform: translateY(-1px);
}

.btn-outline {
  background: #ffffff;
  border: 1px solid #cbd5e1;
  color: #334155;
}

.btn-outline:hover {
  background: #f8fafc;
  color: #0f172a;
  border-color: #94a3b8;
}

.btn-danger-light {
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #dc2626;
}

.btn-danger-light:hover {
  background: #fee2e2;
}

/* ─ Chip Labels ─ */
.chip-label {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 600;
  color: #334155;
  cursor: pointer;
  user-select: none;
  transition: all 0.15s;
}

.chip-label:hover {
  background: #f1f5f9;
  border-color: #cbd5e1;
}

.chip-label:has(input:checked) {
  background: #eef2ff;
  border-color: #818cf8;
  color: #3730a3;
  font-weight: 700;
  box-shadow: 0 1px 2px 0 rgba(79, 70, 229, 0.08);
}

.chip-label input[type="checkbox"] {
  accent-color: #4f46e5;
  width: 15px;
  height: 15px;
}

/* ─ Grids & Layouts ─ */
.det-grid { display: grid; grid-template-columns: 1fr 1.2fr; gap: 24px; align-items: start; }
@media (max-width: 860px) { .det-grid { grid-template-columns: 1fr; } }
.det-controls { display: flex; flex-direction: column; gap: 14px; }
.det-result-img {
  width: 100%; max-height: 380px; object-fit: contain; border-radius: var(--radius-md);
  border: 1px solid #e2e8f0; background: #000; display: none;
}
.det-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.det-tag {
  background: #f1f5f9; border: 1px solid #e2e8f0;
  padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
  display: inline-flex; align-items: center; gap: 6px; color: #1e293b;
}
.det-tag .count { background: #0284c7; color: #fff; border-radius: 10px; padding: 2px 6px; font-size: 10px; }

.cv-palette { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.palette-chip {
  padding: 6px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;
  display: flex; align-items: center; gap: 6px; border: 1px solid rgba(0,0,0,0.1);
}

.compare-container {
  position: relative; width: 100%; height: 260px; border-radius: var(--radius-md);
  overflow: hidden; background: #0f172a; border: 1px solid #e2e8f0; margin: 14px 0;
  display: flex; align-items: center; justify-content: center;
}
.compare-img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
.compare-overlay {
  position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden;
  border-right: 2px solid #fff; box-shadow: 2px 0 10px rgba(0,0,0,0.5);
}
.compare-overlay img { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; }
.slider-handle {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  width: 28px; height: 28px; border-radius: 50%; background: #fff; color: #000;
  display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 800;
  pointer-events: none; box-shadow: 0 0 15px rgba(0,0,0,0.8);
}
.compare-range {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: ew-resize; z-index: 10;
}
.play-placeholder { text-align: center; color: #94a3b8; font-size: 13px; }

.terminal-area {
  background: #0f172a; border: 1px solid #334155; border-radius: var(--radius-md);
  padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #38bdf8;
  max-height: 260px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6;
}

.toasts { position: fixed; top: 20px; right: 20px; z-index: 9999; display: flex; flex-direction: column; gap: 8px; }
.toast {
  padding: 10px 18px; border-radius: var(--radius-sm); font-size: 12.5px; font-weight: 600;
  backdrop-filter: blur(16px); border: 1px solid #e2e8f0;
  box-shadow: 0 10px 30px rgba(0,0,0,0.1); animation: toastIn 0.3s ease;
}
.toast.ok { background: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
.toast.warn { background: #fffbeb; border-color: #fde68a; color: #92400e; }
.toast.err { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
@keyframes toastIn { from { transform: translateX(50px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

.pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981; display: inline-block; }

.pairs-container { display: flex; flex-direction: column; gap: 10px; margin-top: 14px; }
.pair-item {
  background: #f8fafc; border: 1px solid #e2e8f0;
  border-radius: var(--radius-md); padding: 12px; display: grid;
  grid-template-columns: 1fr 1fr auto; gap: 12px; align-items: center;
}
.pair-side label {
  font-size: 11px; font-weight: 600; color: #64748b;
  display: block; margin-bottom: 4px;
}
.pair-side input[type=file] { width: 100%; font-size: 11px; color: #334155; }
.thumb-preview { width: 44px; height: 44px; border-radius: 6px; object-fit: cover; margin-top: 4px; display: none; border: 1px solid #cbd5e1; }
.btn-del-pair {
  background: #ffffff; border: 1px solid #e2e8f0; color: #dc2626;
  border-radius: 8px; width: 32px; height: 32px; cursor: pointer; display: flex;
  align-items: center; justify-content: center; font-size: 13px; transition: all 0.2s;
}
.btn-del-pair:hover { background: #fee2e2; border-color: #fca5a5; }

.dropzone {
  border: 2px dashed #cbd5e1; border-radius: var(--radius-md);
  padding: 26px 16px; text-align: center; cursor: pointer; transition: all 0.2s;
  background: #f8fafc;
}
.dropzone:hover, .dropzone.dragover {
  border-color: #6366f1; background: #eef2ff;
}
.dz-icon { font-size: 32px; margin-bottom: 6px; }
.dz-text { font-size: 13.5px; font-weight: 600; color: #1e293b; }
.dz-sub { font-size: 11.5px; color: #64748b; margin-top: 2px; }

.param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
.param-field label { font-size: 11px; font-weight: 700; color: #475569; display: block; margin-bottom: 5px; }
</style>
</head>
"""

# Extract All Sections cleanly from orig
stats_html = orig[orig.find("<!-- ─ Stats Row ─ -->"):orig.find("<!-- ══════════════════════════════════════════════════════════")]
school_html = orig[orig.find("<!-- ══════════════════════════════════════════════════════════"):orig.find("<!-- ─ MASTER SECTOR 2: AI Difference Heatmap Visualizer ─ -->")]
heatmap_html = orig[orig.find("<!-- ─ MASTER SECTOR 2: AI Difference Heatmap Visualizer ─ -->"):orig.find("<!-- ─ MASTER SECTOR: 11 Professional Studio Editing Services ─ -->")]
services_html = orig[orig.find("<!-- ─ MASTER SECTOR: 11 Professional Studio Editing Services ─ -->"):orig.find("<!-- ─ NEW FEATURE 1: Object Recognition & Visual Tracking Spotlight ─ -->")]
yolo_html = orig[orig.find("<!-- ─ NEW FEATURE 1: Object Recognition & Visual Tracking Spotlight ─ -->"):orig.find("<!-- ─ NEW FEATURE 2: OpenCV & Scikit-Image Multi-Vision Toolkit ─ -->")]
cv_html = orig[orig.find("<!-- ─ NEW FEATURE 2: OpenCV & Scikit-Image Multi-Vision Toolkit ─ -->"):orig.find("<!-- ─ NEW FEATURE 3: Magic Eraser & Inpainting Brush Studio ─ -->")]
magic_html = orig[orig.find("<!-- ─ NEW FEATURE 3: Magic Eraser & Inpainting Brush Studio ─ -->"):orig.find("<!-- ─ Main Section: Upload & Controls ─ -->")]

main_sec = orig[orig.find("<!-- ─ Main Section: Upload & Controls ─ -->"):orig.find("<script>")]
left_col_start = main_sec.find("<!-- Left Column: Fine-tune & Pairs -->")
right_col_start = main_sec.find("<!-- Right Column: Interactive Inference Playground")
finetune_html = main_sec[left_col_start:right_col_start]
restore_html = main_sec[right_col_start:]

scripts = orig[orig.find("<script>"):orig.rfind("</script>") + len("</script>")]

# Build Single Unified HTML Structure
full_html = head_content + """
<body>

<div class="toasts" id="toasts"></div>

<div class="workspace-layout">
  <!-- ═════════ SIDEBAR ═════════ -->
  <aside class="workspace-sidebar">
    <div class="ws-brand">
      <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:40px; height:40px; border-radius:10px; background:#4f46e5; display:flex; align-items:center; justify-content:center; font-size:20px; color:#fff; box-shadow:0 4px 12px rgba(79,70,229,0.25);">🚀</div>
        <div>
          <h1 style="font-size:18px; font-weight:800; color:#0f172a; line-height:1.2;">ISBD Studio</h1>
          <span style="font-size:11px; font-weight:600; color:#0284c7; text-transform:uppercase; letter-spacing:0.5px;">AI Workspace</span>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:8px; font-size:12px; padding:6px 12px; border-radius:20px; background:#f1f5f9; border:1px solid #e2e8f0; color:#475569; margin-top:14px; justify-content:center;">
        <span class="pulse-dot"></span>
        <span id="nav-status">Vision & Trainer Live</span>
      </div>
    </div>

    <nav class="ws-nav">
      <div class="ws-category">🧠 AI ট্রেনিং (Training)</div>
      <button class="ws-btn active" data-tab="tab-dashboard" onclick="switchTab('tab-dashboard')">
        <span class="icon">📊</span> লাইভ ড্যাশবোর্ড
      </button>
      <button class="ws-btn" data-tab="tab-school" onclick="switchTab('tab-school')">
        <span class="icon">🎓</span> Before/After এডিটিং স্কুল
      </button>
      <button class="ws-btn" data-tab="tab-finetune" onclick="switchTab('tab-finetune')">
        <span class="icon">⚡</span> পেয়ার ফাইন-টিউন ল্যাব
      </button>

      <div class="ws-category">🧪 AI টেস্টিং (Testing Lab)</div>
      <button class="ws-btn" data-tab="tab-restore" onclick="switchTab('tab-restore')">
        <span class="icon">✨</span> এআই রিস্টোরেশন ল্যাব
      </button>
      <button class="ws-btn" data-tab="tab-heatmap" onclick="switchTab('tab-heatmap')">
        <span class="icon">🔥</span> ডিফারেন্স হিটম্যাপ
      </button>
      <button class="ws-btn" data-tab="tab-services" onclick="switchTab('tab-services')">
        <span class="icon">💎</span> ১১টি কমার্শিয়াল সার্ভিস
      </button>
      <button class="ws-btn" data-tab="tab-magic" onclick="switchTab('tab-magic')">
        <span class="icon">🪄</span> ম্যাজিক ইরেজার (Inpaint)
      </button>
      <button class="ws-btn" data-tab="tab-yolo" onclick="switchTab('tab-yolo')">
        <span class="icon">👁️</span> অবজেক্ট ট্র্যাকিং (YOLO)
      </button>
      <button class="ws-btn" data-tab="tab-cv" onclick="switchTab('tab-cv')">
        <span class="icon">🎨</span> কালার প্যালেট ও ফিল্টার
      </button>
    </nav>

    <div class="ws-footer">
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size:12px;">
          <span>২৪/৭ সেলফ-লার্নিং:</span>
          <span id="side-train-status" style="color: #059669; font-weight:700;">Active</span>
        </div>
        <button id="btn-toggle-train" onclick="toggleContinuousTraining()" class="btn btn-outline" style="width: 100%; justify-content: center; font-size: 11.5px; padding: 6px; border-color: #fca5a5; color: #dc2626;">
          ⏸️ সেলফ-লার্নিং বন্ধ করুন
        </button>
      </div>
    </div>
  </aside>

  <!-- ═════════ CONTENT AREA ═════════ -->
  <main class="workspace-content">
    
    <!-- Tab 1: Dashboard -->
    <div id="tab-dashboard" class="tab-pane active">
      <div class="panel-header" style="margin-bottom: 18px;">
        <h2><span class="step-badge">📊</span> এআই ট্রেনিং স্ট্যাটাস ও লাইভ পারফরম্যান্স ওভারভিউ</h2>
        <div style="display:flex; align-items:center; gap:10px;">
          <button id="dash-btn-toggle" onclick="toggleContinuousTraining()" class="btn btn-outline" style="font-size: 11.5px; padding: 6px 14px; border-color: #fca5a5; color: #dc2626;">
            ⏸️ সেলফ-লার্নিং পজ করুন
          </button>
          <span style="font-size: 11px; color: #0284c7; font-weight:700;">Autonomous Learning</span>
        </div>
      </div>
""" + stats_html + """
      <div style="display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; margin-top: 24px;">
        <div class="panel-box">
          <div class="panel-header">
            <h2><span class="step-badge">📜</span> লাইভ ট্রেইনিং ইঞ্জিন লগ (Console Tail)</h2>
            <span id="badge-ft" style="font-size: 11px; padding: 2px 8px; border-radius: 12px; background: #f1f5f9; color:#475569;">Idle</span>
          </div>
          <div class="terminal-area" id="term-log">ইঞ্জিন প্রস্তুত — পেয়ার আপলোড বা ফাইন-টিউন ট্র্যাকিং…</div>
        </div>

        <div class="panel-box">
          <div class="panel-header">
            <h2><span class="step-badge">💡</span> ISBD স্টুডিও গাইড ও ফিচার শর্টকাট</h2>
          </div>
          <div style="font-size: 13px; color: #475569; line-height: 1.9;">
            <div style="margin-bottom: 8px;">🎓 <strong>এডিটিং স্কুল:</strong> ফটোশপের Before/After পেয়ার দিয়ে এআই-কে নতুন স্কিল শেখান।</div>
            <div style="margin-bottom: 8px;">🔥 <strong>ডিফারেন্স হিটম্যাপ:</strong> ছবির পরিবর্তিত এরিয়া থার্মাল কালারে পরীক্ষা করুন।</div>
            <div style="margin-bottom: 8px;">💎 <strong>১১টি কমার্শিয়াল সার্ভিস:</strong> এক ক্লিকে ক্লিপিং পাথ, রিটাচিং, শ্যাডো তৈরি।</div>
            <div style="margin-bottom: 8px;">🪄 <strong>ম্যাজিক ইরেজার:</strong> ব্রাশ দিয়ে দাগ, অনাকাঙ্ক্ষিত অবজেক্ট বা ওয়াটারমার্ক মুছুন।</div>
            <div>👁️ <strong>অবজেক্ট ট্র্যাকিং:</strong> YOLOv8 এআই দিয়ে ছবিতে থাকা সব উপাদান চিহ্নিত করুন।</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Tab 2: AI School -->
    <div id="tab-school" class="tab-pane">
""" + school_html + """
    </div>

    <!-- Tab 3: Finetune -->
    <div id="tab-finetune" class="tab-pane">
      <div class="panel-box" style="margin-bottom: 20px;">
        <div class="panel-header">
          <h2><span class="step-badge">⚡</span> পেয়ার ফাইন-টিউন ও ট্রেনিং প্যারামিটার কন্ট্রোল</h2>
          <span style="font-size: 11px; color: #0284c7; font-weight:700;">Manual Batch & Fine-tune Config</span>
        </div>
        <p style="font-size: 13px; color: #475569; margin-bottom: 16px;">
          এখানে ম্যানুয়াল পেয়ার আপলোড করে হাইপারপ্যারামিটার (Steps, Learning Rate, Batch Size) কনফিগার করে ডিপ ফাইন-টিউনিং শুরু করতে পারেন।
        </p>
      </div>
""" + finetune_html + """
      </div>
    </div>

    <!-- Tab 4: Restoration Playground (TESTING 1) -->
    <div id="tab-restore" class="tab-pane">
      <div class="panel-box" style="margin-bottom: 20px;">
        <div class="panel-header">
          <h2><span class="step-badge">✨</span> এআই রিস্টোরেশন ল্যাব ও ইন্টারঅ্যাক্টিভ স্লাইডার</h2>
          <span style="font-size: 11px; color: #4f46e5; font-weight:700;">Real-time Model Inference</span>
        </div>
        <p style="font-size: 13px; color: #475569; margin-bottom: 16px;">
          যেকোনো নষ্ট বা সাধারণ ছবি আপলোড করুন — এআই মডেল রিয়েল-টাইমে রিস্টোর করে বিফোর/আফটার স্লাইডারে তুলনা দেখাবে।
        </p>
      </div>
      <div style="display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px;">
""" + restore_html + """
      </div>
    </div>

    <!-- Tab 5: Heatmap (TESTING 2) -->
    <div id="tab-heatmap" class="tab-pane">
""" + heatmap_html + """
    </div>

    <!-- Tab 6: Commercial Services (TESTING 3) -->
    <div id="tab-services" class="tab-pane">
""" + services_html + """
    </div>

    <!-- Tab 7: Magic Eraser (TESTING 4) -->
    <div id="tab-magic" class="tab-pane">
""" + magic_html + """
    </div>

    <!-- Tab 8: YOLO (TESTING 5) -->
    <div id="tab-yolo" class="tab-pane">
""" + yolo_html + """
    </div>

    <!-- Tab 9: Computer Vision Lab (TESTING 6) -->
    <div id="tab-cv" class="tab-pane">
""" + cv_html + """
    </div>

  </main>
</div>

<script>
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.ws-btn').forEach(b => b.classList.remove('active'));
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
  
  const btn = document.querySelector('[data-tab="' + tabId + '"]');
  if (btn) btn.classList.add('active');
  
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
</script>

""" + scripts + """
</body>
</html>
"""

# Replace in index.html
with open("/home/imon/isbd_model/isbd/templates/index.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("Unified and Clean HTML written successfully! Size:", len(full_html))
