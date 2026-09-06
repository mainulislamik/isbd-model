# -*- coding: utf-8 -*-
"""
Perfect UI Restructurer:
Takes the 100% complete original HTML (which has all working DOM elements and JS logic)
and cleanly organizes each major feature block into modern Sidebar Tab Panes.
"""
import re

with open("/tmp/original_full.html", "r", encoding="utf-8") as f:
    orig = f.read()

# 1. Extract Head (Styles + Fonts)
head_end = orig.find("</head>")
head_html = orig[:head_end]

# 2. Extract Script Part
script_start = orig.find("<script>")
scripts = orig[script_start:]

# 3. Add refined Sidebar and Tab styles
enhanced_styles = """
<style>
/* ─ Professional Sidebar & Tabbed Workspace Layout ─ */
body {
  margin: 0;
  padding: 0;
  overflow-x: hidden;
}

.workspace-layout {
  display: flex;
  min-height: 100vh;
  background: var(--bg);
}

.workspace-sidebar {
  width: 280px;
  background: rgba(10, 14, 23, 0.98);
  border-right: 1px solid var(--card-border);
  backdrop-filter: blur(24px);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  position: sticky;
  top: 0;
  height: 100vh;
  z-index: 100;
}

.ws-brand {
  padding: 22px 18px 16px;
  border-bottom: 1px solid var(--card-border);
}

.ws-nav {
  padding: 14px 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  overflow-y: auto;
}

.ws-category {
  font-size: 10.5px;
  font-weight: 800;
  color: var(--tx-muted);
  text-transform: uppercase;
  letter-spacing: 1.1px;
  padding: 14px 10px 4px;
}

.ws-btn {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  color: var(--tx-secondary);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  background: transparent;
  border: 1px solid transparent;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  width: 100%;
  text-align: left;
  font-family: inherit;
}

.ws-btn:hover {
  background: rgba(255, 255, 255, 0.04);
  color: var(--tx-primary);
  border-color: rgba(255, 255, 255, 0.08);
}

.ws-btn.active {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.22), rgba(6, 182, 212, 0.18));
  border-color: rgba(99, 102, 241, 0.45);
  color: #fff;
  font-weight: 700;
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.18);
}

.ws-btn .icon {
  font-size: 17px;
  width: 22px;
  text-align: center;
}

.ws-footer {
  padding: 14px 18px;
  border-top: 1px solid var(--card-border);
  font-size: 11px;
  color: var(--tx-muted);
  background: rgba(0, 0, 0, 0.25);
}

.workspace-content {
  flex: 1;
  padding: 24px 32px 80px;
  max-width: 1260px;
  overflow-y: auto;
}

.tab-pane {
  display: none;
  animation: paneFadeIn 0.22s ease-out forwards;
}

.tab-pane.active {
  display: block;
}

@keyframes paneFadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 960px) {
  .workspace-layout { flex-direction: column; }
  .workspace-sidebar { width: 100%; height: auto; position: static; }
  .workspace-content { padding: 16px 16px 60px; }
}
</style>
"""

# Extract Exact Sections from original HTML
# 1. Stats Row
stats_html = orig[orig.find("<!-- ─ Stats Row ─ -->"):orig.find("<!-- ══════════════════════════════════════════════════════════")]

# 2. AI Real Editing School
school_html = orig[orig.find("<!-- ══════════════════════════════════════════════════════════"):orig.find("<!-- ─ MASTER SECTOR 2: AI Difference Heatmap Visualizer ─ -->")]

# 3. Difference Heatmap
heatmap_html = orig[orig.find("<!-- ─ MASTER SECTOR 2: AI Difference Heatmap Visualizer ─ -->"):orig.find("<!-- ─ MASTER SECTOR: 11 Professional Studio Editing Services ─ -->")]

# 4. 11 Commercial Studio Services
services_html = orig[orig.find("<!-- ─ MASTER SECTOR: 11 Professional Studio Editing Services ─ -->"):orig.find("<!-- ─ NEW FEATURE 1: Object Recognition & Visual Tracking Spotlight ─ -->")]

# 5. YOLO Object Recognition
yolo_html = orig[orig.find("<!-- ─ NEW FEATURE 1: Object Recognition & Visual Tracking Spotlight ─ -->"):orig.find("<!-- ─ NEW FEATURE 2: OpenCV & Scikit-Image Multi-Vision Toolkit ─ -->")]

# 6. Computer Vision Lab
cv_html = orig[orig.find("<!-- ─ NEW FEATURE 2: OpenCV & Scikit-Image Multi-Vision Toolkit ─ -->"):orig.find("<!-- ─ NEW FEATURE 3: Magic Eraser & Inpainting Brush Studio ─ -->")]

# 7. Magic Eraser Inpainting
magic_html = orig[orig.find("<!-- ─ NEW FEATURE 3: Magic Eraser & Inpainting Brush Studio ─ -->"):orig.find("<!-- ─ Main Section: Upload & Controls ─ -->")]

# 8. Main Section (Left: Pair Upload & Finetune Controls | Right: Playground & Engine Logs)
main_sec = orig[orig.find("<!-- ─ Main Section: Upload & Controls ─ -->"):orig.find("<script>")]

# Extract Left Column: Finetune & Upload
left_col_start = main_sec.find("<!-- Left Column: Fine-tune & Pairs -->")
right_col_start = main_sec.find("<!-- Right Column: Interactive Inference Playground")
finetune_html = main_sec[left_col_start:right_col_start]

# Extract Right Column: Playground & Engine Logs
restore_html = main_sec[right_col_start:]

# Close any open divs in extracted columns
finetune_tab_content = f"""
<div class="panel-box" style="margin-bottom: 20px;">
  <div class="panel-header">
    <h2><span class="step-badge">⚡</span> পেয়ার ফাইন-টিউন ও ট্রেনিং প্যারামিটার কন্ট্রোল</h2>
    <span style="font-size: 11px; color: var(--cyan); font-weight:700;">Manual Batch & Fine-tune Config</span>
  </div>
  <p style="font-size: 12.5px; color: var(--tx-secondary); margin-bottom: 16px;">
    এখানে ম্যানুয়াল পেয়ার আপলোড করে হাইপারপ্যারামিটার (Steps, Learning Rate, Batch Size) কনফিগার করে ডিপ ফাইন-টিউনিং শুরু করতে পারেন।
  </p>
</div>
{finetune_html}
</div>
"""

restore_tab_content = f"""
<div class="panel-box" style="margin-bottom: 20px;">
  <div class="panel-header">
    <h2><span class="step-badge">✨</span> এআই রিস্টোরেশন ল্যাব ও ইন্টারঅ্যাক্টিভ স্লাইডার</h2>
    <span style="font-size: 11px; color: var(--accent); font-weight:700;">Real-time Model Inference</span>
  </div>
  <p style="font-size: 12.5px; color: var(--tx-secondary); margin-bottom: 16px;">
    যেকোনো নষ্ট বা সাধারণ ছবি আপলোড করুন — এআই মডেল রিয়েল-টাইমে রিস্টোর করে বিফোর/আফটার স্লাইডারে তুলনা দেখাবে।
  </p>
</div>
<div style="display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px;">
{restore_html}
</div>
"""

dashboard_tab_content = f"""
<div class="panel-header" style="margin-bottom: 18px;">
  <h2><span class="step-badge">📊</span> এআই ট্রেনিং স্ট্যাটাস ও লাইভ পারফরম্যান্স ওভারভিউ</h2>
  <span style="font-size: 11px; color: var(--cyan); font-weight:700;">Autonomous Learning Dashboard</span>
</div>

{stats_html}

<div style="display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; margin-top: 24px;">
  <div class="panel-box">
    <div class="panel-header">
      <h2><span class="step-badge">📜</span> লাইভ ট্রেইনিং ইঞ্জিন লগ (Console Tail)</h2>
      <span id="badge-ft" style="font-size: 11px; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.05);">Idle</span>
    </div>
    <div class="terminal-area" id="term-log">ইঞ্জিন প্রস্তুত — পেয়ার আপলোড বা ফাইন-টিউন ট্র্যাকিং…</div>
  </div>

  <div class="panel-box">
    <div class="panel-header">
      <h2><span class="step-badge">💡</span> ISBD স্টুডিও গাইড ও ফিচার শর্টকাট</h2>
    </div>
    <div style="font-size: 12.5px; color: var(--tx-secondary); line-height: 1.9;">
      <div style="margin-bottom: 8px;">🎓 <strong>এডিটিং স্কুল:</strong> ফটোশপের Before/After পেয়ার দিয়ে এআই-কে নতুন স্কিল শেখান।</div>
      <div style="margin-bottom: 8px;">🔥 <strong>ডিফারেন্স হিটম্যাপ:</strong> ছবির পরিবর্তিত এরিয়া থার্মাল কালারে পরীক্ষা করুন।</div>
      <div style="margin-bottom: 8px;">💎 <strong>১১টি কমার্শিয়াল সার্ভিস:</strong> এক ক্লিকে ক্লিপিং পাথ, রিটাচিং, শ্যাডো তৈরি।</div>
      <div style="margin-bottom: 8px;">🪄 <strong>ম্যাজিক ইরেজার:</strong> ব্রাশ দিয়ে দাগ, অনাকাঙ্ক্ষিত অবজেক্ট বা ওয়াটারমার্ক মুছুন।</div>
      <div>👁️ <strong>অবজেক্ট ট্র্যাকিং:</strong> YOLOv8 এআই দিয়ে ছবিতে থাকা সব উপাদান চিহ্নিত করুন।</div>
    </div>
  </div>
</div>
"""

# Tab switcher JS
tab_script = """
<script>
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.ws-btn').forEach(b => b.classList.remove('active'));
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
  
  const btn = document.querySelector(`[data-tab="${tabId}"]`);
  if (btn) btn.classList.add('active');
  
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
</script>
"""

# Final Assembly
assembled_html = f"""{head_html}
{enhanced_styles}
</head>
<body>

<div class="toasts" id="toasts"></div>

<div class="workspace-layout">
  <!-- ═════════ SIDEBAR NAVIGATION ═════════ -->
  <aside class="workspace-sidebar">
    <div class="ws-brand">
      <div class="brand">
        <div class="brand-icon">🚀</div>
        <div class="brand-text">
          <h1>ISBD Studio</h1>
          <span>AI Train & Test Suite</span>
        </div>
      </div>
      <div class="top-pill" style="margin-top: 14px; width: 100%; justify-content: center;">
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

      <div class="ws-category" style="margin-top: 6px;">🧪 AI টেস্টিং (Testing Lab)</div>
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
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span>মডেল: <strong>ISBD v1.00</strong></span>
        <span style="color: var(--cyan); font-weight:700;">24/7 Active</span>
      </div>
    </div>
  </aside>

  <!-- ═════════ CONTENT AREA ═════════ -->
  <main class="workspace-content">
    
    <!-- Tab 1: Dashboard -->
    <div id="tab-dashboard" class="tab-pane active">
      {dashboard_tab_content}
    </div>

    <!-- Tab 2: AI School -->
    <div id="tab-school" class="tab-pane">
      {school_html}
    </div>

    <!-- Tab 3: Finetune -->
    <div id="tab-finetune" class="tab-pane">
      {finetune_tab_content}
    </div>

    <!-- Tab 4: Restoration Playground -->
    <div id="tab-restore" class="tab-pane">
      {restore_tab_content}
    </div>

    <!-- Tab 5: Heatmap -->
    <div id="tab-heatmap" class="tab-pane">
      {heatmap_html}
    </div>

    <!-- Tab 6: Commercial Services -->
    <div id="tab-services" class="tab-pane">
      {services_html}
    </div>

    <!-- Tab 7: Magic Eraser -->
    <div id="tab-magic" class="tab-pane">
      {magic_html}
    </div>

    <!-- Tab 8: YOLO -->
    <div id="tab-yolo" class="tab-pane">
      {yolo_html}
    </div>

    <!-- Tab 9: Computer Vision Lab -->
    <div id="tab-cv" class="tab-pane">
      {cv_html}
    </div>

  </main>
</div>

{tab_script}
{scripts}
"""

with open("/home/imon/isbd_model/isbd/templates/index.html", "w", encoding="utf-8") as f:
    f.write(assembled_html)

print("Generated 100% complete index.html! Size:", len(assembled_html))
