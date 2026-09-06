# -*- coding: utf-8 -*-
import re

with open("/home/imon/isbd_model/isbd/templates/index.html", "r", encoding="utf-8") as f:
    orig = f.read()

# Let's extract each distinct section cleanly:
# 1. Head & CSS
head_css = orig[orig.find("<!doctype html>"):orig.find("</head>") + len("</head>")]

# Add sidebar CSS
sidebar_css = """
<style>
/* ─ Sidebar Navigation & Tab Pane System ─ */
.app-container { display: flex; min-height: 100vh; background: var(--bg); }
.app-sidebar {
  width: 270px; background: rgba(13, 17, 26, 0.95); border-right: 1px solid var(--card-border);
  backdrop-filter: blur(20px); display: flex; flex-direction: column; flex-shrink: 0;
  position: sticky; top: 0; height: 100vh; z-index: 100;
}
.side-head { padding: 22px 18px 18px; border-bottom: 1px solid var(--card-border); }
.side-nav { padding: 14px 10px; display: flex; flex-direction: column; gap: 4px; flex: 1; overflow-y: auto; }
.side-cat { font-size: 10px; font-weight: 800; color: var(--tx-muted); text-transform: uppercase; letter-spacing: 1px; padding: 12px 10px 4px; }
.side-btn {
  display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: var(--radius-sm);
  color: var(--tx-secondary); font-size: 12.5px; font-weight: 600; cursor: pointer; background: transparent;
  border: 1px solid transparent; transition: all 0.2s; width: 100%; text-align: left; font-family: inherit;
}
.side-btn:hover { background: rgba(255,255,255,0.04); color: var(--tx-primary); border-color: rgba(255,255,255,0.08); }
.side-btn.active {
  background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.15));
  border-color: rgba(99,102,241,0.4); color: #fff; font-weight: 700;
  box-shadow: 0 4px 14px rgba(99,102,241,0.15);
}
.side-btn .icon { font-size: 16px; width: 22px; text-align: center; }
.side-foot { padding: 14px 18px; border-top: 1px solid var(--card-border); font-size: 11px; color: var(--tx-muted); background: rgba(0,0,0,0.2); }
.app-content { flex: 1; padding: 24px 30px 80px; max-width: 1240px; overflow-y: auto; }
.tab-pane { display: none; animation: fadeInPane 0.2s ease forwards; }
.tab-pane.active { display: block; }
@keyframes fadeInPane { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
@media (max-width: 960px) {
  .app-container { flex-direction: column; }
  .app-sidebar { width: 100%; height: auto; position: static; }
  .app-content { padding: 16px 16px 60px; }
}
</style>
"""

# Extract scripts
script_part = orig[orig.find("<script>"):orig.rfind("</script>") + len("</script>")]

# Extract sections
# Stats
stats_match = re.search(r'(  <!-- ─ Stats Row ─ -->[\s\S]*?</div>\s*</div>)', orig)
stats_html = stats_match.group(1) if stats_match else ""

# School
school_match = re.search(r'(  <!-- ══════════════════════════════════════════════════════════[\s\S]*?🎓 AI REAL EDITING SCHOOL[\s\S]*?</div>\s*</div>\s*</div>)', orig)
school_html = school_match.group(1) if school_match else ""

# Heatmap
heatmap_match = re.search(r'(  <!-- ─ MASTER SECTOR 2: AI Difference Heatmap Visualizer ─ -->[\s\S]*?</div>\s*</div>\s*</div>)', orig)
heatmap_html = heatmap_match.group(1) if heatmap_match else ""

# Commercial services
service_match = re.search(r'(  <!-- ─ MASTER SECTOR: 11 Professional Studio Editing Services ─ -->[\s\S]*?</div>\s*</div>\s*</div>)', orig)
service_html = service_match.group(1) if service_match else ""

# Detector
det_match = re.search(r'(  <!-- ─ NEW FEATURE 1: Object Recognition & Visual Tracking Spotlight ─ -->[\s\S]*?</div>\s*</div>\s*</div>)', orig)
det_html = det_match.group(1) if det_match else ""

# CV Lab
cv_match = re.search(r'(  <!-- ─ NEW FEATURE 2: OpenCV & Scikit-Image Multi-Vision Toolkit ─ -->[\s\S]*?</div>\s*</div>\s*</div>)', orig)
cv_html = cv_match.group(1) if cv_match else ""

# Magic Eraser
inpaint_match = re.search(r'(  <!-- ─ NEW FEATURE 3: Magic Eraser & Inpainting Brush Studio ─ -->[\s\S]*?</div>\s*</div>\s*</div>)', orig)
inpaint_html = inpaint_match.group(1) if inpaint_match else ""

# Main grid (Pairs + Finetune + Playground + Logs)
# Left col: Designer pair upload + Advanced finetune
finetune_match = re.search(r'(  <!-- ─ Main Section: Upload & Controls ─ -->[\s\S]*?<!-- Right Column)', orig)
finetune_html = finetune_match.group(1).replace("<!-- Right Column", "") if finetune_match else ""

# Right col: Playground + Engine logs
play_match = re.search(r'(<!-- Right Column: Interactive Inference Playground[\s\S]*?</div>\s*</div>\s*</div>\s*</div>\s*</div>)', orig)
play_html = play_match.group(1) if play_match else ""

# Build Tab switching JavaScript function
tab_js = """
<script>
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.side-btn').forEach(b => b.classList.remove('active'));
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
  
  const btn = document.querySelector(`[data-tab="${tabId}"]`);
  if (btn) btn.classList.add('active');
  
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
</script>
"""

# Assemble the new complete HTML with Sidebar
final_html = f"""{head_css}
{sidebar_css}
<body>
<div class="toasts" id="toasts"></div>

<div class="app-container">
  <!-- ═════════ SIDEBAR MENU ═════════ -->
  <aside class="app-sidebar">
    <div class="side-head">
      <div class="brand">
        <div class="brand-icon">🚀</div>
        <div class="brand-text">
          <h1>ISBD Studio</h1>
          <span>AI Train & Test Engine</span>
        </div>
      </div>
      <div class="top-pill" style="margin-top: 12px; width: 100%; justify-content: center;">
        <span class="pulse-dot"></span>
        <span id="nav-status">Vision & Trainer Live</span>
      </div>
    </div>

    <nav class="side-nav">
      <div class="side-cat">🧠 AI ট্রেনিং (Training)</div>
      <button class="side-btn active" data-tab="tab-dashboard" onclick="switchTab('tab-dashboard')">
        <span class="icon">📊</span> লাইভ ড্যাশবোর্ড ও মেট্রিক্স
      </button>
      <button class="side-btn" data-tab="tab-school" onclick="switchTab('tab-school')">
        <span class="icon">🎓</span> Before/After এডিটিং স্কুল
      </button>
      <button class="side-btn" data-tab="tab-finetune" onclick="switchTab('tab-finetune')">
        <span class="icon">⚡</span> পেয়ার ফাইন-টিউন ল্যাব
      </button>

      <div class="side-cat">🧪 AI টেস্টিং (Testing Lab)</div>
      <button class="side-btn" data-tab="tab-restore" onclick="switchTab('tab-restore')">
        <span class="icon">✨</span> এআই রিস্টোরেশন ল্যাব
      </button>
      <button class="side-btn" data-tab="tab-heatmap" onclick="switchTab('tab-heatmap')">
        <span class="icon">🔥</span> ডিফারেন্স হিটম্যাপ
      </button>
      <button class="side-btn" data-tab="tab-services" onclick="switchTab('tab-services')">
        <span class="icon">💎</span> ১১টি কমার্শিয়াল সার্ভিস
      </button>
      <button class="side-btn" data-tab="tab-magic" onclick="switchTab('tab-magic')">
        <span class="icon">🪄</span> ম্যাজিক ইরেজার (Inpaint)
      </button>
      <button class="side-btn" data-tab="tab-yolo" onclick="switchTab('tab-yolo')">
        <span class="icon">👁️</span> অবজেক্ট ট্র্যাকিং (YOLO)
      </button>
      <button class="side-btn" data-tab="tab-cv" onclick="switchTab('tab-cv')">
        <span class="icon">🎨</span> কালার প্যালেট ও ফিল্টার
      </button>
    </nav>

    <div class="side-foot">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span>মডেল: <strong>ISBD v1.00</strong></span>
        <span style="color: var(--cyan); font-weight:700;">24/7 Active</span>
      </div>
    </div>
  </aside>

  <!-- ═════════ CONTENT AREA ═════════ -->
  <main class="app-content">

    <!-- 1. DASHBOARD & OVERVIEW TAB -->
    <div id="tab-dashboard" class="tab-pane active">
      <div class="panel-header" style="margin-bottom: 20px;">
        <h2><span class="step-badge">📊</span> এআই ট্রেনিং স্ট্যাটাস ও পারফরম্যান্স মেট্রিক্স</h2>
      </div>
      {stats_html}
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;">
        <div class="panel-box">
          <div class="panel-header">
            <h2><span class="step-badge">📜</span> লাইভ ইঞ্জিন ট্রেইনার লগ</h2>
            <span id="badge-ft" style="font-size: 11px; padding: 2px 8px; border-radius: 12px; background: rgba(255,255,255,0.05);">Idle</span>
          </div>
          <div class="terminal-area" id="term-log">ইঞ্জিন প্রস্তুত — ২৪/৭ সেলফ-লার্নিং সক্রিয়…</div>
        </div>
        <div class="panel-box">
          <div class="panel-header">
            <h2><span class="step-badge">💡</span> কুইক শর্টকাট গাইড</h2>
          </div>
          <p style="font-size: 13px; color: var(--tx-secondary); line-height: 1.8;">
            • <strong>এডিটিং স্কুল:</strong> মানুষের করা আসল Before/After ফটোশপ এডিট আপলোড করে এআই-কে নতুন স্টাইল শেখান।<br>
            • <strong>হিটম্যাপ:</strong> এডিট করা ছবির সূক্ষ্ম পরিবর্তন থার্মাল কালারে পরীক্ষা করুন।<br>
            • <strong>১১টি সার্ভিস:</strong> ক্লিপিং পাথ, রিটাচিং, শ্যাডো ও নেক জয়েন্ট এক ক্লিকে টেস্ট করুন।<br>
            • <strong>ম্যাজিক ইরেজার:</strong> ব্রাশ দিয়ে ড্র করে যেকোনো দাগ বা অবজেক্ট নিখুঁতভাবে মুছুন।
          </p>
        </div>
      </div>
    </div>

    <!-- 2. AI EDITING SCHOOL TAB -->
    <div id="tab-school" class="tab-pane">
      {school_html}
    </div>

    <!-- 3. PAIR FINE-TUNE TAB -->
    <div id="tab-finetune" class="tab-pane">
      {finetune_html}
        </div>
      </div>
    </div>

    <!-- 4. LIVE RESTORATION PLAYGROUND TAB -->
    <div id="tab-restore" class="tab-pane">
      {play_html}
    </div>

    <!-- 5. HEATMAP TAB -->
    <div id="tab-heatmap" class="tab-pane">
      {heatmap_html}
    </div>

    <!-- 6. 11 COMMERCIAL SERVICES TAB -->
    <div id="tab-services" class="tab-pane">
      {service_html}
    </div>

    <!-- 7. MAGIC ERASER TAB -->
    <div id="tab-magic" class="tab-pane">
      {inpaint_html}
    </div>

    <!-- 8. YOLO OBJECT TRACKING TAB -->
    <div id="tab-yolo" class="tab-pane">
      {det_html}
    </div>

    <!-- 9. COMPUTER VISION LAB TAB -->
    <div id="tab-cv" class="tab-pane">
      {cv_html}
    </div>

  </main>
</div>

{tab_js}
{script_part}
</body>
</html>
"""

with open("/home/imon/isbd_model/isbd/templates/index.html", "w", encoding="utf-8") as f:
    f.write(final_html)

print("Generated clean index.html with Sidebar and all 9 Tabs intact! Size:", len(final_html))
