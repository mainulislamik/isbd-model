import os
import re

html_path = "/home/imon/isbd_model/isbd/templates/index.html"
with open(html_path, "r", encoding="utf-8") as f:
    raw_html = f.read()

# 1. Update CSS to include modern responsive sidebar layout
css_replacement = """
/* ─ Modern Master Sidebar + Content Layout ─ */
.app-layout {
  display: flex;
  min-height: 100vh;
  position: relative;
}

.sidebar {
  width: 280px;
  background: rgba(10, 14, 23, 0.95);
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

.sidebar-header {
  padding: 24px 20px 20px;
  border-bottom: 1px solid var(--card-border);
}

.sidebar-nav {
  padding: 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
  overflow-y: auto;
}

.nav-category {
  font-size: 10.5px;
  font-weight: 700;
  color: var(--tx-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: 12px 10px 6px;
}

.nav-tab-btn {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 11px 14px;
  border-radius: var(--radius-sm);
  color: var(--tx-secondary);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  background: transparent;
  border: 1px solid transparent;
  transition: all 0.2s ease;
  width: 100%;
  text-align: left;
}

.nav-tab-btn:hover {
  background: rgba(255, 255, 255, 0.04);
  color: var(--tx-primary);
  border-color: rgba(255, 255, 255, 0.06);
}

.nav-tab-btn.active {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(6, 182, 212, 0.15));
  border-color: rgba(99, 102, 241, 0.4);
  color: #fff;
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
}

.nav-tab-btn .tab-icon {
  font-size: 18px;
}

.sidebar-footer {
  padding: 16px;
  border-top: 1px solid var(--card-border);
  background: rgba(0, 0, 0, 0.2);
}

.content-main {
  flex: 1;
  padding: 24px 32px 80px;
  max-width: 1280px;
  overflow-y: auto;
}

.tab-pane {
  display: none;
  animation: fadeInPane 0.25s ease forwards;
}

.tab-pane.active {
  display: block;
}

@keyframes fadeInPane {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 960px) {
  .app-layout { flex-direction: column; }
  .sidebar { width: 100%; height: auto; position: static; }
  .content-main { padding: 16px 16px 60px; }
}
"""

# Inject CSS before </style>
raw_html = raw_html.replace("</style>", css_replacement + "\n</style>")

# 2. Modify Body wrapper to incorporate Sidebar & Tab Panes
body_start = "<body>\n\n<div class=\"toasts\" id=\"toasts\"></div>"

new_body_structure = """<body>

<div class="toasts" id="toasts"></div>

<div class="app-layout">
  <!-- ═════════ SIDEBAR NAVIGATION ═════════ -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="brand">
        <div class="brand-icon">🚀</div>
        <div class="brand-text">
          <h1>ISBD Studio</h1>
          <span>AI Train & Test Engine</span>
        </div>
      </div>
      <div class="top-pill" style="margin-top: 14px; width: 100%; justify-content: center;">
        <span class="pulse-dot"></span>
        <span id="nav-status">Vision & Trainer Live</span>
      </div>
    </div>

    <nav class="sidebar-nav">
      <div class="nav-category">🧠 AI ট্রেনিং (Training)</div>
      <button class="nav-tab-btn active" onclick="switchTab('tab-dashboard')">
        <span class="tab-icon">📊</span> লাইভ ড্যাশবোর্ড ও স্ট্যাটাস
      </button>
      <button class="nav-tab-btn" onclick="switchTab('tab-school')">
        <span class="tab-icon">🎓</span> Before/After এডিটিং স্কুল
      </button>
      <button class="nav-tab-btn" onclick="switchTab('tab-finetune')">
        <span class="tab-icon">⚡</span> পেয়ার ফাইন-টিউন ল্যাব
      </button>

      <div class="nav-category" style="margin-top: 8px;">🧪 AI টেস্টিং (Testing Lab)</div>
      <button class="nav-tab-btn" onclick="switchTab('tab-restore')">
        <span class="tab-icon">✨</span> এআই রিস্টোরেশন ল্যাব
      </button>
      <button class="nav-tab-btn" onclick="switchTab('tab-heatmap')">
        <span class="tab-icon">🔥</span> ডিফারেন্স হিটম্যাপ
      </button>
      <button class="nav-tab-btn" onclick="switchTab('tab-services')">
        <span class="tab-icon">💎</span> ১১টি কমার্শিয়াল সার্ভিস
      </button>
      <button class="nav-tab-btn" onclick="switchTab('tab-magic-eraser')">
        <span class="tab-icon">🪄</span> ম্যাজিক ইরেজার (Inpaint)
      </button>
      <button class="nav-tab-btn" onclick="switchTab('tab-detector')">
        <span class="tab-icon">👁️</span> অবজেক্ট ট্র্যাকিং (YOLO)
      </button>
      <button class="nav-tab-btn" onclick="switchTab('tab-cv-lab')">
        <span class="tab-icon">🎨</span> কম্পিউটার ভিশন ও প্যালেট
      </button>
    </nav>

    <div class="sidebar-footer">
      <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--tx-muted);">
        <span>মডেল ভার্সন: <strong>v1.00 Pro</strong></span>
        <span style="color: var(--cyan); font-weight:700;">24/7 Active</span>
      </div>
    </div>
  </aside>

  <!-- ═════════ MAIN CONTENT TABS ═════════ -->
  <main class="content-main">
"""

# Replace body top
raw_html = raw_html.replace("<body>\n\n<div class=\"toasts\" id=\"toasts\"></div>\n\n<div class=\"wrapper\">", new_body_structure)

# Remove redundant inner navbar from wrapper
raw_html = re.sub(r'  <!-- ─ Navbar ─ -->[\s\S]*?</div>\s*</div>\s*</div>', '', raw_html, count=1)

print("HTML Structure updated successfully!")
