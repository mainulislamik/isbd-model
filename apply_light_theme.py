# -*- coding: utf-8 -*-
"""
Full Light Mode & High-Contrast Typography Transformation for ISBD Studio
Transforms dark variables and low-contrast muted colors into a clean, modern, ultra-readable Light Theme (Stripe/Linear/Vercel Light aesthetic) with high-contrast text and crisp card borders.
"""

with open("/home/imon/isbd_model/isbd/templates/index.html", "r", encoding="utf-8") as f:
    orig = f.read()

# Replace Root Theme Variables with Crisp Light Palette
old_root = """@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --bg: #07090e; --bg-secondary: #0d111a; --card: rgba(18, 23, 37, 0.75);
  --card-border: rgba(255, 255, 255, 0.08); --glass: rgba(255, 255, 255, 0.03);
  --tx-primary: #f1f5f9; --tx-secondary: #94a3b8; --tx-muted: #64748b;
  --accent: #6366f1; --accent-glow: rgba(99, 102, 241, 0.25);
  --cyan: #06b6d4; --cyan-glow: rgba(6, 182, 212, 0.2);
  --success: #10b981; --success-bg: rgba(16, 185, 129, 0.12);
  --warning: #f59e0b; --warning-bg: rgba(245, 158, 11, 0.12);
  --danger: #ef4444; --danger-bg: rgba(239, 68, 68, 0.12);
  --radius-lg: 18px; --radius-md: 12px; --radius-sm: 8px;
}"""

new_root = """@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root {
  --bg: #f8fafc; --bg-secondary: #ffffff; --card: #ffffff;
  --card-border: #e2e8f0; --glass: #f1f5f9;
  --tx-primary: #0f172a; --tx-secondary: #334155; --tx-muted: #64748b;
  --accent: #4f46e5; --accent-glow: rgba(79, 70, 229, 0.15);
  --cyan: #0284c7; --cyan-glow: rgba(2, 132, 199, 0.15);
  --success: #059669; --success-bg: #ecfdf5;
  --warning: #d97706; --warning-bg: #fffbeb;
  --danger: #dc2626; --danger-bg: #fef2f2;
  --radius-lg: 14px; --radius-md: 10px; --radius-sm: 6px;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
}"""

updated = orig.replace(old_root, new_root)

# Replace Body & Global Background
old_body = """body {
  font-family: 'Plus Jakarta Sans', 'Noto Sans Bengali', system-ui, sans-serif;
  background-color: var(--bg); color: var(--tx-primary); min-height: 100vh;
  background-image:
    radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
    radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.1) 0%, transparent 40%);
  background-attachment: fixed; line-height: 1.5;
}"""

new_body = """body {
  font-family: 'Plus Jakarta Sans', 'Noto Sans Bengali', system-ui, sans-serif;
  background-color: #f8fafc; color: var(--tx-primary); min-height: 100vh;
  line-height: 1.55; -webkit-font-smoothing: antialiased;
}"""

updated = updated.replace(old_body, new_body)

# Replace Workspace Layout Styles with Clean White Sidebar & Crisp Contrast
old_ws_styles = """.workspace-sidebar {
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
}"""

new_ws_styles = """.workspace-sidebar {
  width: 280px;
  background: #ffffff;
  border-right: 1px solid #e2e8f0;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.02);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  position: sticky;
  top: 0;
  height: 100vh;
  z-index: 100;
}"""

updated = updated.replace(old_ws_styles, new_ws_styles)

# Update Brand Gradient Text
old_brand_text = """background: linear-gradient(135deg, #fff, #94a3b8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;"""
new_brand_text = """color: #0f172a; font-weight: 800;"""
updated = updated.replace(old_brand_text, new_brand_text)

# Update Inputs, Selects, and Textareas for Clean High-Contrast Light Mode
old_param_input = """.param-input, .param-select {
  width: 100%; background: rgba(255, 255, 255, 0.04); border: 1px solid var(--card-border);
  color: var(--tx-primary); padding: 8px 12px; border-radius: var(--radius-sm); font-size: 12.5px;
  font-family: inherit; outline: none; transition: border-color 0.2s;
}"""

new_param_input = """.param-input, .param-select {
  width: 100%; background: #ffffff; border: 1px solid #cbd5e1;
  color: #0f172a; font-weight: 500; padding: 8px 12px; border-radius: var(--radius-sm); font-size: 13px;
  font-family: inherit; outline: none; transition: all 0.2s; box-shadow: var(--shadow-sm);
}
.param-input:focus, .param-select:focus {
  border-color: #4f46e5; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
}"""
updated = updated.replace(old_param_input, new_param_input)

# Update Sidebar Navigation Buttons
old_ws_btn = """.ws-btn {
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
}"""

new_ws_btn = """.ws-btn {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  color: #334155;
  font-size: 13px;
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
  background: #f1f5f9;
  color: #0f172a;
}

.ws-btn.active {
  background: #ede9fe;
  border-color: #c4b5fd;
  color: #4338ca;
  font-weight: 700;
}"""
updated = updated.replace(old_ws_btn, new_ws_btn)

# Update Terminal Area for High Contrast
old_terminal = """.terminal-area {
  background: #04060a; border: 1px solid var(--card-border); border-radius: var(--radius-md);
  padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #38bdf8;
  max-height: 260px; overflow-y: auto; white-space: pre-wrap; line-height: 1.5;
}"""

new_terminal = """.terminal-area {
  background: #0f172a; border: 1px solid #334155; border-radius: var(--radius-md);
  padding: 14px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #38bdf8;
  max-height: 260px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
}"""
updated = updated.replace(old_terminal, new_terminal)

# Update Cards and Boxes
old_panel_box = """.panel-box {
  background: var(--card); border: 1px solid var(--card-border);
  backdrop-filter: blur(16px); border-radius: var(--radius-lg);
  padding: 22px; margin-bottom: 24px;
}"""

new_panel_box = """.panel-box {
  background: #ffffff; border: 1px solid #e2e8f0;
  border-radius: var(--radius-lg); box-shadow: var(--shadow-sm);
  padding: 22px; margin-bottom: 24px;
}"""
updated = updated.replace(old_panel_box, new_panel_box)

old_stat_card = """.stat-card {
  background: var(--card); border: 1px solid var(--card-border);
  backdrop-filter: blur(16px); border-radius: var(--radius-md);
  padding: 16px 18px; position: relative; overflow: hidden;
  transition: transform 0.2s, border-color 0.2s;
}"""

new_stat_card = """.stat-card {
  background: #ffffff; border: 1px solid #e2e8f0;
  border-radius: var(--radius-md); box-shadow: var(--shadow-sm);
  padding: 16px 18px; position: relative; overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }"""
updated = updated.replace(old_stat_card, new_stat_card)

# Update Banner Bounding Boxes
updated = updated.replace("background: linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(99,102,241,0.08) 100%);", "background: #ffffff; box-shadow: var(--shadow-sm);")
updated = updated.replace("background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(239, 68, 68, 0.06) 100%);", "background: #ffffff; box-shadow: var(--shadow-sm);")
updated = updated.replace("background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(99, 102, 241, 0.08) 100%);", "background: #ffffff; box-shadow: var(--shadow-sm);")
updated = updated.replace("background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);", "background: #ffffff; box-shadow: var(--shadow-sm);")
updated = updated.replace("background: linear-gradient(135deg, rgba(236, 72, 153, 0.08) 0%, rgba(99, 102, 241, 0.06) 100%);", "background: #ffffff; box-shadow: var(--shadow-sm);")
updated = updated.replace(".cv-lab-banner {\n  background: var(--card); border: 1px solid var(--card-border);", ".cv-lab-banner {\n  background: #ffffff; border: 1px solid #e2e8f0; box-shadow: var(--shadow-sm);")

# Update Dropzones
updated = updated.replace("background: var(--glass);", "background: #f8fafc; border-color: #cbd5e1;")
updated = updated.replace("background: rgba(255, 255, 255, 0.02);", "background: #f8fafc; border-color: #e2e8f0;")
updated = updated.replace("color: #fff;", "color: #0f172a;")

# Fix specific text colors in Insights & Status Box
updated = updated.replace("color:#f1f5f9;", "color:#0f172a; font-weight:600;")
updated = updated.replace("color:#94a3b8;", "color:#475569;")
updated = updated.replace("color:#e2e8f0;", "color:#1e293b;")
updated = updated.replace("color:#a5f3fc;", "color:#0369a1;")
updated = updated.replace("color:#6ee7b7;", "color:#047857;")
updated = updated.replace("color:#fbbf24;", "color:#b45309;")
updated = updated.replace("background:rgba(99,102,241,0.08);", "background:#f5f3ff; border:1px solid #ddd6fe;")
updated = updated.replace("background:rgba(0,0,0,0.3);", "background:#f8fafc; border:1px solid #e2e8f0;")
updated = updated.replace("background:rgba(6,182,212,0.08);", "background:#f0fdfa; border:1px solid #99f6e4;")
updated = updated.replace("background:rgba(0,0,0,0.25);", "background:#f8fafc;")
updated = updated.replace("background:rgba(0,0,0,0.2);", "background:#f8fafc;")

# Save transformed Light Theme
with open("/home/imon/isbd_model/isbd/templates/index.html", "w", encoding="utf-8") as f:
    f.write(updated)

print("Full Light Mode & High-Contrast Typography successfully applied! Size:", len(updated))
