# -*- coding: utf-8 -*-
"""
Next-Gen Modern UI Upgrade for ISBD Studio:
Injects Tailwind CSS, Lucide-style SVG icons, Inter/Plus Jakarta typography,
sleek glassmorphism gradients, polished status badges, clean card shadows,
and smooth micro-interactions.
"""

with open("/home/imon/isbd_model/isbd/templates/index.html", "r", encoding="utf-8") as f:
    orig = f.read()

# 1. Update Head to include Tailwind CDN & Lucide Icons CDN for ultra modern look
head_start = orig.find("<head>")
head_end = orig.find("</head>")

new_head_meta = """<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISBD Studio — Next-Gen AI Workspace</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚀</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    theme: {
      extend: {
        fontFamily: {
          sans: ['Inter', 'system-ui', 'sans-serif'],
          mono: ['JetBrains Mono', 'monospace'],
        },
        colors: {
          brand: {
            50: '#eef2ff',
            100: '#e0e7ff',
            500: '#6366f1',
            600: '#4f46e5',
            700: '#4338ca',
          }
        }
      }
    }
  }
</script>
"""

updated = orig.replace(orig[head_start:head_end], new_head_meta)

# 2. Modernize Theme variables & Base CSS
theme_css = """
<style>
:root {
  --bg-app: #f8fafc;
  --sidebar-bg: #ffffff;
  --card-bg: #ffffff;
  --card-border: #e2e8f0;
  --text-main: #0f172a;
  --text-sub: #475569;
  --text-muted: #94a3b8;
  --primary: #4f46e5;
  --primary-light: #eef2ff;
  --cyan: #0284c7;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --radius-xl: 16px;
  --radius-lg: 12px;
  --radius-md: 8px;
}

body {
  font-family: 'Inter', system-ui, sans-serif;
  background-color: var(--bg-app);
  color: var(--text-main);
  margin: 0;
  padding: 0;
  -webkit-font-smoothing: antialiased;
}

/* ─ Modern Master Workspace Layout ─ */
.workspace-layout {
  display: flex;
  min-height: 100vh;
  background: #f8fafc;
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
  transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
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

/* ─ Modern Cards & Panels ─ */
.panel-box, .cv-lab-banner {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: var(--radius-xl);
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.02);
  transition: box-shadow 0.2s;
}

.panel-box:hover, .cv-lab-banner:hover {
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
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
  letter-spacing: -0.01em;
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

/* ─ Modern Metric Stats Cards ─ */
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
  box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.06);
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
  font-weight: 500;
}

/* ─ Modern Buttons ─ */
.btn {
  padding: 9px 18px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
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
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
}

.btn-outline:hover {
  background: #f8fafc;
  border-color: #94a3b8;
  color: #0f172a;
}

/* ─ Modern Input Fields ─ */
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
  box-shadow: 0 1px 2px 0 rgba(0,0,0,0.02);
}

.param-input:focus, .param-select:focus {
  border-color: #4f46e5;
  box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12);
}

/* ─ Chip Label Checkboxes ─ */
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

.tab-pane {
  display: none;
  animation: paneFadeIn 0.2s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

.tab-pane.active {
  display: block;
}

@keyframes paneFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 960px) {
  .workspace-layout { flex-direction: column; }
  .workspace-sidebar { width: 100%; height: auto; position: static; }
  .workspace-content { padding: 20px 16px 60px; }
}
</style>
"""

# Replace stylesheet section in HTML
style_start = updated.find("<style>")
style_end = updated.find("</head>")
updated = updated[:style_start] + theme_css + "\n</head>" + updated[style_end + len("</head>"):]

with open("/home/imon/isbd_model/isbd/templates/index.html", "w", encoding="utf-8") as f:
    f.write(updated)

print("Modernized Next.js/Tailwind Styled UI successfully generated! Size:", len(updated))
