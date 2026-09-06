# -*- coding: utf-8 -*-
"""
Terminal Viewer Injection Script for All AI Testing Pages:
Adds interactive, modern live terminal logs directly beneath the results of each testing tab.
"""

with open("/home/imon/isbd_model/isbd/templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

def terminal_box(tab_id):
    return f"""
        <!-- 🖥️ Live Terminal Log Console for {tab_id} -->
        <div style="margin-top:14px; background:#0f172a; border:1px solid #334155; border-radius:10px; padding:12px 14px; font-family:'JetBrains Mono',monospace; box-shadow:inset 0 2px 4px rgba(0,0,0,0.3);">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; border-bottom:1px solid #1e293b; padding-bottom:4px;">
            <div style="display:flex; align-items:center; gap:6px; font-size:11px; color:#38bdf8; font-weight:700;">
              <span>🖥️ LIVE ENGINE TERMINAL</span>
              <span style="font-size:9px; background:#1e293b; color:#94a3b8; padding:1px 6px; border-radius:4px;">Execution Stream</span>
            </div>
            <span style="font-size:10px; color:#10b981; font-weight:700;">● Online</span>
          </div>
          <div id="term-log-{tab_id}" style="color:#e2e8f0; font-size:11.5px; line-height:1.6; white-space:pre-wrap; max-height:140px; overflow-y:auto;">
[SYSTEM] Ready for inference. Select image & AI model to stream execution logs...
          </div>
        </div>
    """

# 1. Add to 11 Services
if 'id="term-log-services"' not in html:
    idx = html.find('id="service-result"')
    if idx != -1:
        end_tag = html.find('</div>', idx)
        html = html[:end_tag] + terminal_box("services") + html[end_tag:]

# 2. Add to Object Tracking / YOLO
if 'id="term-log-yolo"' not in html:
    idx = html.find('id="det-result-container"')
    if idx != -1:
        end_tag = html.find('</div>\n        </div>\n      </div>', idx)
        if end_tag != -1:
            html = html[:end_tag] + terminal_box("yolo") + html[end_tag:]

# 3. Add to Computer Vision Lab
if 'id="term-log-cv"' not in html:
    idx = html.find('id="cv-result"')
    if idx != -1:
        end_tag = html.find('</div>', idx)
        html = html[:end_tag] + terminal_box("cv") + html[end_tag:]

# Update JS to pipe logs into terminal boxes
js_updater = """
function appendTabTerminalLog(tabId, logText) {
  const term = document.getElementById(`term-log-${tabId}`);
  if (term && logText) {
    term.textContent = logText;
  }
}
"""

if 'appendTabTerminalLog' not in html:
    script_idx = html.find('<script>')
    html = html[:script_idx + len('<script>')] + js_updater + html[script_idx + len('<script>'):]

with open("/home/imon/isbd_model/isbd/templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Live Terminal Log Consoles successfully injected into testing pages!")
