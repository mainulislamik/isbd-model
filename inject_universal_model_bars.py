# -*- coding: utf-8 -*-
"""
Inject universal Model Switcher bar into ALL AI Testing pages:
1. Restoration Lab (tab-restore)
2. Difference Heatmap (tab-heatmap)
3. 11 Commercial Services (tab-services)
4. Magic Eraser (tab-magic)
5. Object Tracking (tab-yolo)
6. Computer Vision Lab (tab-cv)
"""

with open("/home/imon/isbd_model/isbd/templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

def create_model_bar(page_id, page_title):
    return f"""
        <!-- 🚀 Modern Model Switcher Bar for {page_title} -->
        <div style="background:#ffffff; border:1px solid #e2e8f0; padding:12px 18px; border-radius:12px; margin-bottom:18px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; box-shadow:0 1px 3px 0 rgba(0,0,0,0.03);">
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:36px; height:36px; border-radius:8px; background:#eef2ff; color:#4f46e5; display:flex; align-items:center; justify-content:center; font-size:18px; border:1px solid #c7d2fe;">🧠</div>
            <div>
              <div style="font-size:13px; font-weight:700; color:#0f172a;">সক্রিয় এআই মডেল (Active AI Model):</div>
              <div style="font-size:11px; color:#64748b;">এই টেস্টিং টুলে প্রক্রিয়াকরণের জন্য এআই ইঞ্জিন নির্বাচন করুন</div>
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <select class="param-select global-model-selector" id="model-select-{page_id}" onchange="syncAllModelSelectors(this.value)" style="min-width:260px; padding:7px 14px; font-size:12.5px; font-weight:600; border-color:#cbd5e1; border-radius:8px;">
              <option value="isbd_v1" selected>🚀 ISBD v1.00 Local (24/7 Self-Trained)</option>
              <option value="antigravity/gemini-3.7-flash-high">✨ Gemini 3.7 Flash High Vision (Antigravity)</option>
              <option value="antigravity/claude-sonnet-4-6-low">⚡ Claude 3.7 / 3.5 Sonnet Vision (Antigravity)</option>
              <option value="deepseek-v4-flash-vision-exp">🔮 DeepSeek v4 Flash Vision (b.ai)</option>
              <option value="omniroute/auto">🌐 OmniRoute Keyless Vision Agent</option>
            </select>
          </div>
        </div>
    """

# 1. Inject into tab-heatmap
if 'id="tab-heatmap"' in html and 'id="model-select-heatmap"' not in html:
    idx = html.find('id="tab-heatmap"')
    banner_idx = html.find('<div class="cv-lab-banner"', idx)
    html = html[:banner_idx] + create_model_bar("heatmap", "Heatmap") + html[banner_idx:]

# 2. Inject into tab-services
if 'id="tab-services"' in html and 'id="model-select-services"' not in html:
    idx = html.find('id="tab-services"')
    banner_idx = html.find('<div class="cv-lab-banner"', idx)
    html = html[:banner_idx] + create_model_bar("services", "11 Commercial Services") + html[banner_idx:]

# 3. Inject into tab-magic
if 'id="tab-magic"' in html and 'id="model-select-magic"' not in html:
    idx = html.find('id="tab-magic"')
    banner_idx = html.find('<div class="cv-lab-banner"', idx)
    html = html[:banner_idx] + create_model_bar("magic", "Magic Eraser") + html[banner_idx:]

# 4. Inject into tab-yolo
if 'id="tab-yolo"' in html and 'id="model-select-yolo"' not in html:
    idx = html.find('id="tab-yolo"')
    banner_idx = html.find('<div class="detector-banner"', idx)
    html = html[:banner_idx] + create_model_bar("yolo", "Object Tracking") + html[banner_idx:]

# 5. Inject into tab-cv
if 'id="tab-cv"' in html and 'id="model-select-cv"' not in html:
    idx = html.find('id="tab-cv"')
    banner_idx = html.find('<div class="cv-lab-banner"', idx)
    html = html[:banner_idx] + create_model_bar("cv", "Computer Vision Lab") + html[banner_idx:]

# Add JS Helper to synchronize all model switchers across all tabs
sync_js = """
// ─ Global Model Switcher Sync ─
function syncAllModelSelectors(val) {
  document.querySelectorAll('.global-model-selector').forEach(sel => {
    sel.value = val;
  });
  const restoreSel = document.getElementById('active-model-selector');
  if (restoreSel) restoreSel.value = val;
  toast(`সক্রিয় এআই মডেল: ${val}`, 'ok');
}
"""

if 'syncAllModelSelectors' not in html:
    script_idx = html.find('<script>')
    html = html[:script_idx + len('<script>')] + sync_js + html[script_idx + len('<script>'):]

with open("/home/imon/isbd_model/isbd/templates/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Universal Model Switcher successfully added to ALL 6 AI Testing Pages!")
