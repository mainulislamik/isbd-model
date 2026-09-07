path = "/home/imon/isbd_model/isbd/templates/index.html"
with open(path, "r") as f:
    text = f.read()

tab_func = """
    // --- Navigation Tabs Fix ---
    window.switchTab = function(tabId) {
      document.querySelectorAll('.ws-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      const btn = document.querySelector(`.ws-btn[data-tab='${tabId}']`);
      if (btn) btn.classList.add('active');
      const pane = document.getElementById(tabId);
      if (pane) pane.classList.add('active');
    };
"""

text = text.replace("window.chartInstances = {};", tab_func + "\nwindow.chartInstances = {};")

# Also add the cloud tab button to the sidebar
sidebar_ext = """      <button class="ws-btn" data-tab="tab-settings" onclick="switchTab('tab-settings')">
        <span class="icon">⚙️</span> সিস্টেম সেটিংস
      </button>
      
      <div class="ws-category">☁️ ক্লাউড হাব (Cloud Hub)</div>
      <button class="ws-btn" data-tab="tab-cloud" onclick="switchTab('tab-cloud')">
        <span class="icon">🚀</span> GitHub ও Colab GPU
      </button>"""

text = text.replace("""      <button class="ws-btn" data-tab="tab-settings" onclick="switchTab('tab-settings')">
        <span class="icon">⚙️</span> সিস্টেম সেটিংস
      </button>""", sidebar_ext)


with open(path, "w") as f:
    f.write(text)
