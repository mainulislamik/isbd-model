"""
ISBD v1.00 — Designer Pair Training Panel (Studio)
Upload before/after Photoshop pairs -> converted in RAM to 64px tensors
(images NEVER touch disk) -> fine-tunes the live model via realfinetune.py.
Run: .venv/bin/python -m uvicorn isbd.panel:app --host 0.0.0.0 --port 8077
"""
import hashlib
import io
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
NPZ = DATA / "pairs.npz"
HASHES = DATA / "hashes.json"
FT_LOG = DATA / "ft.log"
FT_STATE = DATA / "ft_state.json"
IMG = 64
MAX_FILE = 25 * 1024 * 1024

app = FastAPI(title="ISBD Studio")


# ---------- helpers ----------
def _hashes():
    try:
        return json.loads(HASHES.read_text())
    except Exception:
        return []


def _n_pairs():
    if not NPZ.exists():
        return 0
    try:
        with np.load(NPZ) as d:
            return len(d["X"])
    except Exception:
        return 0


def _live():
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", "isbd-train", "--no-pager", "-n", "20", "-o", "cat"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        last = {"step": 0, "loss": 0.0}
        for line in out.splitlines():
            if line.strip().startswith("step"):
                p = line.replace("|", " ").split()
                try:
                    last = {"step": int(p[1]), "loss": float(p[3])}
                except Exception:
                    pass
        return last
    except Exception:
        return {"step": 0, "loss": 0.0}


def _lock_busy():
    import fcntl
    path = ROOT / "checkpoints" / "train.lock"
    if not path.exists():
        return False
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _ft_state():
    try:
        st = json.loads(FT_STATE.read_text())
    except Exception:
        st = {"running": False}
    if st.get("running") and st.get("pid"):
        try:
            os.kill(st["pid"], 0)
        except OSError:
            st["running"] = False
            st["stale"] = True
            FT_STATE.write_text(json.dumps(st))
    return st


def _log_tail(n=60):
    try:
        return "\n".join(FT_LOG.read_text().splitlines()[-n:])
    except Exception:
        return ""


# ---------- API ----------
@app.post("/api/pair")
async def add_pair(before: UploadFile = File(...), after: UploadFile = File(...)):
    b, a = await before.read(), await after.read()
    if not b or not a:
        raise HTTPException(400, "ফাইল খালি")
    if len(b) > MAX_FILE or len(a) > MAX_FILE:
        raise HTTPException(413, "ফাইল 25MB-এর বেশি")
    h = hashlib.sha256(b + b"|" + a).hexdigest()
    hs = _hashes()
    if h in hs:
        return {"ok": True, "dup": True, "pairs": len(hs)}

    def prep(bs):
        img = Image.open(io.BytesIO(bs)).convert("RGB")
        arr = np.asarray(img.resize((IMG, IMG), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        return np.ascontiguousarray(arr.transpose(2, 0, 1))

    try:
        x, y = prep(b), prep(a)
    except Exception:
        raise HTTPException(400, "ছবি পড়া যায়নি — JPG/PNG/WebP দিন")

    if NPZ.exists():
        with np.load(NPZ) as d:
            X, Y = d["X"], d["Y"]
        X, Y = np.concatenate([X, x[None]]), np.concatenate([Y, y[None]])
    else:
        X, Y = x[None], y[None]
    np.savez(NPZ, X=X.astype(np.float32), Y=Y.astype(np.float32))  # tensors only — no images
    hs.append(h)
    HASHES.write_text(json.dumps(hs))
    return {"ok": True, "dup": False, "pairs": len(hs)}


@app.post("/api/train")
async def train(steps: int = 300):
    steps = max(20, min(steps, 2000))
    if _n_pairs() == 0:
        raise HTTPException(400, "আগে অন্তত একটি before/after পেয়ার আপলোড করুন")
    if _ft_state().get("running"):
        raise HTTPException(409, "ফাইন-টিউন ইতিমধ্যে চলছে — শেষ হতে দিন")

    def worker():
        FT_STATE.write_text(json.dumps({"running": True, "pid": 0, "started": time.time()}))
        with open(FT_LOG, "w") as log:
            p = subprocess.Popen(
                [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "isbd" / "realfinetune.py"),
                 "--steps", str(steps), "--wait-lock", "900"],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            FT_STATE.write_text(json.dumps({"running": True, "pid": p.pid, "started": time.time()}))
            p.wait()
            FT_STATE.write_text(json.dumps({"running": False, "pid": p.pid, "code": p.returncode,
                                            "ended": time.time()}))

    threading.Thread(target=worker, daemon=True).start()
    return {"queued": True, "steps": steps, "pairs": _n_pairs()}


@app.post("/api/purge")
async def purge():
    if _ft_state().get("running"):
        raise HTTPException(409, "ফাইন-টিউন চলাকালীন ডেটা মুছতে পারবেন না")
    NPZ.unlink(missing_ok=True)
    HASHES.unlink(missing_ok=True)
    return {"ok": True, "pairs": 0}


@app.get("/api/status")
async def status():
    return {
        "live": _live(),
        "lock_busy": _lock_busy(),
        "pairs": _n_pairs(),
        "ft": _ft_state(),
        "log": _log_tail(),
    }


# ---------- UI ----------
HTML = """<!doctype html><html lang="bn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISBD Studio — ডিজাইনার পেয়ার ট্রেনিং</title>
<style>
:root{--bg:#0e1117;--card:#161a23;--line:#2a3040;--tx:#dce3f0;--dim:#8b93a7;--acc:#4cc2ff;--ok:#5ee27f;--warn:#ffb454}
*{box-sizing:border-box}body{margin:0;font-family:'Noto Sans Bengali','Hind Siliguri',system-ui,sans-serif;background:var(--bg);color:var(--tx)}
.wrap{max-width:880px;margin:0 auto;padding:18px}
h1{font-size:21px;margin:0 0 4px}.sub{color:var(--dim);font-size:13px;margin-bottom:14px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;font-size:13px}
.chip b{color:var(--acc)}
h2{font-size:15px;margin:20px 0 8px}
.row{display:grid;grid-template-columns:1fr 1fr auto;gap:10px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:10px;align-items:start}
label{font-size:12px;color:var(--dim);display:block;margin-bottom:4px}
input[type=file]{width:100%;font-size:12px;color:var(--dim)}
img.th{width:54px;height:54px;object-fit:cover;border-radius:6px;border:1px solid var(--line);margin-top:6px}
.btns{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0;align-items:center}
button{background:var(--acc);border:0;color:#06121c;font-weight:700;border-radius:8px;padding:10px 18px;cursor:pointer;font-size:14px}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--tx)}
button:disabled{opacity:.45;cursor:not-allowed}
select{background:var(--card);color:var(--tx);border:1px solid var(--line);border-radius:8px;padding:9px}
.note{background:#1a2130;border:1px solid #2b3b55;border-radius:10px;padding:10px 12px;font-size:12.5px;color:var(--dim);margin:12px 0;line-height:1.8}
pre#log{background:#0a0d13;border:1px solid var(--line);border-radius:10px;padding:10px;font-size:12px;max-height:280px;overflow:auto;white-space:pre-wrap;color:#a9e5b5;min-height:40px}
.st{font-size:12px;margin-top:6px;color:var(--ok)}.st.err{color:#ff7a7a}.st.dup{color:var(--warn)}
.x{background:transparent;border:1px solid var(--line);color:#ff7a7a;border-radius:8px;padding:6px 10px;font-size:12px}
</style></head><body><div class="wrap">
<h1>🎨 ISBD Studio</h1>
<div class="sub">ডিজাইনারদের before/after পেয়ার দিয়ে ISBD v1.00 ফাইন-টিউন</div>
<div class="chips">
 <span class="chip">🔥 24/7 step: <b id="lstep">—</b></span>
 <span class="chip">loss: <b id="lloss">—</b></span>
 <span class="chip">📦 পেয়ার: <b id="npairs">—</b></span>
 <span class="chip">🔒 ট্রেনার lock: <b id="lock">—</b></span>
 <span class="chip">⚙️ ফাইন-টিউন: <b id="ft">—</b></span>
</div>

<h2>১. পেয়ার যোগ করুন (আগে → পরে)</h2>
<div id="rows"></div>
<button class="ghost" onclick="addRow()">+ আরেকটি পেয়ার</button>

<h2>২. প্রসেস ও ফাইন-টিউন</h2>
<div class="btns">
 <select id="steps">
  <option value="100">১০০ ধাপ (দ্রুত)</option>
  <option value="300" selected>৩০০ ধাপ (স্বাভাবিক)</option>
  <option value="600">৬০০ ধাপ</option>
  <option value="1200">১২০০ ধাপ (গভীর)</option>
 </select>
 <button id="tr" onclick="doTrain()">▶ প্রসেস ও ফাইন-টিউন শুরু</button>
 <button class="ghost" onclick="if(confirm('সব টেনসর-ডেটা মুছে ফেলবেন?'))purge()">🗑 সব ডেটা মুছুন</button>
</div>

<div class="note">🔒 <b>আপনার ছবি কোথাও সংরক্ষিত হয় না।</b> আপলোডের সাথে সাথেই মেমরিতে 64px টেনসরে রূপান্তরিত হয় — ডিস্কে কোনো ছবি লেখা হয় না। ট্রেনিংয়ে শুধু টেনসর ব্যবহৃত হয়; উপরের 🗑 বাটনে সেটাও সব মুছে ফেলা যায়।<br>
⚙️ ফাইন-টিউন 24/7 ট্রেনারের lock মুক্ত হলে নিজে থেকেই শুরু হয় — নিচের লগে অপেক্ষার অবস্থা দেখা যায়।</div>

<pre id="log">…</pre>
</div>
<script>
function addRow(){const r=document.createElement('div');r.className='row';
r.innerHTML='<div><label>আগে — Photoshop-এর আগের ছবি</label><input type="file" accept="image/*" onchange="th(this)"><img class="th" style="display:none"></div>'
 +'<div><label>পরে / ফাইনাল — ডিজাইনারের এডিট</label><input type="file" accept="image/*" onchange="th(this)"><img class="th" style="display:none"></div>'
 +'<button class="x" onclick="this.closest(\'.row\').remove()">✕</button>';
document.getElementById('rows').appendChild(r);}
function th(inp){const f=inp.files[0];if(!f)return;const im=inp.parentElement.querySelector('img');
im.src=URL.createObjectURL(f);im.style.display='block';}
function log(s){const el=document.getElementById('log');el.textContent=s+'\\n'+el.textContent;}
async function doTrain(){
 const rs=[...document.querySelectorAll('#rows .row')];
 if(!rs.length){alert('আগে অন্তত একটি পেয়ার যোগ করুন');return}
 const tr=document.getElementById('tr');tr.disabled=true;log('পেয়ার আপলোড হচ্ছে…');
 let ok=0;
 for(const r of rs){
  const [bf,af]=r.querySelectorAll('input[type=file]');
  const s=document.createElement('div');s.className='st';r.appendChild(s);
  if(!bf.files[0]||!af.files[0]){s.className='st err';s.textContent='✗ দুটো ছবি দিন';continue}
  const fd=new FormData();fd.append('before',bf.files[0]);fd.append('after',af.files[0]);
  try{const res=await fetch('/api/pair',{method:'POST',body:fd});const j=await res.json();
   if(res.ok&&!j.dup){ok++;s.textContent='✓ যোগ হয়েছে (মোট '+j.pairs+')'}
   else if(res.ok&&j.dup){s.className='st dup';s.textContent='⚠ ডুপ্লিকেট — আগেই আছে'}
   else{s.className='st err';s.textContent='✗ '+(j.detail||'ব্যর্থ')}
  }catch(e){s.className='st err';s.textContent='✗ নেটওয়ার্ক সমস্যা'}
 }
 if(ok===0){log('নতুন পেয়ার যোগ হয়নি — ট্রেন হবে না।');tr.disabled=false;return}
 const steps=document.getElementById('steps').value;
 try{const res=await fetch('/api/train?steps='+steps,{method:'POST'});const j=await res.json();
  log(res.ok?('ফাইন-টিউন শুরু: '+steps+' ধাপ, '+j.pairs+' পেয়ার — lock মুক্ত হলেই চলবে'):'⚠ '+(j.detail||'ব্যর্থ'));
 }catch(e){log('⚠ শুরু করা যায়নি')}
 tr.disabled=false;
}
function purge(){fetch('/api/purge',{method:'POST'}).then(()=>{log('🗑 সব টেনসর-ডেটা মুছে ফেলা হয়েছে');poll()}).catch(()=>{})}
async function poll(){try{const j=await(await fetch('/api/status')).json();
 document.getElementById('lstep').textContent=j.live.step?j.live.step.toLocaleString():'—';
 document.getElementById('lloss').textContent=j.live.loss||'—';
 document.getElementById('npairs').textContent=j.pairs;
 document.getElementById('lock').textContent=j.lock_busy?'busy':'free';
 document.getElementById('ft').textContent=j.ft.running?'চলছে…':'idle';
 if(j.log)document.getElementById('log').textContent=j.log;}catch(e){}}
setInterval(poll,3000);poll();addRow();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML
