// ISBD Studio v1.00 — Adobe Photoshop UXP Plugin Script
let photoshop = null;
let uxp = null;

try {
  photoshop = require('photoshop');
  uxp = require('uxp');
} catch (e) {
  console.warn('Running outside Photoshop UXP host runtime:', e);
}

function log(msg) {
  const box = document.getElementById('log-box');
  if (box) {
    box.textContent = msg;
    box.scrollTop = box.scrollHeight;
  }
  console.log('[ISBD UXP]', msg);
}

function getApiUrl() {
  const input = document.getElementById('api-url');
  return (input ? input.value : 'http://127.0.0.1:8077').replace(/\/+$/, '');
}

async function checkServerConnection() {
  const statusBadge = document.getElementById('conn-status');
  try {
    const res = await fetch(`${getApiUrl()}/api/status`);
    if (res.ok) {
      if (statusBadge) {
        statusBadge.textContent = '● সংযুক্ত (8077)';
        statusBadge.style.color = '#4ade80';
        statusBadge.style.backgroundColor = '#1e3a1e';
      }
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    if (statusBadge) {
      statusBadge.textContent = '○ অফলাইন';
      statusBadge.style.color = '#f87171';
      statusBadge.style.backgroundColor = '#450a0a';
    }
  }
}

// ── Export Active Canvas as PNG Blob ──
async function getActiveDocumentBlob() {
  if (!photoshop || !photoshop.app || !photoshop.app.activeDocument) {
    throw new Error('দয়া করে ফটোশপে একটি ছবি ওপেন করুন!');
  }

  const doc = photoshop.app.activeDocument;
  const storage = uxp.storage.localFileSystem;
  const tempFolder = await storage.getTemporaryFolder();
  const tempFile = await tempFolder.createFile(`isbd_export_${Date.now()}.png`, { overwrite: true });

  // Use Photoshop UXP saveAs or export
  await doc.saveAs.png(tempFile, { compression: 6 }, true);
  const arrayBuffer = await tempFile.read({ format: uxp.storage.formats.binary });
  return new Blob([arrayBuffer], { type: 'image/png' });
}

// ── Insert Returned Image as a New Layer in Photoshop ──
async function insertLayerFromBlob(blob, layerName) {
  if (!photoshop || !photoshop.app || !photoshop.app.activeDocument) return;

  const storage = uxp.storage.localFileSystem;
  const tempFolder = await storage.getTemporaryFolder();
  const tempFile = await tempFolder.createFile(`isbd_import_${Date.now()}.png`, { overwrite: true });

  const buffer = await blob.arrayBuffer();
  await tempFile.write(buffer, { format: uxp.storage.formats.binary });

  // Place into active document via batchPlay
  const token = storage.createSessionToken(tempFile);
  await photoshop.core.executeAsModal(async () => {
    await photoshop.action.batchPlay([
      {
        _obj: 'placeEvent',
        ID: 1,
        null: { _path: token, _kind: 'local' },
        freeTransformCenterState: { _enum: 'quadCenterState', _value: 'QCSAverage' },
        offset: { _unit: 'pixelsUnit', horizontal: 0, vertical: 0 }
      }
    ], {});
    if (photoshop.app.activeDocument.activeLayers[0]) {
      photoshop.app.activeDocument.activeLayers[0].name = layerName;
    }
  }, { commandName: `ISBD: ${layerName}` });
}

// ── Helper to convert base64 to Blob ──
function b64toBlob(b64Data, contentType = 'image/png') {
  const byteCharacters = atob(b64Data);
  const byteArrays = [];
  for (let offset = 0; offset < byteCharacters.length; offset += 512) {
    const slice = byteCharacters.slice(offset, offset + 512);
    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }
  return new Blob(byteArrays, { type: contentType });
}

// ── UI Actions ──
document.addEventListener('DOMContentLoaded', () => {
  checkServerConnection();
  setInterval(checkServerConnection, 10000);

  // 1. MediaPipe Retouch
  document.getElementById('btn-mediapipe')?.addEventListener('click', async () => {
    try {
      log('মিডিয়াপাইপ ফেস মেশ ও স্কিন রিটাচিং চলছে...');
      const blob = await getActiveDocumentBlob();
      const formData = new FormData();
      formData.append('image', blob, 'image.png');
      formData.append('dermis_smooth', document.getElementById('dermis-smooth')?.value || 70);
      formData.append('teeth_whiten', document.getElementById('teeth-whiten')?.checked ? 'true' : 'false');
      formData.append('eye_clarity', document.getElementById('eye-clarity')?.checked ? 'true' : 'false');

      const res = await fetch(`${getApiUrl()}/api/mediapipe-retouch`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.detail || 'রিটাচিং ব্যর্থ হয়েছে');

      const outBlob = b64toBlob(data.image_base64);
      await insertLayerFromBlob(outBlob, '✨ [ISBD] MediaPipe 468pt Retouch');
      log(`সফল! ${data.description || 'ফেস মেশ ও স্কিন রিটাচ সম্পন্ন'}`);
    } catch (e) {
      log(`ত্রুটি: ${e.message}`);
    }
  });

  // 2. Alpha Matting Cutout
  document.getElementById('btn-alpha-matting')?.addEventListener('click', async () => {
    try {
      log('PyMatting আলফা হেয়ার কাটআউট প্রসেস হচ্ছে...');
      const blob = await getActiveDocumentBlob();
      const formData = new FormData();
      formData.append('image', blob, 'image.png');

      const res = await fetch(`${getApiUrl()}/api/alpha-matting`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.detail || 'আলফা ম্যাটিং ব্যর্থ');

      const outBlob = b64toBlob(data.image_base64);
      await insertLayerFromBlob(outBlob, '✂️ [ISBD] Alpha Hair Cutout');
      log(`সফল! সাব-পিক্সেল আলফা হেয়ার কাটআউট নতুন লেয়ারে যুক্ত হয়েছে (${data.method || 'Closed-Form'})`);
    } catch (e) {
      log(`ত্রুটি: ${e.message}`);
    }
  });

  // 3. Bézier Vector Pen Tool Path
  document.getElementById('btn-vector-path')?.addEventListener('click', async () => {
    try {
      log('Bézier ভেক্টর পেন-টুল ক্লিপিং পাথ তৈরি হচ্ছে...');
      const blob = await getActiveDocumentBlob();
      const formData = new FormData();
      formData.append('image', blob, 'image.png');

      const res = await fetch(`${getApiUrl()}/api/export-bezier-svg`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.detail || 'ভেক্টর পাথ ব্যর্থ');

      log(`সফল! ${data.points_count || 0}টি পেন-টুল অ্যাঙ্কর পয়েন্ট সহ Bézier SVG পাথ প্রস্তুত (${data.paths_count} পাথ)`);
    } catch (e) {
      log(`ত্রুটি: ${e.message}`);
    }
  });

  // 4. Multi-Layer PSD Exporter
  document.getElementById('btn-export-psd')?.addEventListener('click', async () => {
    try {
      log('নন-ডেস্ট্রাক্টিভ মাল্টি-লেয়ার ফটোশপ (.PSD) তৈরি হচ্ছে...');
      const blob = await getActiveDocumentBlob();
      const formData = new FormData();
      formData.append('image', blob, 'image.png');

      const res = await fetch(`${getApiUrl()}/api/export-psd`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('PSD তৈরি করতে ব্যর্থ');

      const psdBlob = await res.blob();
      log(`সফল! ${Math.round(psdBlob.size / 1024)} KB সাইজের ৩-লেয়ার ফটোশপ .PSD ফাইল প্রস্তুত`);
    } catch (e) {
      log(`ত্রুটি: ${e.message}`);
    }
  });

  // 5. ISBD One-Click Retouch
  document.getElementById('btn-quick-retouch')?.addEventListener('click', async () => {
    try {
      log('ISBD AI স্টুডিও এনহ্যান্সার কল করা হচ্ছে...');
      const blob = await getActiveDocumentBlob();
      const formData = new FormData();
      formData.append('image', blob, 'image.png');

      const res = await fetch(`${getApiUrl()}/api/mediapipe-retouch`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.detail || 'এনহ্যান্স ব্যর্থ');

      const outBlob = b64toBlob(data.image_base64);
      await insertLayerFromBlob(outBlob, '⚡ [ISBD] Studio One-Click Enhanced');
      log('সফল! স্টুডিও এনহ্যান্সড লেয়ার ফটোশপে যোগ হয়েছে');
    } catch (e) {
      log(`ত্রুটি: ${e.message}`);
    }
  });
});
