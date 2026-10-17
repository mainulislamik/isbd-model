/**
 * ISBD Studio AI — Photoshop CEP Extension Client (main.js)
 */

var csInterface = new CSInterface();
var fs = null;
var path = null;
var os = null;

if (typeof require !== 'undefined') {
    try {
        fs = require('fs');
        path = require('path');
        os = require('os');
    } catch (e) {
        console.warn('Node.js modules not accessible:', e);
    }
}

function log(msg) {
    var con = document.getElementById('log-console');
    if (con) {
        var time = new Date().toTimeString().split(' ')[0];
        con.innerHTML += '<br>[' + time + '] ' + msg;
        con.scrollTop = con.scrollHeight;
    }
    console.log('[ISBD CEP]', msg);
}

function clearLog() {
    var con = document.getElementById('log-console');
    if (con) con.innerHTML = '[ISBD CEP] কনসোল ক্লিয়ার করা হয়েছে।';
}

function getHostUrl() {
    var input = document.getElementById('host-url');
    var url = (input ? input.value : 'http://127.0.0.1:8077').trim();
    return url.replace(/\/+$/, '');
}

function checkServerPing() {
    var badge = document.getElementById('conn-badge');
    var pingText = document.getElementById('ping-text');
    var url = getHostUrl();

    fetch(url + '/api/status', { method: 'GET' })
        .then(function(res) {
            if (res.ok) {
                if (badge) {
                    badge.className = 'conn-pill';
                    badge.innerHTML = '● সংযুক্ত (' + (url.split(':').pop() || '8077') + ')';
                }
                if (pingText) {
                    pingText.innerHTML = 'Online';
                    pingText.style.color = '#10b981';
                }
            } else {
                throw new Error('HTTP ' + res.status);
            }
        })
        .catch(function(err) {
            if (badge) {
                badge.className = 'conn-pill offline';
                badge.innerHTML = '○ অফলাইন';
            }
            if (pingText) {
                pingText.innerHTML = 'Offline';
                pingText.style.color = '#ef4444';
            }
        });
}

function getTempFilePath(filename) {
    if (os && path) {
        return path.join(os.tmpdir(), filename);
    }
    return 'C:/Windows/Temp/' + filename;
}

// ── Generic Workflow: Export Canvas -> Call AI -> Import as New Layer ──
function processWithAI(endpoint, formData, layerTitle, callback) {
    var tempIn = getTempFilePath('isbd_input_' + Date.now() + '.png');
    // Escape backslashes for ExtendScript
    var tempInEsc = tempIn.replace(/\\/g, '/');

    log('ফটোশপ থেকে অ্যাক্টিভ ক্যানভাস এক্সপোর্ট হচ্ছে...');

    csInterface.evalScript("exportActiveDocPNG('" + tempInEsc + "')", function(res) {
        if (!res || res.indexOf('OK') === -1) {
            log('ত্রুটি: ' + (res || 'ক্যানভাস এক্সপোর্ট ব্যর্থ হয়েছে। ডকুমেন্ট খোলা আছে কি?'));
            return;
        }

        log('এআই সার্ভারে ইমেজ পাঠানো হচ্ছে (' + endpoint + ')...');

        var blob = null;
        if (fs) {
            try {
                var buffer = fs.readFileSync(tempIn);
                blob = new Blob([buffer], { type: 'image/png' });
            } catch (e) {
                log('ফাইল রিড ত্রুটি: ' + e.message);
                return;
            }
        }

        var reqData = formData || new FormData();
        if (blob) {
            reqData.append('image', blob, 'image.png');
        }

        fetch(getHostUrl() + endpoint, {
            method: 'POST',
            body: reqData
        })
        .then(function(response) {
            if (!response.ok) throw new Error('সার্ভার কোড: ' + response.status);
            return response.json();
        })
        .then(function(data) {
            if (!data.ok) throw new Error(data.detail || 'এআই প্রসেসিং ব্যর্থ');

            if (data.image_base64) {
                var tempOut = getTempFilePath('isbd_output_' + Date.now() + '.png');
                var tempOutEsc = tempOut.replace(/\\/g, '/');

                if (fs) {
                    var outBuffer = Buffer.from(data.image_base64, 'base64');
                    fs.writeFileSync(tempOut, outBuffer);

                    csInterface.evalScript("importPNGAsLayer('" + tempOutEsc + "', '" + layerTitle + "')", function(importRes) {
                        if (importRes && importRes.indexOf('OK') !== -1) {
                            log('✅ সফল! নতুন লেয়ার হিসেবে যুক্ত হয়েছে: ' + layerTitle);
                        } else {
                            log('লেয়ার ইনপুট ত্রুটি: ' + importRes);
                        }
                        if (callback) callback(data);
                    });
                } else {
                    log('✅ সম্পন্ন (Node.js FS ছাড়া সরাসরি ব্রাউজারে বেস64 পাওয়া গেছে)');
                }
            } else {
                log('✅ প্রসেসিং সম্পন্ন: ' + (data.description || 'সফল'));
                if (callback) callback(data);
            }
        })
        .catch(function(err) {
            log('❌ নেটওয়ার্ক/সার্ভার সমস্যা: ' + err.message);
        });
    });
}

// ── Event Handlers ──
document.addEventListener('DOMContentLoaded', function() {
    checkServerPing();
    setInterval(checkServerPing, 10000);

    // 1. MediaPipe Retouch
    document.getElementById('btn-mediapipe').addEventListener('click', function() {
        var smooth = document.getElementById('range-smooth').value;
        var teeth = document.getElementById('check-teeth').checked ? 'true' : 'false';
        var eye = document.getElementById('check-eye').checked ? 'true' : 'false';

        var fd = new FormData();
        fd.append('dermis_smooth', smooth);
        fd.append('teeth_whiten', teeth);
        fd.append('eye_clarity', eye);

        log('MediaPipe 468-pt ফেস রিটাচিং রিকোয়েস্ট শুরু...');
        processWithAI('/api/mediapipe-retouch', fd, '✨ [ISBD] MediaPipe 468pt Retouch');
    });

    // 2. One-Click Enhance
    document.getElementById('btn-quick-retouch').addEventListener('click', function() {
        var fd = new FormData();
        fd.append('dermis_smooth', '75');
        fd.append('teeth_whiten', 'true');
        fd.append('eye_clarity', 'true');

        log('স্টুডিও ওয়ান-ক্লিক এনহ্যান্স রিকোয়েস্ট শুরু...');
        processWithAI('/api/mediapipe-retouch', fd, '⚡ [ISBD] Studio Enhanced');
    });

    // 3. PyMatting Cutout
    document.getElementById('btn-matting').addEventListener('click', function() {
        log('PyMatting সাব-পিক্সেল আলফা হেয়ার কাটআউট শুরু...');
        processWithAI('/api/alpha-matting', new FormData(), '✂️ [ISBD] Alpha Hair Cutout');
    });

    // 4. Bézier Vector Pen Tool Path
    document.getElementById('btn-vector').addEventListener('click', function() {
        log('Bézier ভেক্টর পাথ জেনারেট হচ্ছে...');
        processWithAI('/api/export-bezier-svg', new FormData(), '', function(data) {
            csInterface.evalScript("showPhotoshopAlert('✅ Bézier ভেক্টর পাথ সফলভাবে তৈরি হয়েছে! মোট অ্যাঙ্কর পয়েন্ট: " + (data.points_count || 0) + "')");
        });
    });

    // 5. Multi-Layer PSD Exporter
    document.getElementById('btn-psd').addEventListener('click', function() {
        var tempIn = getTempFilePath('isbd_input_psd_' + Date.now() + '.png');
        var tempInEsc = tempIn.replace(/\\/g, '/');

        log('PSD এক্সপোর্টের জন্য ক্যানভাস ইমেজ পাঠানো হচ্ছে...');

        csInterface.evalScript("exportActiveDocPNG('" + tempInEsc + "')", function(res) {
            if (!res || res.indexOf('OK') === -1) {
                log('ত্রুটি: ' + res);
                return;
            }

            var fd = new FormData();
            if (fs) {
                var buffer = fs.readFileSync(tempIn);
                fd.append('image', new Blob([buffer], { type: 'image/png' }), 'image.png');
            }

            fetch(getHostUrl() + '/api/export-psd', { method: 'POST', body: fd })
                .then(function(r) { return r.blob(); })
                .then(function(psdBlob) {
                    var outPsd = getTempFilePath('ISBD_MultiLayer_' + Date.now() + '.psd');
                    if (fs) {
                        psdBlob.arrayBuffer().then(function(ab) {
                            fs.writeFileSync(outPsd, Buffer.from(ab));
                            log('✅ ৫-লেয়ার বিশিষ্ট নন-ডেস্ট্রাক্টিভ .PSD প্রস্তুত: ' + outPsd);
                            csInterface.evalScript("showPhotoshopAlert('✅ ৫-লেয়ার ফটোশপ .PSD প্রস্তুত:\\n" + outPsd.replace(/\\/g, '/') + "')");
                        });
                    }
                })
                .catch(function(e) { log('PSD এক্সপোর্ট ত্রুটি: ' + e.message); });
        });
    });

    var lastAIProposalPath = null;
    var lastAITaskDomain = "general";

    // ── Open-Source SOTA Tools ──
    document.getElementById('btn-birefnet').addEventListener('click', function() {
        log('BiRefNet আল্ট্রা হেয়ার ও ট্রান্সপারেন্ট কাটআউট শুরু...');
        lastAITaskDomain = "matting";
        processWithAI('/api/ai/birefnet-matting', new FormData(), '✂️ [SOTA] BiRefNet Hair Cutout', function(d) {
            if (d && d.image_base64) {
                lastAIProposalPath = getTempFilePath('isbd_ai_prop_' + Date.now() + '.png');
                if (fs) fs.writeFileSync(lastAIProposalPath, Buffer.from(d.image_base64, 'base64'));
            }
        });
    });

    document.getElementById('btn-lama-inpaint').addEventListener('click', function() {
        log('LaMa স্মার্ট ইনপেইন্টার (দাগ ও স্ক্র্যাচ ক্লিন) শুরু...');
        lastAITaskDomain = "inpaint";
        processWithAI('/api/ai/smart-inpaint', new FormData(), '🩹 [SOTA] LaMa Inpaint Clean', function(d) {
            if (d && d.image_base64) {
                lastAIProposalPath = getTempFilePath('isbd_ai_prop_' + Date.now() + '.png');
                if (fs) fs.writeFileSync(lastAIProposalPath, Buffer.from(d.image_base64, 'base64'));
            }
        });
    });

    document.getElementById('btn-realesrgan').addEventListener('click', function() {
        log('Real-ESRGAN 4K টেক্সচার ও সুপার-রেজোলিউশন শুরু...');
        lastAITaskDomain = "superres";
        var fd = new FormData();
        fd.append('scale', '2');
        processWithAI('/api/ai/realesrgan-superres', fd, '💎 [SOTA] Real-ESRGAN 4K Texture', function(d) {
            if (d && d.image_base64) {
                lastAIProposalPath = getTempFilePath('isbd_ai_prop_' + Date.now() + '.png');
                if (fs) fs.writeFileSync(lastAIProposalPath, Buffer.from(d.image_base64, 'base64'));
            }
        });
    });

    document.getElementById('btn-codeformer').addEventListener('click', function() {
        log('CodeFormer ফেস রিস্টোরার ও স্কিন পোরস রিকভারি শুরু...');
        lastAITaskDomain = "retouch";
        var fd = new FormData();
        fd.append('smooth_strength', '0.7');
        processWithAI('/api/ai/codeformer-beauty', fd, '👤 [SOTA] CodeFormer Face Restore', function(d) {
            if (d && d.image_base64) {
                lastAIProposalPath = getTempFilePath('isbd_ai_prop_' + Date.now() + '.png');
                if (fs) fs.writeFileSync(lastAIProposalPath, Buffer.from(d.image_base64, 'base64'));
            }
        });
    });

    // ── 1-Click Human Correction Teaching ──
    document.getElementById('btn-teach-correction').addEventListener('click', function() {
        var currentPath = getTempFilePath('isbd_human_final_' + Date.now() + '.png');
        var curEsc = currentPath.replace(/\\/g, '/');

        log('👨🎨 আপনার ম্যানুয়াল কারেকশন ক্যাপচার করা হচ্ছে...');
        csInterface.evalScript("captureCurrentSnapshot('" + curEsc + "')", function(res) {
            if (!res || res.indexOf('OK') === -1) {
                log('ক্যাপচার ত্রুটি: ' + res);
                return;
            }

            var fd = new FormData();
            var origP = baselinePath;
            var propP = lastAIProposalPath || baselinePath;

            if (fs) {
                try {
                    if (fs.existsSync(origP)) fd.append('original', new Blob([fs.readFileSync(origP)], { type: 'image/png' }), 'orig.png');
                    if (fs.existsSync(propP)) fd.append('proposal', new Blob([fs.readFileSync(propP)], { type: 'image/png' }), 'prop.png');
                    if (fs.existsSync(currentPath)) fd.append('human_final', new Blob([fs.readFileSync(currentPath)], { type: 'image/png' }), 'final.png');
                } catch(e) {
                    log('ফাইল এরর: ' + e.message);
                    return;
                }
            }

            fd.append('task_domain', lastAITaskDomain);
            fd.append('designer_id', 'designer_' + (os ? os.hostname() : 'win_pc'));

            log('🎯 হিউম্যান কারেকশন নিউরাল অ্যাডাপ্টারে ট্রেইন করা হচ্ছে (' + lastAITaskDomain + ')...');
            fetch(getHostUrl() + '/api/ai/teach-correction', { method: 'POST', body: fd })
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.ok) {
                        var imp = data.stats && data.stats.improvement_pct ? (' (উন্নতি: +' + data.stats.improvement_pct + '%)') : '';
                        log('🎉 অসাধারণ! এআই আপনার এডিট থেকে শিখেছে' + imp);
                        csInterface.evalScript("showPhotoshopAlert('✅ হিউম্যান লার্নিং সফল!\\nমডেলটি আপনার এডিট অনুকরণ করে অ্যাডাপ্টার আপডেট করেছে" + imp + "')");
                        updateHarvestStats();
                    } else {
                        log('লার্নিং নোট: ' + (data.error || 'ব্যর্থ'));
                    }
                })
                .catch(function(e) {
                    log('লার্নিং এরর: ' + e.message);
                });
        });
    });

    // ── 6. Autonomous Live Action Observer, ROI Micro-Patcher & Swarm Telemetry Harvester ──
    var baselineCaptured = false;
    var baselinePath = getTempFilePath('isbd_baseline_' + Date.now() + '.png');
    var lastHistoryIndex = -1;
    var validStrokes = 0;
    var undoneStrokes = 0;
    var currentSessionTools = {};
    var currentPrimaryDomain = 'general';
    var lastTelemetryData = null;
    var lastDocName = '';

    var toolNamesMap = {
        "spotHealingBrushTool": "স্পট হিলিং ব্রাশ",
        "healingBrushTool": "হিলিং ব্রাশ",
        "patchTool": "প্যাচ টুল",
        "cloneStampTool": "ক্লোন স্ট্যাম্প",
        "patternStampTool": "প্যাটার্ন স্ট্যাম্প",
        "paintbrushTool": "পেইন্ট ব্রাশ",
        "pencilTool": "পেন্সিল টুল",
        "penTool": "পেন টুল (পাথ/মাস্ক)",
        "quickSelectTool": "কুইক সিলেকশন",
        "magicWandTool": "ম্যাজিক ওয়ান্ড",
        "lassoTool": "ল্যাসো টুল",
        "polySelTool": "পলিগোনাল ল্যাসো",
        "magneticLassoTool": "ম্যাগনেটিক ল্যাসো",
        "dodgeTool": "ডজ টুল (হাইলাইট)",
        "burnTool": "বার্ন টুল (শ্যাডো)",
        "spongeTool": "স্পঞ্জ টুল",
        "blurTool": "ব্লার টুল",
        "sharpenTool": "শার্পেন টুল",
        "smudgeTool": "স্মাজ টুল",
        "cropTool": "ক্রপ টুল",
        "eraserTool": "ইরেজার",
        "historyBrushTool": "হিস্ট্রি ব্রাশ"
    };

    var domainNamesMap = {
        "retouch": "স্কিন ও ফেস রিটাচ",
        "matting": "হেয়ার/ফার ক্লিপিং ও মাস্ক",
        "lighting": "লাইটিং ও টোন ব্যালেন্স",
        "ecommerce": "ই-কমার্স প্রডাক্ট ও শ্যাডো",
        "general": "সাধারণ এডিটিং"
    };

    function captureBaseline(showNotice) {
        var pEsc = baselinePath.replace(/\\/g, '/');
        csInterface.evalScript("captureBaselineSnapshot('" + pEsc + "')", function(res) {
            if (res && res.indexOf('OK') !== -1) {
                baselineCaptured = true;
                validStrokes = 0;
                undoneStrokes = 0;
                currentSessionTools = {};
                var strokeEl = document.getElementById('telemetry-strokes');
                if (strokeEl) strokeEl.innerText = '0';
                var undoEl = document.getElementById('telemetry-undos');
                if (undoEl) undoEl.innerText = '0';
                if (showNotice) log('✅ নতুন ক্যানভাস বেসলাইন রেকর্ড করা হয়েছে।');
            }
        });
    }

    setTimeout(function() { captureBaseline(false); }, 1500);

    document.getElementById('btn-reset-baseline').addEventListener('click', function() {
        baselinePath = getTempFilePath('isbd_baseline_' + Date.now() + '.png');
        captureBaseline(true);
    });

    // ── Live Telemetry HUD & Action Polling Engine (Every 2.5 seconds) ──
    function pollEditorTelemetry() {
        csInterface.evalScript("getLiveEditorTelemetry()", function(res) {
            if (!res) return;
            try {
                var t = JSON.parse(res);
                if (!t.ok) return;

                lastTelemetryData = t;

                // Document switched? Reset session baseline
                if (lastDocName && lastDocName !== t.docName) {
                    lastDocName = t.docName;
                    baselinePath = getTempFilePath('isbd_baseline_' + Date.now() + '.png');
                    captureBaseline(false);
                    return;
                }
                lastDocName = t.docName;

                // Detect History Changes: Valid forward stroke vs Undo (Ctrl+Z)
                if (lastHistoryIndex !== -1 && t.historyIndex !== undefined) {
                    if (t.historyIndex < lastHistoryIndex) {
                        // Undo detected! User discarded previous work
                        var diffUndo = lastHistoryIndex - t.historyIndex;
                        undoneStrokes += diffUndo;
                        var undoEl = document.getElementById('telemetry-undos');
                        if (undoEl) undoEl.innerText = undoneStrokes;
                    } else if (t.historyIndex > lastHistoryIndex) {
                        // Valid forward editing stroke/action
                        var diffStroke = t.historyIndex - lastHistoryIndex;
                        validStrokes += diffStroke;
                        var strokeEl = document.getElementById('telemetry-strokes');
                        if (strokeEl) strokeEl.innerText = validStrokes;

                        // Track tool usage frequency
                        if (t.tool) {
                            currentSessionTools[t.tool] = (currentSessionTools[t.tool] || 0) + 1;
                        }
                        if (t.domain && t.domain !== 'general') {
                            currentPrimaryDomain = t.domain;
                        }
                    }
                }
                lastHistoryIndex = (t.historyIndex !== undefined) ? t.historyIndex : lastHistoryIndex;

                // Update UI elements
                var toolEl = document.getElementById('telemetry-tool');
                if (toolEl) {
                    var friendlyTool = toolNamesMap[t.tool] || t.tool || 'অ্যাক্টিভ';
                    toolEl.innerText = friendlyTool;
                }

                var domEl = document.getElementById('telemetry-domain');
                if (domEl) {
                    domEl.innerText = domainNamesMap[t.domain] || t.domain;
                }

                var layerEl = document.getElementById('telemetry-layer-info');
                if (layerEl) {
                    var adjText = t.isAdjustment ? ' [Adj]' : '';
                    layerEl.innerText = (t.layerBlendMode || 'Normal') + ' (' + (t.layerOpacity || 100) + '%)' + adjText;
                }

                // Passive auto-save sync: If auto-learn enabled and editor committed 5+ quality strokes
                var autoLearn = document.getElementById('check-autolearn').checked;
                if (autoLearn && validStrokes >= 8) {
                    syncDesignerHarvest(true); // background silent harvest
                }

            } catch (err) {}
        });
    }

    setInterval(pollEditorTelemetry, 2500);

    // ── Smart ROI Cropper & Telemetry Harvester ──
    function syncDesignerHarvest(isAuto) {
        var autoLearn = document.getElementById('check-autolearn').checked;
        if (!autoLearn && isAuto) return;

        if (!baselineCaptured || (fs && !fs.existsSync(baselinePath))) {
            captureBaseline(false);
            if (!isAuto) log('বেসলাইন স্ন্যাপশট প্রস্তুত করা হচ্ছে, অনুগ্রহ করে কিছুক্ষণ পর চেষ্টা করুন।');
            return;
        }

        var editorTagInput = document.getElementById('input-editor-id');
        var editorId = (editorTagInput && editorTagInput.value.trim()) ? editorTagInput.value.trim() : ('designer_' + (os ? os.hostname() : 'ps_seat'));

        var roiCropEnabled = document.getElementById('check-roi-crop') ? document.getElementById('check-roi-crop').checked : true;
        var hasRoiSelection = lastTelemetryData && lastTelemetryData.hasSelection && lastTelemetryData.selectionBounds;

        var currentPath = getTempFilePath('isbd_current_' + Date.now() + '.png');
        var curEsc = currentPath.replace(/\\/g, '/');

        if (!isAuto) log('ডিজাইনার এডিটিং পেয়ার প্রস্তুত করা হচ্ছে...');

        // If user has an active selection and ROI crop is checked, harvest exact 512x512 micro-patch
        var captureScript = "captureCurrentSnapshot('" + curEsc + "')";
        var roiCropped = false;

        if (roiCropEnabled && hasRoiSelection) {
            var b = lastTelemetryData.selectionBounds; // [x1, y1, x2, y2]
            captureScript = "captureROIPatchSnapshot('" + curEsc + "', " + b[0] + ", " + b[1] + ", " + b[2] + ", " + b[3] + ", 512)";
            roiCropped = true;
        }

        csInterface.evalScript(captureScript, function(res) {
            if (!res || res.indexOf('OK') === -1) {
                if (!isAuto) log('ক্যাপচার ত্রুটি: ' + res);
                return;
            }

            csInterface.evalScript("getActiveDocDetails()", function(docDetailsStr) {
                var docName = 'untitled.psd';
                try {
                    var docObj = JSON.parse(docDetailsStr);
                    if (docObj.ok) docName = docObj.name;
                } catch(e) {}

                var fd = new FormData();
                if (fs) {
                    var bBuf = fs.readFileSync(baselinePath);
                    var aBuf = fs.readFileSync(currentPath);
                    fd.append('before', new Blob([bBuf], { type: 'image/png' }), 'before.png');
                    fd.append('after', new Blob([aBuf], { type: 'image/png' }), 'after.png');
                }

                var toolsList = Object.keys(currentSessionTools);
                if (toolsList.length === 0) toolsList = [lastTelemetryData ? lastTelemetryData.tool : 'retouch'];

                var meta = {
                    designer_id: editorId,
                    doc_name: docName,
                    domain: currentPrimaryDomain,
                    tools_used: toolsList,
                    stroke_count: validStrokes,
                    undos_filtered: undoneStrokes,
                    roi_cropped: roiCropped,
                    layer_info: lastTelemetryData ? {
                        name: lastTelemetryData.layerName,
                        opacity: lastTelemetryData.layerOpacity,
                        blend_mode: lastTelemetryData.layerBlendMode,
                        is_adjustment: lastTelemetryData.isAdjustment
                    } : {},
                    timestamp: new Date().toISOString()
                };
                fd.append('metadata', JSON.stringify(meta));

                fetch(getHostUrl() + '/api/harvest/designer-sample', { method: 'POST', body: fd })
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data.ok) {
                            var domainBengali = domainNamesMap[meta.domain] || meta.domain;
                            log('🎯 ' + editorId + ' এর কাজ সিঙ্ক হয়েছে! [' + domainBengali + ' | স্ট্রোক: ' + validStrokes + 'টি, Undo বাদ: ' + undoneStrokes + 'টি] মোট ডাটাবেজ: ' + data.total_samples + ' টি');
                            updateHarvestStats();
                            baselinePath = currentPath;
                            validStrokes = 0;
                            undoneStrokes = 0;
                            currentSessionTools = {};
                            var sEl = document.getElementById('telemetry-strokes');
                            if (sEl) sEl.innerText = '0';
                            var uEl = document.getElementById('telemetry-undos');
                            if (uEl) uEl.innerText = '0';
                        } else {
                            if (!isAuto) log('হার্ভেস্ট নোট: ' + (data.error || 'বাদ দেওয়া হয়েছে'));
                        }
                    })
                    .catch(function(e) {
                        if (!isAuto) log('হার্ভেস্ট সার্ভার ত্রুটি: ' + e.message);
                    });
            });
        });
    }

    document.getElementById('btn-harvest-sync').addEventListener('click', function() {
        syncDesignerHarvest(false);
    });

    function updateHarvestStats() {
        fetch(getHostUrl() + '/api/harvest/stats')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.ok) {
                    var el = document.getElementById('harvest-count');
                    if (el) el.innerText = (data.today_samples || 0) + ' টি';
                }
            })
            .catch(function() {});
    }

    updateHarvestStats();
    setInterval(updateHarvestStats, 60000);

    // ── 7. Silent Background Auto-Updater Engine ──
    function checkAndApplyAutoUpdate(isManual) {
        var banner = document.getElementById('update-banner');
        var hostUrl = getHostUrl();

        if (typeof require === 'undefined') {
            if (isManual && banner) {
                banner.style.display = 'block';
                banner.style.color = '#f59e0b';
                banner.innerText = 'Node.js রানটাইম অফলাইন — ব্রাউজার প্রিভিউ মোড';
                setTimeout(function() { banner.style.display = 'none'; }, 3000);
            }
            return;
        }

        try {
            var fs = require('fs');
            var path = require('path');
            var extPath = '';
            if (typeof csInterface !== 'undefined' && typeof SystemPath !== 'undefined') {
                extPath = csInterface.getSystemPath(SystemPath.EXTENSION);
            }
            if (!extPath) {
                extPath = typeof __dirname !== 'undefined' ? __dirname : '';
            }
            if (!extPath) return;

            var versionFile = path.join(extPath, 'version.json');
            var localInfo = { version: "1.0.0", bundle_hash: "" };
            if (fs.existsSync(versionFile)) {
                try {
                    localInfo = JSON.parse(fs.readFileSync(versionFile, 'utf8'));
                } catch(e) {}
            }

            if (isManual && banner) {
                banner.style.display = 'block';
                banner.style.color = '#3b82f6';
                banner.innerText = '🔄 সার্ভারে লাইভ আপডেট খোঁজা হচ্ছে...';
            }

            fetch(hostUrl + '/api/extension/version', { method: 'GET', cache: 'no-cache' })
                .then(function(res) {
                    if (!res.ok) throw new Error('HTTP ' + res.status);
                    return res.json();
                })
                .then(function(srv) {
                    if (!srv.ok || !srv.bundle_hash) return;

                    // Already up to date
                    if (localInfo.bundle_hash === srv.bundle_hash) {
                        if (isManual && banner) {
                            banner.style.display = 'block';
                            banner.style.color = '#10b981';
                            banner.innerText = '✓ প্লাগইন সম্পূর্ণ আপ-টু-ডেট (v' + (srv.version || '1.2.0') + ')';
                            setTimeout(function() { banner.style.display = 'none'; }, 3500);
                        }
                        return;
                    }

                    // Update found on server!
                    if (banner) {
                        banner.style.display = 'block';
                        banner.style.color = '#3b82f6';
                        banner.innerText = '⚡ সার্ভার থেকে নতুন ফিচার আপডেট হচ্ছে... (' + srv.bundle_hash + ')';
                    }

                    fetch(hostUrl + '/api/extension/bundle', { method: 'GET', cache: 'no-cache' })
                        .then(function(bRes) { return bRes.json(); })
                        .then(function(bundle) {
                            if (!bundle.ok || !bundle.files) return;

                            var files = bundle.files;
                            var fileKeys = Object.keys(files);
                            var updatedCount = 0;

                            fileKeys.forEach(function(relPath) {
                                try {
                                    var destPath = path.join(extPath, relPath);
                                    var dir = path.dirname(destPath);
                                    if (!fs.existsSync(dir)) {
                                        fs.mkdirSync(dir, { recursive: true });
                                    }
                                    var buf = Buffer.from(files[relPath], 'base64');
                                    fs.writeFileSync(destPath, buf);
                                    updatedCount++;
                                } catch(fErr) {
                                    console.error('[AutoUpdater] Failed writing ' + relPath, fErr);
                                }
                            });

                            var newVersionData = {
                                version: bundle.version || srv.version || "1.2.0",
                                bundle_hash: bundle.bundle_hash,
                                updated_at: new Date().toISOString(),
                                files_count: updatedCount
                            };
                            fs.writeFileSync(versionFile, JSON.stringify(newVersionData, null, 2), 'utf8');

                            if (banner) {
                                banner.style.color = '#10b981';
                                banner.innerText = '✨ আপডেট সফল! প্লাগইন স্বয়ংক্রিয় রিলোড হচ্ছে...';
                            }
                            log('🚀 ফটোশপ এক্সটেনশন স্বয়ংক্রিয়ভাবে v' + newVersionData.version + '-এ আপডেট হয়েছে!');

                            setTimeout(function() {
                                window.location.reload();
                            }, 800);
                        })
                        .catch(function(bErr) {
                            console.error('[AutoUpdater] Bundle download failed:', bErr);
                        });
                })
                .catch(function(err) {
                    if (isManual && banner) {
                        banner.style.display = 'block';
                        banner.style.color = '#ef4444';
                        banner.innerText = 'সার্ভার সংযোগ ব্যর্থ বা অফলাইন';
                        setTimeout(function() { banner.style.display = 'none'; }, 3500);
                    }
                    console.log('[AutoUpdater] Auto-update ping ignored:', err.message);
                });
        } catch(err) {
            console.error('[AutoUpdater] Error in updater:', err);
        }
    }

    var btnSync = document.getElementById('btn-sync-update');
    if (btnSync) {
        btnSync.addEventListener('click', function() {
            checkAndApplyAutoUpdate(true);
        });
    }

    // Silent check on panel open
    setTimeout(function() {
        checkAndApplyAutoUpdate(false);
    }, 2500);
});
