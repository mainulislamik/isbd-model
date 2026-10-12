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

    // 6. Autonomous Designer Action Observer & Harvest Engine
    var baselineCaptured = false;
    var baselinePath = getTempFilePath('isbd_baseline_' + Date.now() + '.png');

    function captureBaseline(showNotice) {
        var pEsc = baselinePath.replace(/\\/g, '/');
        csInterface.evalScript("captureBaselineSnapshot('" + pEsc + "')", function(res) {
            if (res && res.indexOf('OK') !== -1) {
                baselineCaptured = true;
                if (showNotice) log('✅ নতুন ক্যানভাস বেসলাইন স্ন্যাপশট রেকর্ড করা হয়েছে।');
            }
        });
    }

    setTimeout(function() { captureBaseline(false); }, 1500);

    document.getElementById('btn-reset-baseline').addEventListener('click', function() {
        baselinePath = getTempFilePath('isbd_baseline_' + Date.now() + '.png');
        captureBaseline(true);
    });

    function syncDesignerHarvest() {
        var autoLearn = document.getElementById('check-autolearn').checked;
        if (!autoLearn) {
            log('অটো-লার্নিং বন্ধ থাকায় স্যাম্পল পাঠানো হয়নি।');
            return;
        }

        if (!baselineCaptured || (fs && !fs.existsSync(baselinePath))) {
            captureBaseline(false);
            log('বেসলাইন স্ন্যাপশট প্রস্তুত করা হচ্ছে, অনুগ্রহ করে কিছুক্ষণ পর চেষ্টা করুন।');
            return;
        }

        var currentPath = getTempFilePath('isbd_current_' + Date.now() + '.png');
        var curEsc = currentPath.replace(/\\/g, '/');

        log('ডিজাইনার এডিটিং পেয়ার প্রস্তুত করা হচ্ছে...');
        csInterface.evalScript("captureCurrentSnapshot('" + curEsc + "')", function(res) {
            if (!res || res.indexOf('OK') === -1) {
                log('ক্যাপচার ত্রুটি: ' + res);
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
                var meta = {
                    designer_id: 'designer_' + (os ? os.hostname() : 'win_pc'),
                    doc_name: docName,
                    tools_used: ['retouch', 'frequency_separation', 'clipping'],
                    timestamp: new Date().toISOString()
                };
                fd.append('metadata', JSON.stringify(meta));

                fetch(getHostUrl() + '/api/harvest/designer-sample', { method: 'POST', body: fd })
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        if (data.ok) {
                            log('🎯 এআই ট্রেনিংয়ে নতুন পেয়ার সিঙ্ক হয়েছে! মোট স্যাম্পল: ' + data.total_samples + ' টি');
                            updateHarvestStats();
                            baselinePath = currentPath;
                        } else {
                            log('হার্ভেস্ট নোট: ' + (data.error || 'বাদ দেওয়া হয়েছে'));
                        }
                    })
                    .catch(function(e) {
                        log('হার্ভেস্ট সার্ভার ত্রুটি: ' + e.message);
                    });
            });
        });
    }

    document.getElementById('btn-harvest-sync').addEventListener('click', function() {
        syncDesignerHarvest();
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
});
