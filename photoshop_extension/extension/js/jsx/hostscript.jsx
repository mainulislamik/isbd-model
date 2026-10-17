/**
 * ISBD Studio AI — Photoshop Host ExtendScript (hostscript.jsx)
 * Supports Adobe Photoshop CC 2018 through CC 2026+
 * Real-time Action Observer & Swarm Telemetry Harvester
 */

function isDocumentOpen() {
    try {
        return (app.documents.length > 0) ? "TRUE" : "FALSE";
    } catch (e) {
        return "FALSE";
    }
}

function _safeEscape(str) {
    if (!str) return "";
    return ("" + str).replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}

function classifyToolDomain(toolName, historyName) {
    var t = (toolName || "").toLowerCase();
    var h = (historyName || "").toLowerCase();
    
    // Retouch: Healing brushes, patch tool, clone stamp, frequency separation
    if (t.indexOf("healing") !== -1 || t.indexOf("patch") !== -1 || t.indexOf("stamp") !== -1 ||
        h.indexOf("healing") !== -1 || h.indexOf("stamp") !== -1 || h.indexOf("skin") !== -1 ||
        h.indexOf("frequency") !== -1 || h.indexOf("high pass") !== -1 || h.indexOf("dust") !== -1) {
        return "retouch";
    }
    
    // Matting: Pen tool, Lasso, Quick Select, Magic Wand, Object Select, Layer Masks
    if (t.indexOf("pen") !== -1 || t.indexOf("select") !== -1 || t.indexOf("lasso") !== -1 ||
        t.indexOf("wand") !== -1 || h.indexOf("mask") !== -1 || h.indexOf("matting") !== -1 ||
        h.indexOf("refine edge") !== -1 || h.indexOf("select subject") !== -1) {
        return "matting";
    }
    
    // Lighting: Dodge, Burn, Sponge, Curves, Levels, Exposure, Brightness/Contrast
    if (t.indexOf("dodge") !== -1 || t.indexOf("burn") !== -1 || t.indexOf("sponge") !== -1 ||
        h.indexOf("curves") !== -1 || h.indexOf("levels") !== -1 || h.indexOf("exposure") !== -1 ||
        h.indexOf("brightness") !== -1 || h.indexOf("shadow") !== -1 || h.indexOf("highlight") !== -1) {
        return "lighting";
    }
    
    // E-Commerce: Crop, White background, Solid color fill, Drop shadow
    if (t.indexOf("crop") !== -1 || h.indexOf("crop") !== -1 || h.indexOf("white") !== -1 ||
        h.indexOf("canvas size") !== -1 || h.indexOf("shadow") !== -1) {
        return "ecommerce";
    }
    
    return "general";
}

function getLiveEditorTelemetry() {
    try {
        if (app.documents.length === 0) {
            return '{"ok":false,"error":"কোনো ডকুমেন্ট নেই"}';
        }
        var doc = app.activeDocument;
        var tool = "";
        try { tool = app.currentTool; } catch(err) { tool = "unknown"; }
        
        var histName = "";
        var histIndex = 0;
        var histTotal = 0;
        try {
            histTotal = doc.historyStates.length;
            var activeState = doc.activeHistoryState;
            histName = activeState.name;
            for (var i = 0; i < histTotal; i++) {
                if (doc.historyStates[i] === activeState) {
                    histIndex = i;
                    break;
                }
            }
        } catch(err) {}
        
        var layerName = "";
        var layerOpacity = 100;
        var layerBlendMode = "normal";
        var isAdjustment = false;
        try {
            if (doc.activeLayer) {
                layerName = doc.activeLayer.name;
                layerOpacity = Math.round(doc.activeLayer.opacity);
                layerBlendMode = doc.activeLayer.blendMode.toString().replace("BlendMode.", "").toLowerCase();
                try {
                    isAdjustment = (doc.activeLayer.kind === LayerKind.CURVES || 
                                    doc.activeLayer.kind === LayerKind.LEVELS || 
                                    doc.activeLayer.kind === LayerKind.HUESATURATION ||
                                    doc.activeLayer.kind === LayerKind.BRIGHTNESSCONTRAST);
                } catch(kErr) {}
            }
        } catch(err) {}
        
        var hasSelection = false;
        var selBoundsStr = "null";
        try {
            var sb = doc.selection.bounds;
            hasSelection = true;
            selBoundsStr = "[" + Math.round(sb[0].as("px")) + "," + Math.round(sb[1].as("px")) + "," + Math.round(sb[2].as("px")) + "," + Math.round(sb[3].as("px")) + "]";
        } catch(err) {
            hasSelection = false;
            selBoundsStr = "null";
        }
        
        var domain = classifyToolDomain(tool, histName);
        
        var json = '{' +
            '"ok":true,' +
            '"tool":"' + _safeEscape(tool) + '",' +
            '"domain":"' + _safeEscape(domain) + '",' +
            '"historyName":"' + _safeEscape(histName) + '",' +
            '"historyIndex":' + histIndex + ',' +
            '"historyTotal":' + histTotal + ',' +
            '"layerName":"' + _safeEscape(layerName) + '",' +
            '"layerOpacity":' + layerOpacity + ',' +
            '"layerBlendMode":"' + _safeEscape(layerBlendMode) + '",' +
            '"isAdjustment":' + (isAdjustment ? "true" : "false") + ',' +
            '"hasSelection":' + (hasSelection ? "true" : "false") + ',' +
            '"selectionBounds":' + selBoundsStr + ',' +
            '"docWidth":' + Math.round(doc.width.as("px")) + ',' +
            '"docHeight":' + Math.round(doc.height.as("px")) + ',' +
            '"docName":"' + _safeEscape(doc.name) + '"' +
        '}';
        return json;
    } catch (e) {
        return '{"ok":false,"error":"' + _safeEscape(e.toString()) + '"}';
    }
}

function getActiveDocDetails() {
    try {
        if (app.documents.length === 0) return JSON.stringify({ ok: false, error: "কোনো ডকুমেন্ট ওপেন নেই" });
        var doc = app.activeDocument;
        return JSON.stringify({
            ok: true,
            name: doc.name,
            width: doc.width.as("px"),
            height: doc.height.as("px"),
            resolution: doc.resolution,
            mode: doc.mode.toString()
        });
    } catch (e) {
        return JSON.stringify({ ok: false, error: e.toString() });
    }
}

function exportActiveDocPNG(pngPath) {
    try {
        if (app.documents.length === 0) return "ERROR: কোনো সক্রিয় ডকুমেন্ট নেই";
        var doc = app.activeDocument;
        var outFile = new File(pngPath);
        
        if (!outFile.parent.exists) {
            outFile.parent.create();
        }

        var exportOptions = new ExportOptionsSaveForWeb();
        exportOptions.format = SaveDocumentType.PNG;
        exportOptions.PNG8 = false;
        exportOptions.transparency = true;
        exportOptions.interlaced = false;
        exportOptions.includeProfile = true;

        doc.exportDocument(outFile, ExportType.SAVEFORWEB, exportOptions);
        return "OK";
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}

function captureScaledSnapshot(outPath, maxDim) {
    try {
        if (app.documents.length === 0) return "ERROR: কোনো সক্রিয় ডকুমেন্ট নেই";
        var doc = app.activeDocument;
        var outFile = new File(outPath);
        if (!outFile.parent.exists) outFile.parent.create();

        var dupDoc = doc.duplicate("ISBD_Telemetry_Temp", true);
        
        if (maxDim && maxDim > 0) {
            var w = dupDoc.width.as("px");
            var h = dupDoc.height.as("px");
            if (w > maxDim || h > maxDim) {
                if (w >= h) {
                    dupDoc.resizeImage(UnitValue(maxDim, "px"), null, null, ResampleMethod.BICUBIC);
                } else {
                    dupDoc.resizeImage(null, UnitValue(maxDim, "px"), null, ResampleMethod.BICUBIC);
                }
            }
        }
        
        var exportOptions = new ExportOptionsSaveForWeb();
        exportOptions.format = SaveDocumentType.PNG;
        exportOptions.PNG8 = false;
        exportOptions.transparency = true;
        exportOptions.interlaced = false;
        exportOptions.includeProfile = false;

        dupDoc.exportDocument(outFile, ExportType.SAVEFORWEB, exportOptions);
        dupDoc.close(SaveOptions.DONOTSAVECHANGES);
        return "OK";
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}

function captureROIPatchSnapshot(outPath, x1, y1, x2, y2, patchSize) {
    try {
        if (app.documents.length === 0) return "ERROR: কোনো সক্রিয় ডকুমেন্ট নেই";
        var doc = app.activeDocument;
        var outFile = new File(outPath);
        if (!outFile.parent.exists) outFile.parent.create();

        var dupDoc = doc.duplicate("ISBD_ROI_Temp", true);
        
        // Crop duplicate to ROI
        var cropBounds = [UnitValue(x1, "px"), UnitValue(y1, "px"), UnitValue(x2, "px"), UnitValue(y2, "px")];
        dupDoc.crop(cropBounds);
        
        if (patchSize && patchSize > 0) {
            dupDoc.resizeImage(UnitValue(patchSize, "px"), UnitValue(patchSize, "px"), null, ResampleMethod.BICUBIC);
        }

        var exportOptions = new ExportOptionsSaveForWeb();
        exportOptions.format = SaveDocumentType.PNG;
        exportOptions.PNG8 = false;
        exportOptions.transparency = true;
        exportOptions.interlaced = false;

        dupDoc.exportDocument(outFile, ExportType.SAVEFORWEB, exportOptions);
        dupDoc.close(SaveOptions.DONOTSAVECHANGES);
        return "OK";
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}

function importPNGAsLayer(pngPath, layerName) {
    try {
        if (app.documents.length === 0) return "ERROR: কোনো সক্রিয় ডকুমেন্ট নেই";
        var targetDoc = app.activeDocument;
        var tempFile = new File(pngPath);
        if (!tempFile.exists) return "ERROR: টেম্প ফাইল পাওয়া যায়নি: " + pngPath;

        var openedDoc = app.open(tempFile);
        var srcLayer = openedDoc.activeLayer;
        
        srcLayer.duplicate(targetDoc, ElementPlacement.PLACEATBEGINNING);
        openedDoc.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = targetDoc;
        if (targetDoc.activeLayer) {
            targetDoc.activeLayer.name = layerName;
        }
        return "OK";
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}

function showPhotoshopAlert(msg) {
    try {
        alert(msg);
        return "OK";
    } catch (e) {
        return "ERROR";
    }
}

function captureBaselineSnapshot(outPath) {
    try {
        if (app.documents.length === 0) return "ERROR: কোনো সক্রিয় ডকুমেন্ট নেই";
        return captureScaledSnapshot(outPath, 1024);
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}

function captureCurrentSnapshot(outPath) {
    try {
        if (app.documents.length === 0) return "ERROR: কোনো সক্রিয় ডকুমেন্ট নেই";
        return captureScaledSnapshot(outPath, 1024);
    } catch (e) {
        return "ERROR: " + e.toString();
    }
}
