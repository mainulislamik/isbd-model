/**
 * ISBD Studio AI — Photoshop Host ExtendScript (hostscript.jsx)
 * Supports Adobe Photoshop CC 2018 through CC 2026+
 */

function isDocumentOpen() {
    try {
        return (app.documents.length > 0) ? "TRUE" : "FALSE";
    } catch (e) {
        return "FALSE";
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
        
        // Ensure parent directory exists
        if (!outFile.parent.exists) {
            outFile.parent.create();
        }

        var exportOptions = new ExportOptionsSaveForWeb();
        exportOptions.format = SaveDocumentType.PNG;
        exportOptions.PNG8 = false; // PNG-24 High Quality
        exportOptions.transparency = true;
        exportOptions.interlaced = false;
        exportOptions.includeProfile = true;

        doc.exportDocument(outFile, ExportType.SAVEFORWEB, exportOptions);
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
        
        // Duplicate into target document at the top of layer stack
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
