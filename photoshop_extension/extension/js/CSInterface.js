/**
 * CSInterface - v9.4.0
 * Adobe Common Extensibility Platform (CEP) Interface Library
 */
function CSInterface() {}

CSInterface.prototype.hostEnvironment = window.__adobe_cep__ ? JSON.parse(window.__adobe_cep__.getHostEnvironment()) : null;

CSInterface.prototype.getHostEnvironment = function() {
    this.hostEnvironment = window.__adobe_cep__ ? JSON.parse(window.__adobe_cep__.getHostEnvironment()) : null;
    return this.hostEnvironment;
};

CSInterface.prototype.closeExtension = function() {
    if (window.__adobe_cep__) window.__adobe_cep__.closeExtension();
};

CSInterface.prototype.getSystemPath = function(pathType) {
    var path = "";
    if (window.__adobe_cep__) {
        path = window.__adobe_cep__.getSystemPath(pathType);
    }
    return path;
};

CSInterface.prototype.evalScript = function(script, callback) {
    if (window.__adobe_cep__) {
        if (callback === null || callback === undefined) {
            callback = function(result) {};
        }
        window.__adobe_cep__.evalScript(script, callback);
    } else {
        console.warn("[CSInterface] evalScript called outside CEP:", script);
        if (callback) callback("EvalScript_Simulation_No_CEP");
    }
};

CSInterface.prototype.openURLInDefaultBrowser = function(url) {
    if (window.cep && window.cep.util) {
        window.cep.util.openURLInDefaultBrowser(url);
    } else if (window.__adobe_cep__) {
        window.__adobe_cep__.invokeSync("openURLInDefaultBrowser", url);
    } else {
        window.open(url, "_blank");
    }
};

CSInterface.prototype.addEventListener = function(type, listener, obj) {
    if (window.__adobe_cep__) window.__adobe_cep__.addEventListener(type, listener, obj);
};

CSInterface.prototype.removeEventListener = function(type, listener, obj) {
    if (window.__adobe_cep__) window.__adobe_cep__.removeEventListener(type, listener, obj);
};

CSInterface.prototype.dispatchEvent = function(event) {
    if (window.__adobe_cep__) window.__adobe_cep__.dispatchEvent(event);
};

CSInterface.prototype.requestOpenExtension = function(extensionId, params) {
    if (window.__adobe_cep__) window.__adobe_cep__.requestOpenExtension(extensionId, params);
};

// SystemPath Constants
SystemPath = {
    USER_DATA: "userData",
    COMMON_FILES: "commonFiles",
    MY_DOCUMENTS: "myDocuments",
    APPLICATION: "application",
    EXTENSION: "extension",
    HOST_APPLICATION: "hostApplication"
};
