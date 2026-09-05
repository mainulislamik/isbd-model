"""
ISBD v1.00 — Hermes AI Model Providers & Settings Manager
Bridges ISBD Studio testing lab with Hermes-connected vision/language models
(Antigravity, Gemini, Claude, OmniRoute, b.ai, Ollama, OpenRouter).
"""
import io
import os
import json
import yaml
import base64
import urllib.request
from pathlib import Path
from PIL import Image

HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"
ISBD_MODELS_CACHE = Path(__file__).resolve().parent.parent / "data" / "connected_models.json"

# Built-in Default Image Models Available in Hermes
BUILTIN_IMAGE_MODELS = [
    {
        "id": "isbd_v1",
        "name": "ISBD v1.00 Local (Default)",
        "provider": "Local Engine",
        "type": "restoration",
        "badge": "24/7 Self-Trained",
        "vision_support": True
    },
    {
        "id": "antigravity/gemini-3.7-flash-high",
        "name": "Gemini 3.7 Flash High Vision",
        "provider": "Antigravity (Free)",
        "type": "vlm_vision",
        "badge": "High Resolution",
        "vision_support": True
    },
    {
        "id": "antigravity/claude-sonnet-4-6-low",
        "name": "Claude 3.7 / 3.5 Sonnet Vision",
        "provider": "Antigravity (Free)",
        "type": "vlm_vision",
        "badge": "Pro Vision",
        "vision_support": True
    },
    {
        "id": "deepseek-v4-flash-vision-exp",
        "name": "DeepSeek v4 Flash Vision",
        "provider": "b.ai Free",
        "type": "vlm_vision",
        "badge": "Fast Vision",
        "vision_support": True
    },
    {
        "id": "omniroute/auto",
        "name": "OmniRoute Keyless Vision Agent",
        "provider": "OmniRoute LAN",
        "type": "router",
        "badge": "LAN Multi-Model",
        "vision_support": True
    }
]


def load_hermes_providers():
    """Read Hermes config.yaml and extract all registered custom and system providers."""
    providers = []
    if not HERMES_CONFIG.exists():
        return {"providers": [], "custom_providers": []}

    try:
        with open(HERMES_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        custom_list = cfg.get("custom_providers", [])
        for cp in custom_list:
            name = cp.get("name", "Custom Provider")
            base_url = cp.get("base_url", "")
            key_env = cp.get("key_env", "")
            api_key = os.environ.get(key_env, "") if key_env else cp.get("api_key", "")
            
            # Mask API key for security (show only last 4 chars)
            masked_key = (api_key[:4] + "..." + api_key[-4:]) if (api_key and len(api_key) > 8) else ("Configured" if api_key else "Keyless / Local")
            
            models = list(cp.get("models", {}).keys()) if isinstance(cp.get("models"), dict) else []
            if cp.get("model") and cp.get("model") not in models:
                models.append(cp.get("model"))

            providers.append({
                "name": name,
                "base_url": base_url,
                "key_env": key_env,
                "masked_key": masked_key,
                "api_mode": cp.get("api_mode", "chat_completions"),
                "models": models,
                "active_model": cp.get("model", models[0] if models else "")
            })

        return {
            "current_default_provider": cfg.get("model", {}).get("provider", "custom:omniroute-free"),
            "current_default_model": cfg.get("model", {}).get("model", "auto"),
            "providers": providers
        }
    except Exception as e:
        return {"error": str(e), "providers": []}


def save_hermes_provider_config(provider_name: str, base_url: str = None, api_key: str = None, default_model: str = None):
    """Update or add a provider in ~/.hermes/config.yaml."""
    if not HERMES_CONFIG.exists():
        return {"ok": False, "error": "Hermes config not found"}

    try:
        with open(HERMES_CONFIG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        custom_list = cfg.get("custom_providers", [])
        matched = False
        for cp in custom_list:
            if cp.get("name") == provider_name:
                matched = True
                if base_url:
                    cp["base_url"] = base_url
                if default_model:
                    cp["model"] = default_model
                    if "models" in cp and isinstance(cp["models"], dict):
                        cp["models"][default_model] = {}
                if api_key:
                    if cp.get("key_env"):
                        os.environ[cp["key_env"]] = api_key
                    else:
                        cp["api_key"] = api_key
                break

        if not matched and base_url:
            dm = default_model if default_model else "default"
            new_cp = {
                "name": provider_name,
                "base_url": base_url,
                "model": dm,
                "api_mode": "chat_completions",
                "models": {dm: {}}
            }
            if api_key:
                new_cp["api_key"] = api_key
            custom_list.append(new_cp)

        cfg["custom_providers"] = custom_list
        with open(HERMES_CONFIG, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        return {"ok": True, "message": f"Provider '{provider_name}' সফলভাবে আপডেট করা হয়েছে!"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_available_test_models():
    """Return all models ready for image restoration and vision editing."""
    models = list(BUILTIN_IMAGE_MODELS)
    hermes_data = load_hermes_providers()
    
    # Add discovered vision models from custom providers
    for prov in hermes_data.get("providers", []):
        pname = prov["name"]
        for m in prov.get("models", []):
            if any(k in m.lower() for k in ["vision", "vl", "flash", "gemini", "claude", "qwen", "hy3", "auto"]):
                # Avoid duplicates
                if not any(x["id"] == f"{pname}:{m}" for x in models):
                    models.append({
                        "id": f"{pname}:{m}",
                        "name": f"{m} ({pname})",
                        "provider": pname,
                        "type": "hermes_bridge",
                        "badge": "Hermes Connected",
                        "vision_support": True
                    })
    return models
