"""
ISBD v1.00 — Smart Hardware Thermal Monitor & Auto-Throttler v2
- 85°C: throttle training (was 80°C) — less aggressive, more training time
- 72°C: resume (was 68°C) — shorter cooldown waits
- 95°C: emergency halt (hardware protection)
- Adaptive: if cooling takes >5min, lower threshold by5°C
"""
import time
from pathlib import Path


def get_cpu_temp():
    """Read highest CPU core temperature in Celsius."""
    try:
        temps = []
        # Check thermal zones
        for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                t = int(p.read_text().strip()) / 1000.0
                if 20 <= t <= 120:
                    temps.append(t)
            except Exception:
                pass
        # Check coretemp via hwmon
        for p in Path("/sys/class/hwmon").glob("hwmon*/temp*_input"):
            try:
                t = int(p.read_text().strip()) / 1000.0
                if 20 <= t <= 120:
                    temps.append(t)
            except Exception:
                pass
        if temps:
            return max(temps)
    except Exception:
        pass
    return 60.0  # Fallback default safe temp


def auto_cool_if_needed(high_threshold=85.0, target_cool=72.0):
    """
    Thermal Guard v2:
    - Less aggressive thresholds (85/72 vs old 80/68) → more training time
    - Emergency halt at95°C regardless of threshold
    - Adaptive: if cooling takes >5min, tighten threshold by5°C
    """
    temp = get_cpu_temp()

    # Emergency: always halt at95°C+
    if temp >= 95.0:
        print(f"[thermal guard] 🚨 EMERGENCY: {temp:.1f}°C — Halting to protect hardware!", flush=True)
        while True:
            time.sleep(15)
            current = get_cpu_temp()
            if current <= 72.0:
                print(f"[thermal guard] ✅ Emergency resolved: {current:.1f}°C — Resuming.", flush=True)
                break
            else:
                print(f"[thermal guard] 🚨 Emergency cooling... {current:.1f}°C", flush=True)
        return True

    if temp >= high_threshold:
        print(f"[thermal guard] ⚠️ CPU {temp:.1f}°C (limit {high_threshold:.0f}°C) — Cooling...", flush=True)
        t0 = time.time()
        adaptive_target = target_cool
        while True:
            time.sleep(10)
            current = get_cpu_temp()
            elapsed = time.time() - t0
            # Adaptive: if cooling >5min, tighten target
            if elapsed > 300 and adaptive_target > 65.0:
                adaptive_target -= 5.0
                print(f"[thermal guard] Adaptive: lowering target to {adaptive_target:.0f}°C", flush=True)
                t0 = time.time()  # reset timer
            if current <= adaptive_target:
                print(f"[thermal guard] ✅ Cooled to {current:.1f}°C (target {adaptive_target:.0f}°C) — Resuming.", flush=True)
                break
            else:
                if int(elapsed) % 30 == 0:  # log every30s
                    print(f"[thermal guard] Cooling... {current:.1f}°C (target {adaptive_target:.0f}°C)", flush=True)
        return True
    return False


def get_thermal_status():
    """Return thermal info for UI display."""
    temp = get_cpu_temp()
    if temp >= 95:
        return {"temp": temp, "status": "emergency", "emoji": "🚨"}
    elif temp >= 85:
        return {"temp": temp, "status": "hot", "emoji": "🔥"}
    elif temp >= 75:
        return {"temp": temp, "status": "warm", "emoji": "⚠️"}
    else:
        return {"temp": temp, "status": "cool", "emoji": "✅"}
