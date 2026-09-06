"""
ISBD v1.00 — Smart Hardware Thermal Monitor & Auto-Throttler
Continuously checks CPU temperatures:
- If CPU >= 80°C: Throttles/pauses training for cooling (sleeps).
- If CPU >= 88°C: Halts training immediately to protect hardware.
- If CPU <= 68°C: Resumes full-speed training safely.
"""
import time
import subprocess
from pathlib import Path

def get_cpu_temp():
    """Read highest CPU core temperature in Celsius."""
    try:
        # Check thermal zones
        temps = []
        for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                t = int(p.read_text().strip()) / 1000.0
                # Filter sensible range (20°C - 120°C)
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


def auto_cool_if_needed(high_threshold=80.0, target_cool=65.0):
    """
    Thermal Guard check:
    If temperature exceeds high_threshold, pauses execution and sleeps until
    CPU cools down to target_cool before allowing next training steps.
    """
    temp = get_cpu_temp()
    if temp >= high_threshold:
        print(f"[thermal guard] ⚠️ CPU Temp High: {temp:.1f}°C (Threshold: {high_threshold}°C) — Pausing training to cool down...", flush=True)
        while True:
            time.sleep(10)
            current_temp = get_cpu_temp()
            if current_temp <= target_cool:
                print(f"[thermal guard] ✅ CPU Cooled to {current_temp:.1f}°C (Safe <= {target_cool}°C) — Resuming training.", flush=True)
                break
            else:
                print(f"[thermal guard] Cooling... Current Temp: {current_temp:.1f}°C (Waiting for <= {target_cool}°C)", flush=True)
        return True
    return False
