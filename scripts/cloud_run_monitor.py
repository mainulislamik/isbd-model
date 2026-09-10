#!/usr/bin/env python3
"""
ISBD v1.00 — Cloud Run Monitor (one-shot)
Watches a specific GitHub Actions cloud-training run.
On success: pulls new weights, restarts ONLY isbd-panel (trainer stays
paused per on-demand rule), then prints a final report.
Usage: python scripts/cloud_run_monitor.py <run_id> [poll_seconds]
"""
import subprocess, sys, time, json, os, urllib.request
from pathlib import Path

REPO = "mainulislamik/isbd-model"
DIR = Path("/home/imon/isbd_model")

def get_token():
    env_file = DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GITHUB_TOKEN", "")

def run_status(run_id, token):
    url = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("User-Agent", "ISBD-Cloud-Monitor")
    req.add_header("Accept", "application/vnd.github.v3+json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode())
            return d.get("status"), d.get("conclusion")
    except Exception as e:
        print(f"[Monitor] API error: {e}", flush=True)
        return None, None

def report_state():
    try:
        h = json.loads((DIR / "checkpoints" / "history.json").read_text())
        steps = h.get("total_steps", "?")
        losses = h.get("losses", [])
        last = f"{losses[-1]:.4f}" if losses else "?"
        best = min(losses) if losses else None
        best = f"{best:.4f}" if best is not None else "?"
        print(f"[Report] total_steps={steps} | last_loss={last} | best_loss={best}", flush=True)
    except Exception as e:
        print(f"[Report] history read error: {e}", flush=True)

def main():
    if len(sys.argv) < 2:
        print("usage: cloud_run_monitor.py <run_id> [poll_seconds]")
        sys.exit(1)
    run_id = sys.argv[1]
    poll = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    token = get_token()
    print(f"[Monitor] Watching cloud run {run_id} (poll {poll}s)...", flush=True)
    report_state()

    t0 = time.time()
    while True:
        status, conclusion = run_status(run_id, token)
        mins = (time.time() - t0) / 60
        print(f"[Monitor] {mins:5.1f} min | status={status} conclusion={conclusion}", flush=True)

        if status == "completed":
            if conclusion == "success":
                print("[Monitor] ✅ Cloud run SUCCEEDED — pulling new weights...", flush=True)
                r = subprocess.run(["git", "pull", "origin", "main"], cwd=DIR,
                                   capture_output=True, text=True)
                print(r.stdout.strip() or r.stderr.strip(), flush=True)
                r2 = subprocess.run(["git", "log", "--oneline", "-3"], cwd=DIR,
                                    capture_output=True, text=True)
                print("[Monitor] New commits:\n" + r2.stdout, flush=True)
                # Restart ONLY the panel so it serves the fresh weights.
                # isbd-trainer stays paused (on-demand rule).
                subprocess.run(["docker", "compose", "restart", "isbd-panel"], cwd=DIR)
                # verify panel is healthy
                for _ in range(10):
                    try:
                        with urllib.request.urlopen("http://localhost:8077/api/status", timeout=5) as resp:
                            if resp.status == 200:
                                print("[Monitor] Panel healthy after restart ✅", flush=True)
                                break
                    except Exception:
                        time.sleep(3)
                report_state()
                print(f"[Monitor] DONE — cloud training {run_id} synced to local panel.", flush=True)
            else:
                print(f"[Monitor] ❌ Cloud run finished with conclusion: {conclusion}", flush=True)
            return
        time.sleep(poll)

if __name__ == "__main__":
    main()
