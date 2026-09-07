#!/usr/bin/env python3
import time
import subprocess
import os
import json
import urllib.request
from pathlib import Path

REPO = "mainulislamik/isbd-model"
RUN_URL = f"https://api.github.com/repos/{REPO}/actions/runs?per_page=1"
DIR = Path("/home/imon/isbd_model")

def get_token():
    env_file = DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GITHUB_TOKEN", "")

def check_status(token):
    req = urllib.request.Request(RUN_URL)
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("User-Agent", "ISBD-Cloud-Watcher")
    req.add_header("Accept", "application/vnd.github.v3+json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if data.get("workflow_runs"):
                run = data["workflow_runs"][0]
                return run.get("id"), run.get("status"), run.get("conclusion")
    except Exception as e:
        print(f"[Watcher] API check error: {e}")
    return None, None, None

def main():
    token = get_token()
    print(f"[Watcher] Started monitoring GitHub Cloud Trainer for {REPO}...")
    
    last_synced_id = None
    
    while True:
        run_id, status, conclusion = check_status(token)
        print(f"[Watcher] Current Run #{run_id}: status={status}, conclusion={conclusion}")
        
        if status == "completed":
            if conclusion == "success":
                print(f"[Watcher] Run #{run_id} completed successfully! Pulling latest trained weights...")
                try:
                    res = subprocess.run(["git", "pull", "origin", "main"], cwd=DIR, capture_output=True, text=True)
                    print(f"[Watcher] Git pull output:\n{res.stdout}\n{res.stderr}")
                    
                    # Restart docker container to load new weights
                    subprocess.run(["docker", "compose", "restart", "isbd-panel", "isbd-trainer"], cwd=DIR, capture_output=True)
                    print(f"[Watcher] Successfully synced and restarted services!")
                except Exception as ex:
                    print(f"[Watcher] Error pulling/restarting: {ex}")
                break
            else:
                print(f"[Watcher] Run #{run_id} completed with status: {conclusion}")
                break
        
        # Wait 30 seconds before next check
        time.sleep(30)

if __name__ == "__main__":
    main()
