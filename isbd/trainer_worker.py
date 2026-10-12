"""
ISBD v1.00 — Standalone Remote Trainer Worker
Run this script on ANY machine (High-End GPU Rig, Laptop, Cloud Server).
It autonomously synchronizes checkpoints and designer dataset from the master ISBD server,
trains using that machine's native hardware, and pushes model improvements back!
"""

import os
import sys
import time
import json
import zipfile
import argparse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isbd.portable_trainer import run_training_cycle, detect_hardware_profile


def main():
    parser = argparse.ArgumentParser(description="ISBD Autonomous Remote Trainer Worker")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8077",
                        help="ISBD Studio Server Base URL (e.g. http://192.168.0.100:8077)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Hardware device (auto, cuda, cpu, mps, xpu)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Steps to train per cycle")
    parser.add_argument("--cycles", type=int, default=1,
                        help="Number of training cycles (0 for continuous loop)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Interval seconds between cycles")
    args = parser.parse_args()

    server = args.server.rstrip("/")
    profile = detect_hardware_profile(args.device)

    print("=" * 65)
    print("      ISBD v1.00 — Autonomous Remote Trainer Worker")
    print("=" * 65)
    print(f"[*] Target Master Server : {server}")
    print(f"[*] Detected Hardware    : {profile['device_name']}")
    print(f"[*] Selected Device      : {profile['device']}")
    print(f"[*] Batch / Accumulation : {profile['batch_size']} / {profile['accum_steps']}")
    print(f"[*] Precision (AMP)      : {'Enabled (FP16)' if profile['use_amp'] else 'Standard (FP32)'}")
    print("=" * 65)

    # 1. Ping server
    try:
        r = requests.get(f"{server}/api/status", timeout=10)
        status = r.json()
        print(f"[+] Connected to ISBD Master Server! Server Step: {status.get('live', {}).get('step', '—')}")
    except Exception as e:
        print(f"[-] Cannot connect to ISBD server at {server}: {e}")
        print("[-] Please ensure the server is running and accessible over LAN/WAN.")
        sys.exit(1)

    cycle_count = 0
    while True:
        cycle_count += 1
        print(f"\n[>>> Cycle #{cycle_count}] Starting training run of {args.steps} steps...")

        # 2. Sync latest dataset from master
        try:
            ds_res = requests.get(f"{server}/api/harvest/dataset.zip", timeout=30)
            if ds_res.status_code == 200 and len(ds_res.content) > 100:
                cache_zip = ROOT / "data" / "remote_dataset.zip"
                cache_zip.parent.mkdir(parents=True, exist_ok=True)
                cache_zip.write_bytes(ds_res.content)
                with zipfile.ZipFile(cache_zip, "r") as zf:
                    extract_to = ROOT / "data" / "harvested_samples"
                    extract_to.mkdir(parents=True, exist_ok=True)
                    zf.extractall(extract_to)
                print("[+] Successfully synchronized designer dataset from master server.")
        except Exception as e:
            print(f"[!] Dataset sync note: {e} (continuing with synthetic curriculum)")

        # 3. Execute training cycle
        try:
            result = run_training_cycle(
                steps=args.steps,
                device_override=args.device,
                reporter_cb=lambda step, loss: print(f"    --> Step: {step} | Loss: {loss:.4f}")
            )
            print(f"[+] Cycle #{cycle_count} Completed: Step {result['start_step']} -> {result['final_step']} | Loss: {result['final_loss']} | Speed: {result['speed_it_s']} it/s")

            # 4. Sync updated checkpoint back to server
            last_pt = ROOT / "checkpoints" / "last.pt"
            hist_json = ROOT / "checkpoints" / "history.json"
            if last_pt.exists() and hist_json.exists():
                files = {
                    "checkpoint": open(last_pt, "rb"),
                    "history": open(hist_json, "rb")
                }
                data = {
                    "worker_name": f"Remote_{profile['device_name']}",
                    "final_step": result["final_step"],
                    "final_loss": result["final_loss"]
                }
                sync_r = requests.post(f"{server}/api/trainer/sync-checkpoint", files=files, data=data, timeout=30)
                if sync_r.status_code == 200:
                    print("[+] Checkpoint and metrics successfully pushed back to ISBD Server!")
                else:
                    print(f"[-] Checkpoint push returned status {sync_r.status_code}: {sync_r.text}")
        except Exception as e:
            print(f"[-] Training cycle error: {e}")

        if args.cycles > 0 and cycle_count >= args.cycles:
            print(f"\n[✓] Completed {cycle_count} cycles. Worker finished.")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
