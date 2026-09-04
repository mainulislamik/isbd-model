"""
ISBD v1.00 — Continuous Trainer (never-stop learning)
Runs training in endless rounds of N steps. Auto-restarts on crash (with backoff).
Use: .venv/bin/python isbd/continuous.py [--round-steps 2000] [--batch 8]
Run under: nohup / systemd / cron — the loop keeps the model learning forever.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
TRAIN = ROOT / "isbd" / "train.py"


def read_step():
    try:
        import json
        h = json.loads((ROOT / "checkpoints" / "history.json").read_text())
        return h.get("total_steps", 0)
    except Exception:
        return 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-restarts", type=int, default=50)
    args = ap.parse_args()

    restarts = 0
    print(f"[continuous] ISBD never-stop learning | round {args.round_steps} steps", flush=True)
    while restarts < args.max_restarts:
        before = read_step()
        t0 = time.time()
        r = subprocess.run(
            [str(PY), str(TRAIN), "--steps", str(args.round_steps), "--batch", str(args.batch)],
            cwd=str(ROOT),
        )
        after = read_step()
        dt = time.time() - t0
        print(f"[continuous] round done: step {before} -> {after} ({dt:.0f}s, rc={r.returncode})", flush=True)

        if after == before:  # no progress -> crash or lock held; backoff
            restarts += 1
            wait = min(30 * restarts, 300)
            print(f"[continuous] no progress (attempt {restarts}) — sleeping {wait}s", flush=True)
            time.sleep(wait)
        else:
            restarts = 0  # progress resets the counter
            time.sleep(2)  # tiny breather between rounds


if __name__ == "__main__":
    main()
