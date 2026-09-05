"""
ISBD v1.00 — Continuous Trainer (never-stop learning)
Runs training in endless rounds of N steps. Auto-restarts on crash (with backoff).
Evals unseen quality each round; nightly cron commits + pushes.
Use: .venv/bin/python isbd/continuous.py [--round-steps 2000] [--batch 8]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
TRAIN = ROOT / "isbd" / "train.py"
EVAL = ROOT / "isbd" / "eval.py"


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
    ap.add_argument("--max-restarts", type=int, default=60)
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

        # eval unseen quality every round
        try:
            ev = subprocess.run([str(PY), str(EVAL)], cwd=str(ROOT),
                                 capture_output=True, text=True, timeout=600)
            print(f"[eval] {ev.stdout.strip()}", flush=True)
        except Exception as e:
            print(f"[eval] skipped: {e}", flush=True)

        # Auto-commit & push every round (or every ~5,000 steps) so checkpoints are always safe in Git
        try:
            subprocess.run(["git", "add", "checkpoints/best.pt", "checkpoints/history.json", "data/real_pairs.npz", "data/real_pairs_log.json"], cwd=str(ROOT), capture_output=True)
            commit_res = subprocess.run(["git", "commit", "-m", f"Auto-checkpoint update: step {after} (Continuous self-learning)"], cwd=str(ROOT), capture_output=True, text=True)
            if commit_res.returncode == 0:
                push_res = subprocess.run(["git", "push", "origin", "main"], cwd=str(ROOT), capture_output=True, text=True, timeout=45)
                if push_res.returncode == 0:
                    print(f"[git] Synced checkpoint & training data to GitHub main at step {after} ✓", flush=True)
        except Exception as e:
            print(f"[git] auto-push skipped: {e}", flush=True)

        if after == before:  # no progress -> crash or lock held; backoff
            restarts += 1
            wait = min(30 * restarts, 300)
            print(f"[continuous] no progress (attempt {restarts}) — sleeping {wait}s", flush=True)
            time.sleep(wait)
        else:
            restarts = 0  # progress resets the counter
            time.sleep(2)  # tiny breather between rounds

    print("[continuous] max restarts reached — giving up (systemd will restart service)", flush=True)


if __name__ == "__main__":
    main()
