"""
ISBD v1.00 — Continuous Trainer v2 (never-stop learning)
Upgrades:
- Disk auto-cleanup (old checkpoints, sample images)
- Passes model + accum-steps flags to train.py
- Better round reporting with thermal info
"""
import subprocess
import sys
import time
import json
from pathlib import Path
import os
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
# Docker-compatible python resolver
PY = Path((ROOT / ".venv" / "bin" / "python")) if (ROOT / ".venv" / "bin" / "python").exists() else Path(shutil.which("python") or shutil.which("python3") or sys.executable)
TRAIN = ROOT / "isbd" / "train.py"
EVAL = ROOT / "isbd" / "eval.py"


def read_step():
    try:
        h = json.loads((ROOT / "checkpoints" / "history.json").read_text())
        return h.get("total_steps", 0)
    except Exception:
        return 0


def cleanup_old_files():
    """Remove old sample images and checkpoint archives to save disk."""
    # Clean old sample grids (>7 days)
    samples = ROOT / "samples"
    if samples.exists():
        cutoff = time.time() - 7 * 86400
        for f in samples.glob("step_*.png"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception:
                pass

    # Clean old checkpoint backups (>30 days)
    ckpt = ROOT / "checkpoints"
    if ckpt.exists():
        cutoff = time.time() - 30 * 86400
        for f in ckpt.glob("best_*.pt"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception:
                pass


def get_disk_usage():
    """Return disk usage info for logging."""
    try:
        st = os.statvfs(str(ROOT))
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        total_gb = (st.f_blocks * st.f_frsize) / (1024**3)
        return f"{free_gb:.1f}GB free / {total_gb:.0f}GB"
    except Exception:
        return "unknown"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-restarts", type=int, default=60)
    ap.add_argument("--model", choices=["tiny", "small"], default="small",
                    help="Model: tiny (117K) or small (500K)")
    ap.add_argument("--accum-steps", type=int, default=2,
                    help="Gradient accumulation steps")
    args = ap.parse_args()

    restarts = 0
    print(f"[continuous] ISBD v1.00 never-stop | model={args.model} | round={args.round_steps} steps | accum={args.accum_steps} | disk={get_disk_usage()}", flush=True)

    while restarts < args.max_restarts:
        before = read_step()
        t0 = time.time()

        r = subprocess.run(
            [str(PY), str(TRAIN),
             "--steps", str(args.round_steps),
             "--batch", str(args.batch),
             "--model", args.model,
             "--accum-steps", str(args.accum_steps),
             "--resume"],  # ALWAYS resume — without this flag every round
                            # restarted from scratch (cost 2 full cloud runs)
            cwd=str(ROOT),
        )

        after = read_step()
        dt = time.time() - t0
        steps_done = after - before
        speed = steps_done / dt if dt > 0 else 0
        print(f"[continuous] round: {before}→{after} ({steps_done} steps, {dt:.0f}s, {speed:.1f} steps/s, rc={r.returncode})", flush=True)

        # eval unseen quality every round
        try:
            ev = subprocess.run([str(PY), str(EVAL)], cwd=str(ROOT),
                                capture_output=True, text=True, timeout=600)
            print(f"[eval] {ev.stdout.strip()}", flush=True)
        except Exception as e:
            print(f"[eval] skipped: {e}", flush=True)

        # Auto-commit & push every round
        # NOTE: samples/ is gitignored — do NOT `git add` it (that made the final
        # sync step exit 1 and silently dropped 6 hours of training before).
        # Also: a runner has no default git identity — set one or commit fails.
        try:
            subprocess.run(["git", "config", "user.email", "cloud-runner@users.noreply.github.com"], cwd=str(ROOT), capture_output=True)
            subprocess.run(["git", "config", "user.name", "ISBD Cloud Runner"], cwd=str(ROOT), capture_output=True)
            add_res = subprocess.run(
                ["git", "add", "checkpoints/best.pt", "checkpoints/cloud_state.pt",
                 "checkpoints/history.json",
                 "data/real_pairs.npz", "data/real_pairs_log.json"],
                cwd=str(ROOT), capture_output=True, text=True)
            if add_res.returncode != 0:
                print(f"[git] add failed: {add_res.stderr.strip()[:200]}", flush=True)
            commit_res = subprocess.run(
                ["git", "commit", "-m", f"Auto-checkpoint: step {after} ({args.model}, {steps_done} steps/round)"],
                cwd=str(ROOT), capture_output=True, text=True)
            if commit_res.returncode == 0:
                # sync with remote first (another run may have pushed) then push
                pull_res = subprocess.run(["git", "pull", "--rebase", "origin", "main"],
                                         cwd=str(ROOT), capture_output=True, text=True, timeout=60)
                if pull_res.returncode != 0:
                    subprocess.run(["git", "rebase", "--abort"], cwd=str(ROOT), capture_output=True)
                    print(f"[git] pull --rebase failed (push skipped this round): {pull_res.stderr.strip()[:200]}", flush=True)
                else:
                    push_res = subprocess.run(["git", "push", "origin", "main"],
                                              cwd=str(ROOT), capture_output=True, text=True, timeout=60)
                    if push_res.returncode == 0:
                        print(f"[git] Synced to GitHub ✓ step {after} | {get_disk_usage()}", flush=True)
                    else:
                        print(f"[git] PUSH FAILED: {push_res.stderr.strip()[:300]}", flush=True)
            else:
                print(f"[git] commit failed: {commit_res.stderr.strip()[:300]}", flush=True)
        except Exception as e:
            print(f"[git] auto-push skipped: {e}", flush=True)

        # ── Thermal Guard between rounds ──
        try:
            from isbd.thermal_guard import auto_cool_if_needed, get_cpu_temp
            auto_cool_if_needed(high_threshold=85.0, target_cool=72.0)
        except Exception:
            pass

        # ── Disk cleanup every10 rounds ──
        if restarts % 10 == 0:
            cleanup_old_files()

        if after == before:  # no progress -> crash or lock held; backoff
            restarts += 1
            wait = min(30 * restarts, 300)
            print(f"[continuous] no progress (attempt {restarts}) — sleeping {wait}s", flush=True)
            time.sleep(wait)
        else:
            restarts = 0  # progress resets the counter
            time.sleep(2)  # tiny breather between rounds

    print("[continuous] max restarts reached — giving up", flush=True)


if __name__ == "__main__":
    main()
