#!/bin/bash
# Sync AI Checkpoints and Training Data between Local PC & Picasa VPS, then push to GitHub
LOCKFILE="/tmp/isbd_picasa_sync.lock"

# Avoid concurrent runs
if [ -e "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE" 2>/dev/null)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        exit 0
    fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# Ensure HTTP/1.1 for reliable fast GitHub push
git -C /home/imon/isbd_model config http.version HTTP/1.1
git -C /home/imon/isbd_model config http.postBuffer 524288000

# Check if Picasa VPS is reachable
if ssh -q -o ConnectTimeout=5 -o BatchMode=yes picasa exit; then
    # 1. PUSH real human pairs, high-res patches, service buckets, and composite assets to Picasa VPS for 24/7 training
    rsync -avz --update /home/imon/isbd_model/data/real_pairs.npz picasa:/opt/isbd_model/data/ >/dev/null 2>&1
    rsync -avz --update /home/imon/isbd_model/data/real_pairs_log.json picasa:/opt/isbd_model/data/ >/dev/null 2>&1
    rsync -avz --update /home/imon/isbd_model/data/real_patches_256.npz picasa:/opt/isbd_model/data/ >/dev/null 2>&1
    rsync -avz --update /home/imon/isbd_model/data/self_pool.npz picasa:/opt/isbd_model/data/ >/dev/null 2>&1
    rsync -avz --update /home/imon/isbd_model/data/curriculum_state.json picasa:/opt/isbd_model/data/ >/dev/null 2>&1

    if [ -d "/home/imon/isbd_model/data/buckets" ]; then
        rsync -avz --update /home/imon/isbd_model/data/buckets/ picasa:/opt/isbd_model/data/buckets/ >/dev/null 2>&1
    fi
    if [ -d "/home/imon/isbd_model/data/composite_assets" ]; then
        rsync -avz --update /home/imon/isbd_model/data/composite_assets/ picasa:/opt/isbd_model/data/composite_assets/ >/dev/null 2>&1
    fi
    if [ -d "/home/imon/isbd_model/data/ref_assets" ]; then
        rsync -avz --update /home/imon/isbd_model/data/ref_assets/ picasa:/opt/isbd_model/data/ref_assets/ >/dev/null 2>&1
    fi
    if [ -d "/home/imon/isbd_model/data/harvested_samples" ]; then
        rsync -avz --update /home/imon/isbd_model/data/harvested_samples/ picasa:/opt/isbd_model/data/harvested_samples/ >/dev/null 2>&1
    fi
    if [ -d "/home/imon/isbd_model/checkpoints/adapters" ]; then
        rsync -avz --update /home/imon/isbd_model/checkpoints/adapters/ picasa:/opt/isbd_model/checkpoints/adapters/ >/dev/null 2>&1
    fi

    # 2. PUSH latest codebase (isbd/) to Picasa VPS
    rsync -avz --update --exclude '__pycache__' /home/imon/isbd_model/isbd/ picasa:/opt/isbd_model/isbd/ >/dev/null 2>&1

    # 3. PULL trained weights, history, and checkpoints from Picasa VPS to Local PC
    rsync -avz --update picasa:/opt/isbd_model/checkpoints/ /home/imon/isbd_model/checkpoints/ >/dev/null 2>&1

    # 4. If new checkpoints or data arrived, commit and push to GitHub
    if ! git -C /home/imon/isbd_model diff --quiet checkpoints/ data/ 2>/dev/null; then
        STEPS=$(python3 -c "import json; print(json.load(open('/home/imon/isbd_model/checkpoints/history.json')).get('total_steps', 'latest'))" 2>/dev/null || echo "latest")
        git -C /home/imon/isbd_model add checkpoints/ data/ >/dev/null 2>&1
        git -C /home/imon/isbd_model commit -m "Auto-checkpoint: Step ${STEPS} (Picasa VPS 24/7 Trainer)" >/dev/null 2>&1
        git -C /home/imon/isbd_model push origin main >/dev/null 2>&1
    fi
fi
