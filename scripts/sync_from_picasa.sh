#!/bin/bash
# Sync AI Checkpoints and Training Data from Picasa VPS to Local PC & Push to GitHub
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
    # Sync checkpoints (weights, history, state)
    rsync -avz --update picasa:/opt/isbd_model/checkpoints/ /home/imon/isbd_model/checkpoints/ >/dev/null 2>&1
    # Sync data (real pairs, ft state)
    rsync -avz --update picasa:/opt/isbd_model/data/ /home/imon/isbd_model/data/ >/dev/null 2>&1

    # If new checkpoints or data arrived, commit and push to GitHub
    if ! git -C /home/imon/isbd_model diff --quiet checkpoints/ data/ 2>/dev/null; then
        STEPS=$(python3 -c "import json; print(json.load(open('/home/imon/isbd_model/checkpoints/history.json')).get('total_steps', 'latest'))" 2>/dev/null || echo "latest")
        git -C /home/imon/isbd_model add checkpoints/ data/ >/dev/null 2>&1
        git -C /home/imon/isbd_model commit -m "Auto-checkpoint: Step ${STEPS} (Picasa VPS 24/7 Trainer)" >/dev/null 2>&1
        git -C /home/imon/isbd_model push origin main >/dev/null 2>&1
    fi
fi
