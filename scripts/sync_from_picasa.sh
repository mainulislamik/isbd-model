#!/bin/bash
# Sync AI Checkpoints and Training Data from Picasa VPS to Local PC
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

# Check if Picasa VPS is reachable
if ssh -q -o ConnectTimeout=5 -o BatchMode=yes picasa exit; then
    # Sync checkpoints (weights, history, state)
    rsync -avz --update picasa:/opt/isbd_model/checkpoints/ /home/imon/isbd_model/checkpoints/ >/dev/null 2>&1
    # Sync data (real pairs, ft state)
    rsync -avz --update picasa:/opt/isbd_model/data/ /home/imon/isbd_model/data/ >/dev/null 2>&1
fi
