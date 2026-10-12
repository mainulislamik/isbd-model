#!/usr/bin/env bash
# ISBD v1.00 — Start Remote Trainer Worker on Any Linux/Cloud Machine
set -e

SERVER_URL="${1:-http://127.0.0.1:8077}"
DEVICE="${2:-auto}"
STEPS="${3:-100}"

echo "========================================================"
echo "    ISBD Studio AI — Autonomous Remote Trainer"
echo "========================================================"
echo "Connecting to Server : $SERVER_URL"
echo "Hardware Device      : $DEVICE"
echo "Steps per Cycle      : $STEPS"
echo "========================================================"

# Run with python
python3 -m isbd.trainer_worker --server "$SERVER_URL" --device "$DEVICE" --steps "$STEPS" --cycles 0
