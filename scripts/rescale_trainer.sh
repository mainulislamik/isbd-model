#!/usr/bin/env bash
# ==============================================================================
# ISBD Trainer Dynamic Resource Scaler (Picasa VPS)
# Rules:
# - Saturday (6) and Sunday (7): Full VPS resources (cpus=7.0, mem=5.5g)
# - Monday (1) to Friday (5): Safe background resources (cpus=2.0, mem=2.5g)
# ==============================================================================

set -euo pipefail

TARGET_MODE="${1:-auto}"
CONTAINER_NAME="isbd-trainer"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] [rescale_trainer] Container ${CONTAINER_NAME} is not running. Skipping."
    exit 0
fi

# Determine day of week in UTC (1=Monday, ..., 6=Saturday, 7=Sunday)
DAY_OF_WEEK=$(date -u +%u)
DAY_NAME=$(date -u +%A)

MODE=""
if [ "$TARGET_MODE" = "weekend" ]; then
    MODE="weekend"
elif [ "$TARGET_MODE" = "weekday" ]; then
    MODE="weekday"
elif [ "$TARGET_MODE" = "auto" ]; then
    if [ "$DAY_OF_WEEK" -eq 6 ] || [ "$DAY_OF_WEEK" -eq 7 ]; then
        MODE="weekend"
    else
        MODE="weekday"
    fi
else
    echo "Usage: $0 [auto|weekend|weekday]"
    exit 1
fi

if [ "$MODE" = "weekend" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] [rescale_trainer] 🚀 Activating WEEKEND FULL RESOURCE MODE (${DAY_NAME}): cpus=7.0, mem=5.5g"
    docker update --cpus 7.0 --memory 5.5g --memory-swap 6.5g "${CONTAINER_NAME}"
else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] [rescale_trainer] 🏢 Activating WEEKDAY NORMAL RESOURCE MODE (${DAY_NAME}): cpus=2.0, mem=2.5g"
    docker update --cpus 2.0 --memory 2.5g --memory-swap 3.5g "${CONTAINER_NAME}"
fi
