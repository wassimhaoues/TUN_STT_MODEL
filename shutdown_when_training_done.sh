#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="shutdown_monitor.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Searching for Whisper training process..."

PID="$(pgrep -f "python.*training/train_whisper_small.py" | head -n 1 || true)"

if [[ -z "$PID" ]]; then
  log "No training process found. Not shutting down."
  exit 1
fi

log "Found training PID: $PID"
log "Process command:"
ps -p "$PID" -o pid,etime,%cpu,%mem,cmd | tee -a "$LOG_FILE"

log "Requesting sudo now so shutdown can run later..."
sudo -v

# Keep sudo alive while training runs
while true; do
  sudo -n true 2>/dev/null || exit 1
  sleep 60
done &
SUDO_KEEPALIVE_PID=$!

cleanup() {
  kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
}
trap cleanup EXIT

log "Monitoring training process. Laptop will shut down when PID $PID exits."

while kill -0 "$PID" 2>/dev/null; do
  GPU_INFO="$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits | head -n 1 || true)"
  log "Training still running. GPU: $GPU_INFO"
  sleep 60
done

log "Training process ended."
log "Syncing disk..."
sync

log "Shutting down laptop now."
sudo shutdown -h now
