#!/usr/bin/env bash
# Pull the restoration training project code from the remote GPU server
# back to this (local) machine. Run it ON YOUR LOCAL MACHINE:
#   bash transfer.sh              # pulls into ./restoration next to this script
set -euo pipefail

# ------------------------- EDIT ME -------------------------
SSH_USER="root"
SSH_HOST="202.215.217.129"   # <-- server IP   ($PUBLIC_IPADDR on the server)
SSH_PORT="60950"             # <-- server port ($VAST_TCP_PORT_22 on the server)
REMOTE_DIR="/workspace"      # project parent directory on the server
PROJECT_NAME="restoration"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."  # extract here
# ------------------------------------------------------------

echo "Pulling ${REMOTE_DIR}/${PROJECT_NAME}/ from ${SSH_USER}@${SSH_HOST}:${SSH_PORT} -> ${LOCAL_DIR}/${PROJECT_NAME}"

# Stream a tarball of the code from the server, excluding datasets,
# checkpoints, logs, and other outputs (several GB of them live next to
# the code on the server; only the code comes back).
ssh -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" \
    "cd '${REMOTE_DIR}' && tar czf - \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.claude' \
        --exclude='.venv' \
        --exclude='*.zip' \
        --exclude='*.pth' \
        --exclude='data' \
        --exclude='runs' \
        --exclude='checkpoints' \
        --exclude='tb' \
        '${PROJECT_NAME}'" \
  | tar xzf - -C "$LOCAL_DIR"

echo "Done. Pulled code (checkpoints/data stay on the server):"
echo "  ${LOCAL_DIR}/${PROJECT_NAME}"
echo
echo "To also fetch the baseline metrics or a trained checkpoint later, e.g.:"
echo "  scp -P ${SSH_PORT} ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/${PROJECT_NAME}/checkpoints/baseline_results.json ."
echo "  scp -P ${SSH_PORT} ${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/${PROJECT_NAME}/checkpoints/team17_ft/best_psnr.pth ."
