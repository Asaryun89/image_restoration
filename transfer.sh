#!/usr/bin/env bash
# Push the SRHCnet inference webapp from THIS local machine UP to the remote
# GPU server, so the app can be served there. Run it ON YOUR LOCAL MACHINE:
#
#   bash transfer.sh                    # push code only (checkpoint assumed present on server)
#   bash transfer.sh --with-checkpoint  # also upload checkpoints/best_psnr.pth (~2.5 GB)
#
# The remote training server almost certainly already holds a valid
# checkpoint (that is where best_psnr.pth came from), so the checkpoint
# upload is opt-in — skip it unless the server has no usable .pth.
set -euo pipefail

# ------------------------- EDIT ME -------------------------
SSH_USER="root"
SSH_HOST="14.136.19.62"   # <-- server IP   ($PUBLIC_IPADDR on the server)
SSH_PORT="30038"             # <-- server port ($VAST_TCP_PORT_22 on the server)
REMOTE_DIR="/workspace"      # parent directory on the server
PROJECT_NAME="srhc_infer"    # app is deployed to $REMOTE_DIR/$PROJECT_NAME
CHECKPOINT="checkpoints/best_psnr.pth"   # local path (relative to this script)
# ------------------------------------------------------------

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this project's dir
REMOTE_PATH="${REMOTE_DIR}/${PROJECT_NAME}/"

WITH_CHECKPOINT=0
for arg in "$@"; do
    [ "$arg" = "--with-checkpoint" ] && WITH_CHECKPOINT=1
done

echo "Pushing webapp -> ${SSH_USER}@${SSH_HOST}:${SSH_PORT}:${REMOTE_PATH}"

# Make sure the destination exists on the server.
ssh -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "mkdir -p '${REMOTE_PATH}'"

# Stream only the webapp files (explicit list — nothing else rides along:
# no corrupt/large *.pth, no datasets, no venv, no local .claude state).
# __pycache__/*.pyc are stripped in case any exist under models/.
tar czf - -C "$LOCAL_DIR" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        app.py \
        infer.py \
        requirements.txt \
        README.md \
        models \
        static \
  | ssh -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "tar xzf - -C '${REMOTE_PATH}'"

echo "Code pushed."

# Optional: upload a valid checkpoint (large). Uses rsync when available
# (resumable, shows progress); falls back to scp otherwise.
if [ "$WITH_CHECKPOINT" = "1" ]; then
    if [ ! -f "${LOCAL_DIR}/${CHECKPOINT}" ]; then
        echo "ERROR: ${LOCAL_DIR}/${CHECKPOINT} not found." >&2
        exit 1
    fi
    echo "Uploading ${CHECKPOINT} (~$(du -h "${LOCAL_DIR}/${CHECKPOINT}" | cut -f1))..."
    ssh -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "mkdir -p '${REMOTE_PATH}/checkpoints'"
    if command -v rsync >/dev/null 2>&1; then
        rsync -e "ssh -p ${SSH_PORT}" -avhP \
            "${LOCAL_DIR}/${CHECKPOINT}" \
            "${SSH_USER}@${SSH_HOST}:${REMOTE_PATH}/${CHECKPOINT}"
    else
        scp -P "$SSH_PORT" "${LOCAL_DIR}/${CHECKPOINT}" \
            "${SSH_USER}@${SSH_HOST}:${REMOTE_PATH}/${CHECKPOINT}"
    fi
    echo "Checkpoint uploaded."
fi

echo
echo "Done. Next, on the server:"
echo "  ssh -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST}"
echo "  cd ${REMOTE_PATH}"
echo "  pip install -r requirements.txt          # torch CUDA build recommended"
echo "  python app.py --checkpoint ${CHECKPOINT} --host 0.0.0.0 --port 8000"
echo
echo "Then reach the app from any device via an SSH tunnel (safest):"
echo "  ssh -p ${SSH_PORT} -L 8000:127.0.0.1:8000 ${SSH_USER}@${SSH_HOST}"
echo "  # open http://127.0.0.1:8000 locally"
