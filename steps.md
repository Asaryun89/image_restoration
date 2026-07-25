# Deploying the SRHCnet webapp to a new GPU server

Playbook for moving the backend to a fresh rented GPU instance (written for
Vast.ai; see the last section for plain servers). Verified against a Vast.ai
"vLLM-Omni" instance, RTX PRO 6000 Blackwell.

## 1. Update the connection details

Edit the `EDIT ME` block in `transfer.sh`:

- `SSH_HOST` — the new server's public IP.
- `SSH_PORT` — the mapped SSH port from the Vast.ai console's connect button
  (`$VAST_TCP_PORT_22` on the server), **not** 22. Port 22 on the public IP is
  the provider's host machine and will reject your key.

The mapping changes every time an instance is stopped and restarted, so
re-check it whenever the connection is refused.

## 2. Push code + checkpoints

```bash
bash transfer.sh          # code only, a few seconds
# SRHCnet weights (~2.5 GB) and ESRGAN weights (~64 MB):
scp -P <PORT> checkpoints/best_psnr.pth \
    root@<IP>:/workspace/srhc_infer/checkpoints/
scp -P <PORT> checkpoints/RRDB_ESRGAN_x4.pth \
    root@<IP>:/workspace/srhc_infer/checkpoints/
```

- The app serves both models; it auto-loads whichever checkpoints are
  present at startup and hides the picker if only one is found. Upload both
  for the SRHCnet/ESRGAN switch; skip `RRDB_ESRGAN_x4.pth` for SRHCnet only.
- `best_psnr.pth` is the slow part (~2.5 GB).
- Do **not** use rsync from Git Bash on Windows — it fails with
  `dup() in/out/err failed` (MSYS rsync cannot spawn the native Windows
  OpenSSH client). scp works.
- Verify integrity after upload: `md5sum` on both sides must match
  (SRHCnet checkpoint: `608e383e5ab72157cc086c65bdba519f`).

## 3. Install deps on the server

```bash
ssh -p <PORT> root@<IP>
source /venv/main/bin/activate
uv pip install fastapi 'uvicorn[standard]' python-multipart pillow numpy
```

Torch with CUDA is preinstalled on Vast images — confirm with
`python -c "import torch; print(torch.cuda.is_available())"` → `True`.
On Blackwell-class GPUs (RTX 50xx / RTX PRO 6000 / B200) the torch wheel
must be built for CUDA >= 12.8 (`cu128`+); older wheels install fine but die
with "no kernel image is available" at the first GPU op.

## 4. Wire the service (Vast.ai-specific)

Three files on the server. Run the app as a supervisor service behind the
instance's Caddy token-auth edge — not loose in a shell (dies on disconnect,
no restart, no logs).

### 4a. Wrapper script `/opt/supervisor-scripts/srhc-upscaler.sh`

```bash
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "SRHC Upscaler"

source /venv/main/bin/activate
cd /workspace/srhc_infer
pty python app.py --checkpoint checkpoints/best_psnr.pth \
    --device cuda --host 127.0.0.1 --port 17080 2>&1
```

`chmod +x` it. The `"SRHC Upscaler"` label must match the portal.yaml entry
below.

### 4b. Supervisor unit `/etc/supervisor/conf.d/srhc-upscaler.conf`

```ini
[program:srhc-upscaler]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/srhc-upscaler.sh
autostart=true
autorestart=unexpected
startsecs=90
startretries=10
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_logfile_maxbytes=0
```

`startsecs=90` because loading the 2.5 GB checkpoint takes a while.

### 4c. Portal entry (Caddy auth edge)

Find a free external port — the numbers differ per instance:

```bash
vast-capabilities | jq '.instance.open_ports[] | select(.in_use==false)'
```

Add the app to `/etc/portal.yaml` (external_port = the free `container_port`,
e.g. 10100; internal_port = 17080) and restart Caddy:

```bash
python3 -c "import yaml; d=yaml.safe_load(open('/etc/portal.yaml')); \
d['applications']['SRHC Upscaler']={'hostname':'localhost','external_port':10100,\
'internal_port':17080,'open_path':'/','name':'SRHC Upscaler'}; \
yaml.safe_dump(d, open('/etc/portal.yaml','w'), sort_keys=False)"
supervisorctl restart caddy
```

### 4d. Start it — only after the checkpoint upload has finished

```bash
supervisorctl reread && supervisorctl update
supervisorctl status srhc-upscaler      # STARTING -> RUNNING (~1-2 min)
tail -f /var/log/portal/srhc-upscaler.log
```

If started before the checkpoint exists, it crash-loops with a
"checkpoint corrupted / not found" error — just restart it after the upload.

## 5. Verify

On the server, grab the token and the public app port:

```bash
echo $OPEN_BUTTON_TOKEN
echo $VAST_TCP_PORT_10100     # public port for external_port 10100
```

From your own machine:

```bash
curl -H "Authorization: Bearer <TOKEN>" http://<IP>:<PUBLIC_APP_PORT>/api/info
# expect {"device":"cuda","scale":4,...}

curl -H "Authorization: Bearer <TOKEN>" -F "file=@test.png" \
     -o out.png http://<IP>:<PUBLIC_APP_PORT>/api/upscale
```

Browser: `http://<IP>:<PUBLIC_APP_PORT>/?token=<TOKEN>` (or the "Open"
button next to *SRHC Upscaler* in the Vast Instance Portal).

Reference timings (RTX PRO 6000): 48x48 → 0.26 s; 1280x800 → ~12 s
inference. The app takes ~10 GB VRAM and coexists with the instance's
vLLM service.

## Caveats

- **No persistent volume ⇒ a recycle/destroy wipes everything** (code,
  checkpoint, service files). Plain stop/start is safe. Check with
  `vast-capabilities | jq '.instance.workspace_is_volume'`.
- The token, free-port numbers, and SSH port all change per instance —
  re-derive them each time; don't reuse bookmarked URLs.
- The app has no auth of its own; Caddy provides it. Never bind it to
  `0.0.0.0` on a raw open port unless you intend it to be public.

## Non-Vast server (plain Linux + GPU)

Skip step 4 entirely:

```bash
pip install -r requirements.txt      # needs a CUDA torch build
tmux new -s srhc
python app.py --checkpoint checkpoints/best_psnr.pth --host 0.0.0.0 --port 8000
```

Put your own auth in front (reverse proxy) or keep it private via an SSH
tunnel: `ssh -p <PORT> -L 8000:127.0.0.1:8000 root@<IP>` then open
`http://127.0.0.1:8000` locally.
