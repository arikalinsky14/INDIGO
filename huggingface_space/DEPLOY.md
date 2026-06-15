# Deploying INDIGO to Hugging Face Spaces (free tier)

End-to-end recipe for getting a public `https://huggingface.co/spaces/<you>/indigo` URL up. Free tier: 2 vCPU, 16 GB RAM, no GPU, sleeps after 48 h of inactivity (wakes on first request, ~30 s cold start).

## 0. One-time account setup

1. Create a free account at <https://huggingface.co/join>.
2. Add an SSH key under **Settings → SSH and GPG Keys** (or use a write token under **Access Tokens** for HTTPS).
3. Install Git LFS locally: <https://git-lfs.com/>. Verify with `git lfs version`.

## 1. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. **Space name:** `indigo` (or whatever — this becomes the URL).
3. **License:** MIT.
4. **Space SDK:** **Docker** (not Streamlit / Gradio).
5. **Docker template:** **Blank**.
6. **Visibility:** Public (or Private if you want auth-gated access).
7. Click **Create Space**. You'll land on an empty repo.

## 2. Set the OpenAI secret (optional but recommended)

If you want users to be able to use the natural-language **Prompt** tab without pasting their own key:

1. On the Space page → **Settings** → **Variables and secrets** → **New secret**.
2. **Name:** `OPENAI_API_KEY`
3. **Value:** `sk-...` (your key)
4. **Restart** the Space after saving.

You pay for those parse calls (~$0.001 each). If you'd rather have users bring their own key, skip this step — the UI has a key-paste field in Advanced settings.

To **disable** LLM-authored custom constraints (recommended for any public-facing Space): add a second secret with name `INDIGO_ALLOW_CUSTOM_CONSTRAINTS` and value `0`. This blocks the second-call code-generation path; the 8 standard constraints still work.

## 3. Push the code from your machine

The Space is its own git repo. We'll mirror this repo's code into it, swap the README for the HF-flavoured one, and push.

```bash
# In a fresh shell, NOT inside your INDIGO checkout:
cd ~/some-parent-dir

# Clone the empty Space
git clone https://huggingface.co/spaces/<your-username>/indigo
cd indigo

# Configure LFS for big binary files (the model checkpoint).
# These patterns match how INDIGO emits checkpoints.
git lfs track "data/checkpoints/**/model.pt"
git lfs track "data/checkpoints/**/optimizer.pt"
git lfs track "data/checkpoints/**/*.bin"
git lfs track "data/checkpoints/**/*.safetensors"
git add .gitattributes

# Pull in the INDIGO source. Easiest: copy the files in.
# (Adjust the source path to your INDIGO checkout.)
rsync -a --delete \
    --exclude='.git' --exclude='__pycache__' \
    --exclude='.venv' --exclude='venv' \
    --exclude='data/datasets' --exclude='data/train' --exclude='data/test' \
    --exclude='slurms' --exclude='outputs' --exclude='job-outputs' \
    --exclude='create_dataset/data_prompts' \
    ~/INDIGO/ ./

# Bring in just the ONE checkpoint you want to ship.
# (HF Spaces free LFS quota is 50 GB per repo — a single PyTorch
# checkpoint is well under that.)
mkdir -p data/checkpoints
cp -r ~/INDIGO/data/checkpoints/flex_raw_spectrum_..._cross_attn.../latest/ \
      data/checkpoints/flex_raw_spectrum_..._cross_attn.../latest/

# Swap the README — HF reads YAML frontmatter from this file to know
# which SDK / port to use, and renders it on the Space landing page.
cp huggingface_space/README.md README.md
```

Windows PowerShell equivalent of the rsync line:

```powershell
robocopy C:\path\to\INDIGO . /MIR `
    /XD .git __pycache__ .venv venv data\datasets data\train data\test `
        slurms outputs job-outputs create_dataset\data_prompts
```

Then commit + push:

```bash
git add .
git commit -m "Initial deploy: INDIGO inference server"
git push origin main
```

The first push uploads the checkpoint via LFS — give it 5–20 minutes depending on your upload speed.

## 4. Watch the build

The Space page shows a **Building** status as soon as you push. Click **Logs** → **Build logs** to watch the Docker build. Expected to take 4–8 minutes the first time (PyTorch + JAX wheels). Then **Container logs** show the FastAPI server starting:

```
[server] device: cpu
[server] loading model from data/checkpoints/...
[server] JLL library: 84 materials from ...
[server] default pool (M_MAX-capped): 32
[server] prewarming JAX + pipeline (one tiny dummy solve)...
[server] prewarm done in 22.4s
[server] listening on http://0.0.0.0:7860
```

Once **App is running** is green, visit `https://huggingface.co/spaces/<you>/indigo`.

## 5. Update workflow

When you change code in this repo:

```bash
# In INDIGO/
git push origin main          # for the main repo

# In ~/indigo (the Space clone)
rsync -a --delete --exclude '...' ~/INDIGO/ ./
git add . && git commit -m "..."
git push origin main           # triggers HF rebuild
```

If you only changed code (not deps), the rebuild is fast — Docker reuses the deps layer. Bumping `requirements.txt` triggers the slow full rebuild.

For a long checkpoint update, you can rsync just the checkpoint file and skip the code copy:
```bash
cp ~/INDIGO/data/checkpoints/<tag>/latest/model.pt \
   ./data/checkpoints/<tag>/latest/model.pt
git add data/checkpoints/<tag>/latest/model.pt
git commit -m "Refresh checkpoint"
git push
```

## 6. Sharing the link

The public URL is `https://huggingface.co/spaces/<your-username>/indigo`. Send it as-is. First visitor after a long idle pays the ~30 s cold-start; subsequent users for the next 48 h get an already-warm server.

For your non-technical user: include screenshots of the **Prompt** tab and the **Structured** tab in a one-paragraph email. Mention "first request takes about 30 seconds; later ones are quick."

## Common issues

| Symptom | Fix |
|---|---|
| Build fails at `pip install jaxlayerlumos` | Make sure `build-essential` is in the Dockerfile's `apt-get install`. It already is in this repo's `Dockerfile`. |
| Container starts then exits with `model not found` | Checkpoint LFS push didn't complete. Run `git lfs ls-files` in the Space repo to confirm. |
| Browser shows 503 / 502 immediately | The first cold start hasn't finished. Wait 30 s and refresh. |
| "Connection closed before a result arrived" | This was a bug fixed in the SSE result frame (NaN tokens) — make sure you're on a commit dated 2026-06-14 or later. |
| Space is asleep | First request wakes it; expect 30–60 s to first byte. To keep it warm, paid tiers offer "always on" for ~$9/mo. |
| Want auth-gated access | Make the Space **Private** (Settings → Visibility). Only logged-in collaborators see it. |
