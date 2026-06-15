# Container for the INDIGO inference frontend.
#
# Targets two equivalent uses:
#   - Hugging Face Spaces (Docker SDK): the platform sets $PORT to 7860
#     and exposes the public URL.
#   - Local docker run -p 8000:8000 indigo for self-hosted.
#
# Image notes
# -----------
# - Pinned to Python 3.11 slim. jax + torch + numpy wheels all ship on cp311.
# - Build-essential is kept around because jaxlayerlumos compiles a small C
#   extension on install on some platforms. libgomp1 covers OpenMP runtime.
# - The image does NOT bake in a checkpoint. Checkpoint(s) live under
#   data/checkpoints/<tag>/latest/ in the repo (via Git LFS on Spaces) and
#   are mounted as part of the build context.
# - JAX is forced to CPU because the physics chain blocks jit/vmap (see
#   inference/src/simulate.py docstring on the JLL stackrt_eps_mu_base
#   assert). Saves a GPU image and avoids cuDNN/cuBLAS dependency drift.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JAX_PLATFORMS=cpu \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user (UID 1000). Mirror
# that here so the same Dockerfile works in both Spaces and `docker run` —
# easier to predict where the writable directories are.
RUN useradd -m -u 1000 indigo
USER indigo
WORKDIR /home/indigo/app

# Install deps first so the layer is cached between code-only changes.
COPY --chown=indigo:indigo requirements.txt .
RUN pip install --user -r requirements.txt
ENV PATH="/home/indigo/.local/bin:${PATH}"

# Now the code + checkpoint(s).
COPY --chown=indigo:indigo . .

# Default to 7860 (HF Spaces convention) but honour $PORT if the platform
# overrides it (Render, Railway, Fly all do).
EXPOSE 7860

CMD ["sh", "-c", "python -m inference.frontend.server --host 0.0.0.0 --port ${PORT:-7860}"]
