# RunPod Serverless worker image. Same pinned RADTTS fork and the same
# training pipeline code as src/training/Dockerfile (the standalone
# docker-run/torchrun path) — this is the SAME training logic, just wrapped
# for a different execution model: RunPod provisions a GPU worker on-demand
# per job and bills per second, instead of an always-on box. This is the
# "provider-aware GPU dispatch, workload-demand matching" line from your
# resume, applied to this training job specifically.
#
# This Dockerfile lives under src/runpod/model-training/ (not alongside the
# package at src/training/) because it, handler.py, and scripts/ are the
# RunPod-specific deploy wrapper — none of them import `runpod` in the actual
# training package, only here. The package itself (src/training/) stays
# deployment-agnostic and gets COPYed in from its own top-level location.
#
# Build from the rapgenie/ REPO ROOT, not from this
# src/runpod/model-training/ directory, so it can COPY src/training,
# src/blob_storage, src/configs, and this directory's own handler.py from
# here:
#
#   docker build --platform linux/amd64 \
#     -f src/runpod/model-training/Dockerfile \
#     -t DOCKERHUB_USERNAME/radtts-train-worker:v1.0.0 \
#     .
#
# --platform linux/amd64 is required by RunPod even if you're building on
# Apple Silicon — see https://docs.runpod.io/serverless/workers/deploy

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-dev python3.10-dev libpython3.10-dev python3-pip git ffmpeg sox build-essential && \
    rm -rf /var/lib/apt/lists/*

# Same fork, same pinned commit, same reasoning as src/training/Dockerfile:
# upstream NVIDIA/RADTTS's requirements.txt breaks on a clean install
# (unpinned numpy/numba/llvmlite, setuptools>=81 dropping distutils,
# matplotlib>=3.8 vs contourpy). Fixed once in rapbot-ai/radtts@a517713 —
# don't reintroduce it.
ARG RADTTS_REPO=https://github.com/rapbot-ai/radtts.git
ARG RADTTS_COMMIT=a5177136f37fa55c74ad16c1a2cbd705a543a5ad
RUN git clone ${RADTTS_REPO} /opt/RADTTS && \
    cd /opt/RADTTS && git checkout ${RADTTS_COMMIT}

# rapgenie's own commit, not RADTTS_COMMIT above (that's the pinned vendored
# fork's commit, a separate thing). Baked in at build time because /app never
# has a .git directory — only specific files get COPYed in below, not the
# whole repo — so config.py's git_commit property can't shell out to `git
# rev-parse` inside the container the way it can when you run this locally
# from an actual checkout. build_worker.js passes this as
# --build-arg GIT_COMMIT=$(git rev-parse --short HEAD), computed on your
# machine where .git genuinely exists.
ARG GIT_COMMIT=unknown

WORKDIR /app

# The RADTTS requirements include the Python `sox` wrapper, whose legacy
# setup.py imports numpy and typing_extensions and shells out to the SoX
# binary during metadata generation. Install those prerequisites first so
# the pinned requirements file can resolve cleanly under a modern pip build.
RUN pip3 install --no-cache-dir numpy==1.23.5 typing_extensions

# Install GPU PyTorch from the official cu121 index instead of from PyPI.
# The plain PyPI resolver pulls enormous nvidia-* wheel dependencies on top of
# the CUDA base image, which bloats the image and makes pushes unreliable.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchaudio

# RADTTS's runtime does not need notebook tooling in the worker image.
# Excluding torch/torchaudio here avoids re-resolving them from PyPI after the
# explicit cu121 install above.
RUN grep -v -E '^(torch|torchaudio|jupyter)$' /opt/RADTTS/requirements.txt > /tmp/radtts-requirements.txt && \
    pip3 install --no-cache-dir -r /tmp/radtts-requirements.txt

# RADTTS's own train.py does `from torch.utils.tensorboard import SummaryWriter`,
# but `tensorboard` itself isn't listed in the fork's requirements.txt (checked —
# it's missing upstream too, unrelated to what commit a517713 pinned). Not
# something to fix by editing the vendored fork's requirements.txt in place, so
# it's installed explicitly here instead.
RUN pip3 install --no-cache-dir tensorboard==2.16.2

COPY src/training/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# The runpod SDK itself — not needed by the plain docker-run path, only this one.
RUN pip3 install --no-cache-dir "runpod~=1.7"

# src/training -> /app/src/training (the "training" package, importable via
# PYTHONPATH=/app/src as `training.X`, matching src/training/Dockerfile).
# src/blob_storage -> /app/src/blob_storage: the blob-store client, kept at
# top level (not nested under either worker) because it's not RunPod-specific
# — src/runpod/inference/ needs the exact same S3-download behavior for its
# checkpoint/vocoder.
COPY src/training ./src/training
COPY src/blob_storage ./src/blob_storage
COPY src/configs/train.yaml ./configs/train.yaml
COPY src/runpod/model-training/handler.py .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    GIT_COMMIT=${GIT_COMMIT}

CMD ["python3", "-u", "handler.py"]
