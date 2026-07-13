# RunPod Serverless worker image for INFERENCE — a separate endpoint from
# src/runpod/Dockerfile (training), not the same image with a task switch.
#
# Training and inference have genuinely different resource shapes: training
# is one long-running job needing a big container disk (dataset + warmstart +
# checkpoints) and a long execution timeout; inference is a single forward
# pass needing only a checkpoint + vocoder and returning in seconds. Sharing
# an endpoint would mean sizing everything for the worse case. Splitting the
# Dockerfile costs a bit of duplication in the apt-get/pip setup below (both
# still clone the same pinned RADTTS fork — that's a shared dependency, not
# duplicated logic), in exchange for each endpoint being configured — disk,
# GPU tier, timeout, scaling — for what it actually does.
#
# Build from the rapgenie/ REPO ROOT, not from this src/runpod-inference/
# directory, so it can COPY src/inference, src/storage, src/configs, and
# src/runpod-inference/handler.py from here:
#
#   docker build --platform linux/amd64 \
#     -f src/runpod-inference/Dockerfile \
#     -t DOCKERHUB_USERNAME/radtts-infer-worker:v1.0.0 \
#     .
#
# --platform linux/amd64 is required by RunPod even if you're building on
# Apple Silicon — see https://docs.runpod.io/serverless/workers/deploy

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-dev python3.10-dev libpython3.10-dev python3-pip git ffmpeg sox build-essential && \
    rm -rf /var/lib/apt/lists/*

# Same fork, same pinned commit, same reasoning as ../training/Dockerfile:
# upstream NVIDIA/RADTTS's requirements.txt breaks on a clean install
# (unpinned numpy/numba/llvmlite, setuptools>=81 dropping distutils,
# matplotlib>=3.8 vs contourpy). Fixed once in rapbot-ai/radtts@a517713 —
# don't reintroduce it. Inference needs this same checkout for radtts.py,
# inference.py, hifigan_*.py, and tts_text_processing/* — a different set of
# entrypoints off the fork than training uses, but the same fork.
ARG RADTTS_REPO=https://github.com/rapbot-ai/radtts.git
ARG RADTTS_COMMIT=a5177136f37fa55c74ad16c1a2cbd705a543a5ad
RUN git clone ${RADTTS_REPO} /opt/RADTTS && \
    cd /opt/RADTTS && git checkout ${RADTTS_COMMIT}

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

# Excludes torch/torchaudio (installed above from cu121) and jupyter (no
# notebook tooling needed at runtime). Unlike the training image, tensorboard
# is NOT installed here — that's only needed by RADTTS's train.py
# (SummaryWriter), which this worker never imports.
RUN grep -v -E '^(torch|torchaudio|jupyter)$' /opt/RADTTS/requirements.txt > /tmp/radtts-requirements.txt && \
    pip3 install --no-cache-dir -r /tmp/radtts-requirements.txt

COPY src/inference/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# The runpod SDK itself — not needed by the plain docker-run path, only this one.
RUN pip3 install --no-cache-dir "runpod~=1.7"

# src/inference -> /app/src/inference (the "inference" package), src/storage
# -> /app/src/storage (shared blob-store client, also used by training — see
# src/runpod/Dockerfile). PYTHONPATH=/app/src makes both importable as
# top-level packages: `from inference.config import ...`,
# `from storage.storage import ...`.
COPY src/inference ./src/inference
COPY src/storage ./src/storage

# The static speaker-roster filelist (see src/configs/speaker_roster.txt) —
# lets RADTTS rebuild the same speaker->id lookup table used at training
# time without needing the full training dataset. Path here must match
# SPEAKER_ROSTER_DIR in src/inference/infer.py.
COPY src/configs/speaker_roster.txt ./configs/speaker_roster.txt

COPY src/runpod-inference/handler.py .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

CMD ["python3", "-u", "handler.py"]
