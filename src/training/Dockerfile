# Build from the rapgenie/ REPO ROOT, not from this src/training/ directory,
# so it can COPY src/training and src/configs from here:
#
#   docker build -f src/training/Dockerfile -t radtts-train .
#
# Pinned CUDA + PyTorch base, instead of the notebook's Colab VM (whatever
# CUDA/driver version Google happens to have provisioned that day).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-dev python3.10-dev libpython3.10-dev python3-pip git ffmpeg sox build-essential && \
    rm -rf /var/lib/apt/lists/*

# RADTTS is vendored at build time from OUR fork, not upstream NVIDIA/RADTTS.
# Upstream's requirements.txt has known dependency-resolution breakage under
# a modern pip resolver: unpinned numpy/numba/llvmlite (numba and llvmlite
# are tightly version-coupled — mismatched pairs fail to import), matplotlib
# >=3.8 breaking against contourpy, and setuptools>=81 dropping the
# distutils shim this codebase still imports. That's exactly what commit
# a5177136f37fa55c74ad16c1a2cbd705a543a5ad ("chore: pin dependencies") on
# rapbot-ai/radtts fixes. Cloning upstream here instead of the fork would
# silently reintroduce a bug that's already been solved once — pin to the
# fork, pin to the exact commit, not a branch.
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

# Install the fork's pinned ML/audio stack FIRST — it owns numpy, numba,
# llvmlite, scipy, scikit-learn, pandas, matplotlib, contourpy, setuptools,
# librosa, and unidecode, all pinned/capped together on purpose. We exclude
# torch/torchaudio here because they were already installed from the cu121
# index above, and jupyter because the training image does not need notebook
# tooling at runtime.
RUN grep -v -E '^(torch|torchaudio|jupyter)$' /opt/RADTTS/requirements.txt > /tmp/radtts-requirements.txt && \
    pip3 install --no-cache-dir -r /tmp/radtts-requirements.txt

COPY src/training/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# src/training -> /app/src/training (the "training" package), src/configs ->
# /app/configs. PYTHONPATH=/app/src makes "training" importable as a
# top-level package: `from training.config import ...`, matching the folder
# name exactly.
COPY src/training ./src/training
COPY src/configs/train.yaml ./configs/train.yaml

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

ENTRYPOINT ["python3", "-m", "training.train"]
CMD ["--config", "configs/train.yaml"]
