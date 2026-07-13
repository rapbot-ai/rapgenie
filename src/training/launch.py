"""Distributed launch wrapper.

The notebook trained on whatever single GPU Colab handed it — there was no
path to multi-GPU. `torchrun` handles process spawning, rank assignment, and
rendezvous for you; this module just wires it into the same `training.train`
entrypoint so single-GPU and multi-GPU runs are the *same command* with a
different `--nproc_per_node`, instead of separate code paths.

Usage (from rapgenie/src/, so the `training` package resolves):
    torchrun --nproc_per_node=4 -m training.launch --config configs/train.yaml
"""

from __future__ import annotations

import os

from training.train import main as train_main


def main() -> None:
    # torchrun sets these env vars for every spawned process. RADTTS's own
    # train.py reads dist_config.dist_url / dist_backend for its process
    # group init — we just make sure this process's rank/world_size are
    # visible to it via the environment, which is the standard torchrun
    # contract, rather than re-implementing rendezvous ourselves.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank == 0:
        print(f"[launch] starting distributed run: world_size={world_size}")

    train_main()


if __name__ == "__main__":
    main()
