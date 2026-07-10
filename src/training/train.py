"""Production training entrypoint.

This is what replaces the notebook cell:

    %cd /content/RADTTS
    !python3 train.py -c configs/config_ljs_dap.json -p train_config.learning_rate=0.0005 ...

It's intentionally a thin wrapper *around* the vendored RADTTS `train.py`
rather than a rewrite of it — you don't fork and modify a third-party
research repo's training loop; you control it from the outside through
config, environment, and process boundaries. What this adds on top:

  1. Validates config and dataset before touching a GPU (fail fast, cheap).
  2. Resolves all paths through the storage abstraction instead of assuming
     Google Drive is mounted at a fixed path.
  3. Sets up structured logging + W&B, tagged with the run_id (name + config
     hash + git commit) so every run is traceable back to what produced it.
  4. Handles resume explicitly through config instead of sed-patching
     `train.py`'s source to change the resume iteration.
  5. Uploads checkpoints to blob storage as they're written, instead of
     relying on Drive-mount durability.
  6. Exits with a non-zero code and a clear message on failure, so it behaves
     correctly under a Kubernetes Job's restart policy.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from training.config import ConfigError, PipelineConfig, load_config
from training.data_validation import DatasetValidationError, assert_dataset_ready
from training.storage import build_blob_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("radtts_train")

RADTTS_REPO = Path("/opt/RADTTS")  # baked into the Docker image, not git-cloned at runtime


def resolve_dataset(cfg: PipelineConfig, local_data_dir: Path) -> None:
    """Pull the aligned dataset down from blob storage to local disk for
    training, then validate it. Mirrors what Drive-mounting gave you for
    free, but explicit and backend-agnostic."""
    store = build_blob_store(cfg.storage.backend, cfg.storage.bucket)
    local_data_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("training.txt", "validation.txt"):
        remote = f"{cfg.storage.data_prefix}/{fname}"
        local = local_data_dir / fname
        logger.info("downloading %s -> %s", remote, local)
        store.download(remote, local)

    wavs_remote = f"{cfg.storage.data_prefix}/wavs"
    wavs_local = local_data_dir / "wavs"
    logger.info("downloading %s -> %s", wavs_remote, wavs_local)
    store.download_prefix(wavs_remote, wavs_local)

    assert_dataset_ready(
        train_filelist=local_data_dir / "training.txt",
        val_filelist=local_data_dir / "validation.txt",
        audio_dir=local_data_dir / "wavs",
    )


def resolve_warmstart_checkpoint(cfg: PipelineConfig, local_ckpt_dir: Path) -> Path:
    store = build_blob_store(cfg.storage.backend, cfg.storage.bucket)
    local_path = local_ckpt_dir / "warmstart.pt"
    if not local_path.exists():
        logger.info("downloading warmstart checkpoint -> %s", local_path)
        store.download(cfg.storage.warmstart_checkpoint, local_path)
    return local_path


def build_train_command(cfg: PipelineConfig, warmstart_ckpt: Path, output_dir: Path) -> list[str]:
    """Translate the typed config into RADTTS's expected `-p key=value`
    overrides. This is the *only* place in the whole pipeline that speaks
    RADTTS's flat CLI dialect — everywhere else in this codebase deals with
    the structured PipelineConfig."""
    overrides = {
        "train_config.learning_rate": cfg.train.learning_rate,
        "train_config.epochs": cfg.train.epochs,
        "train_config.weight_decay": cfg.train.weight_decay,
        "train_config.batch_size": cfg.train.batch_size,
        "train_config.optim_algo": cfg.train.optim_algo,
        "train_config.use_amp": cfg.train.use_amp,
        "train_config.grad_clip_val": cfg.train.grad_clip_val,
        "train_config.iters_per_checkpoint": cfg.train.iters_per_checkpoint,
        "train_config.unfreeze_modules": cfg.train.unfreeze_modules,
        "train_config.output_directory": str(output_dir),
        "train_config.warmstart_checkpoint_path": str(warmstart_ckpt),
        "model_config.n_speakers": cfg.raw["model"]["n_speakers"],
    }
    if cfg.resume.enabled:
        overrides["train_config.checkpoint_path"] = str(output_dir / cfg.resume.from_checkpoint)

    p_args = [f"{k}={v}" for k, v in overrides.items()]
    return [
        "python3",
        str(RADTTS_REPO / "train.py"),
        "-c",
        str(RADTTS_REPO / "configs" / "config_ljs_dap.json"),
        "-p",
        *p_args,
    ]


def upload_checkpoints(cfg: PipelineConfig, output_dir: Path) -> None:
    store = build_blob_store(cfg.storage.backend, cfg.storage.bucket)
    for ckpt in sorted(output_dir.glob("model_*")):
        remote = f"{cfg.storage.checkpoint_prefix}/{cfg.run_id}/{ckpt.name}"
        logger.info("uploading checkpoint %s -> %s", ckpt, remote)
        store.upload(ckpt, remote)


def run(config_path: Path) -> int:
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        logger.error("config validation failed: %s", e)
        return 2

    logger.info("run_id=%s config_hash=%s git_commit=%s", cfg.run_id, cfg.config_hash, cfg.git_commit)

    if cfg.tracking_wandb_enabled():
        _init_wandb(cfg)

    local_data_dir = Path("/data") / cfg.run_id
    local_ckpt_dir = Path("/checkpoints") / cfg.run_id
    local_ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolve_dataset(cfg, local_data_dir)
    except DatasetValidationError as e:
        logger.error("dataset validation failed, refusing to start training:\n%s", e)
        return 3

    warmstart_ckpt = resolve_warmstart_checkpoint(cfg, local_ckpt_dir)
    cmd = build_train_command(cfg, warmstart_ckpt, local_ckpt_dir)
    logger.info("launching: %s", " ".join(cmd))

    proc = subprocess.run(cmd)
    upload_checkpoints(cfg, local_ckpt_dir)

    if proc.returncode != 0:
        logger.error("training process exited with code %s", proc.returncode)
    return proc.returncode


def _init_wandb(cfg: PipelineConfig) -> None:
    import wandb

    wandb.init(
        project=cfg.raw["tracking"]["wandb"]["project"],
        name=cfg.run_id,
        tags=cfg.raw["tracking"]["wandb"].get("tags", []),
        config=cfg.raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RADTTS production training entrypoint")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.config))


if __name__ == "__main__":
    main()
