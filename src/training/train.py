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
  4. Handles resume explicitly as a per-job value passed in from the outside.
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
import threading
import time
from pathlib import Path

from training.config import ConfigError, PipelineConfig, ResumeConfig, load_config
from training.data_validation import DatasetValidationError, assert_dataset_ready
from blob_storage.blob_storage import build_blob_store

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


def resolve_vocoder(cfg: PipelineConfig, local_ckpt_dir: Path) -> tuple[Path, Path]:
    """Download the HiFi-GAN vocoder checkpoint + config used to synthesize
    audio for validation-time logging (compute_validation_loss -> load_vocoder
    in the vendored train.py/inference.py). RADTTS's bundled config_ljs_dap.json
    points at 'models/hifigan_config_22khz.json' and 'models/hifigan_ljs_generator_v1'
    by default — files that ship with nobody's checkout, same category as the
    warmstart checkpoint. StorageConfig already had vocoder_checkpoint/
    vocoder_config fields (and train.yaml already sets them); this was just
    never wired up to actually download them and override the paths."""
    store = build_blob_store(cfg.storage.backend, cfg.storage.bucket)
    local_ckpt = local_ckpt_dir / "vocoder.pt"
    local_config = local_ckpt_dir / "vocoder_config.json"
    if not local_ckpt.exists():
        logger.info("downloading vocoder checkpoint -> %s", local_ckpt)
        store.download(cfg.storage.vocoder_checkpoint, local_ckpt)
    if not local_config.exists():
        logger.info("downloading vocoder config -> %s", local_config)
        store.download(cfg.storage.vocoder_config, local_config)
    return local_config, local_ckpt


def build_train_command(
    cfg: PipelineConfig,
    warmstart_ckpt: Path,
    output_dir: Path,
    local_data_dir: Path,
    vocoder_config: Path,
    vocoder_ckpt: Path,
) -> list[str]:
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
        # RADTTS's own bundled config_ljs_dap.json points these at its demo
        # filelist ('filelists/ljs_audiopath_text_speaker_train_filelist.txt'),
        # which doesn't exist in this image. Point it at what
        # resolve_dataset() actually downloaded instead. audiodir is left
        # alone — it's already "wavs", matching the folder name we download
        # into.
        "data_config.training_files.LJS.basedir": f"{local_data_dir}/",
        "data_config.training_files.LJS.filelist": "training.txt",
        "data_config.validation_files.LJS.basedir": f"{local_data_dir}/",
        "data_config.validation_files.LJS.filelist": "validation.txt",
        # Same story as data_config above: RADTTS's bundled default
        # ('models/hifigan_config_22khz.json', 'models/hifigan_ljs_generator_v1')
        # isn't in this image. Point at what resolve_vocoder() downloaded.
        "train_config.vocoder_config_path": str(vocoder_config),
        "train_config.vocoder_checkpoint_path": str(vocoder_ckpt),
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


def _watch_and_offload_checkpoints(
    cfg: PipelineConfig, output_dir: Path, stop_event: threading.Event, poll_interval: float = 15.0
) -> None:
    """Runs in a background thread for the lifetime of the RADTTS training
    subprocess. Uploads each new checkpoint to blob storage and deletes it
    locally as soon as it appears, instead of leaving everything on disk
    until the whole run finishes.

    RADTTS's own save_checkpoint() never rotates or deletes old checkpoints
    — every iters_per_checkpoint iterations it writes a brand new model_<N>
    file and keeps every previous one forever. On a fixed-size container
    disk that grows without bound and eventually fails mid-write (exactly
    what happened: torch.save's PytorchStreamWriter died with a byte-count
    mismatch after the 4th checkpoint filled the disk). Uploading-then-
    deleting each checkpoint as it's written keeps local disk usage roughly
    constant no matter how long training runs, without touching vendored
    RADTTS source — this thread just watches the output directory from
    the outside.

    torch.save() isn't atomic (no write-to-temp-then-rename), so a
    checkpoint file can be mid-write when this thread notices it. Skip any
    file whose size hasn't stabilized across a short gap; pick it up on the
    next poll instead of risking an upload of a truncated checkpoint."""
    store = build_blob_store(cfg.storage.backend, cfg.storage.bucket)
    uploaded: set[str] = set()

    def _sweep() -> None:
        for ckpt in sorted(output_dir.glob("model_*")):
            if ckpt.name in uploaded:
                continue
            try:
                size_before = ckpt.stat().st_size
                time.sleep(1)
                size_after = ckpt.stat().st_size
            except FileNotFoundError:
                continue  # vanished between glob() and stat() — skip, not ours to worry about
            if size_before != size_after:
                continue  # still being written; catch it on the next sweep

            remote = f"{cfg.storage.checkpoint_prefix}/{cfg.run_id}/{ckpt.name}"
            logger.info("uploading checkpoint %s -> %s (then deleting local copy)", ckpt, remote)
            store.upload(ckpt, remote)
            ckpt.unlink()
            uploaded.add(ckpt.name)

    while not stop_event.wait(poll_interval):
        _sweep()
    _sweep()  # final sweep once the subprocess has exited, to catch the last checkpoint written


def run(config_path: Path, resume: ResumeConfig | None = None) -> int:
    # `resume` is passed straight into load_config() as its one and only
    # source — never read from config_path's YAML, never applied as a
    # second, later override on top of it. See config.py's load_config()
    # docstring for why: two places that could each supply a resume value
    # is exactly what we're avoiding here, even if they'd usually agree.
    try:
        cfg = load_config(config_path, resume=resume)
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
    vocoder_config, vocoder_ckpt = resolve_vocoder(cfg, local_ckpt_dir)
    cmd = build_train_command(cfg, warmstart_ckpt, local_ckpt_dir, local_data_dir, vocoder_config, vocoder_ckpt)
    logger.info("launching: %s", " ".join(cmd))

    # RADTTS's bundled config_ljs_dap.json is full of paths relative to the
    # RADTTS repo itself — heteronyms_path, phoneme_dict_path,
    # vocoder_config_path, vocoder_checkpoint_path, betabinom_cache_path,
    # etc. That's not an oversight: the original notebook did `%cd
    # /content/RADTTS` before invoking train.py, so those paths always
    # resolved relative to the repo root. Our subprocess otherwise inherits
    # this process's cwd (/app, per the Dockerfile's WORKDIR), so without
    # this, each relative path fails FileNotFoundError one at a time as
    # training progresses further. Setting cwd here reproduces the
    # notebook's %cd — the paths WE override above (checkpoints, warmstart,
    # local_data_dir) are all absolute, so this doesn't affect them.
    stop_watcher = threading.Event()
    watcher = threading.Thread(
        target=_watch_and_offload_checkpoints,
        args=(cfg, local_ckpt_dir, stop_watcher),
        daemon=True,
    )
    watcher.start()

    proc = subprocess.run(cmd, cwd=str(RADTTS_REPO))

    stop_watcher.set()
    watcher.join(timeout=120)  # give the final sweep time to upload whatever's left

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
