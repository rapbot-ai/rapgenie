from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from training.config import ConfigError, EarlyStoppingConfig, PipelineConfig, ResumeConfig, load_config
from training.data_validation import DatasetValidationError, assert_dataset_ready
from blob_storage.blob_storage import build_blob_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("radtts_train")

# WALKTHROUGH 1a: the model's repo is baked into the Docker image
RADTTS_REPO = Path("/opt/RADTTS")


def resolve_dataset(cfg: PipelineConfig, local_data_dir: Path) -> None:
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
        "data_config.training_files.LJS.basedir": f"{local_data_dir}/",
        "data_config.training_files.LJS.filelist": "training.txt",
        "data_config.validation_files.LJS.basedir": f"{local_data_dir}/",
        "data_config.validation_files.LJS.filelist": "validation.txt",
        "train_config.vocoder_config_path": str(vocoder_config),
        "train_config.vocoder_checkpoint_path": str(vocoder_ckpt),
    }
    if cfg.resume.enabled:
        # WALKTHROUGH 5b: resume points the model at a saved checkpoint
        overrides["train_config.checkpoint_path"] = str(output_dir / cfg.resume.from_checkpoint)

    p_args = [f"{k}={v}" for k, v in overrides.items()]
    return [
        "python3",
        # WALKTHROUGH 1b: we call the model's own training script
        str(RADTTS_REPO / "train.py"),
        "-c",
        str(RADTTS_REPO / "configs" / "config_ljs_dap.json"),
        "-p",
        *p_args,
    ]


_TRAIN_ITER_RE = re.compile(r"^iter:\s*(\d+)\s*\(")


def _parse_train_line(line: str) -> tuple[int, dict[str, float]] | None:
    segments = [s.strip() for s in line.split("|")]
    match = _TRAIN_ITER_RE.match(segments[0])
    if not match:
        return None

    iteration = int(match.group(1))
    metrics: dict[str, float] = {}
    for segment in segments[1:]:
        if ":" not in segment:
            continue
        key, _, value = segment.partition(":")
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return iteration, metrics


_VAL_LOSS_RE = re.compile(r"^Validation loss:\s*(\{.*\})\s*$")
_VAL_LOSS_ENTRY_RE = re.compile(r"'([^']+)':\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _parse_val_line(line: str) -> dict[str, float] | None:
    match = _VAL_LOSS_RE.match(line)
    if not match:
        return None
    metrics: dict[str, float] = {}
    for key, value in _VAL_LOSS_ENTRY_RE.findall(match.group(1)):
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _run_training_subprocess(
    cmd: list[str],
    cwd: str,
    wandb_enabled: bool,
    early_stopping: EarlyStoppingConfig | None = None,
) -> int:
    if wandb_enabled:
        import wandb

    es = early_stopping if (early_stopping is not None and early_stopping.enabled) else None

    # WALKTHROUGH 1c: we start it as a subprocess
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    last_iteration: int | None = None
    best_value: float | None = None
    best_step: int | None = None
    early_stopped = False
    # WALKTHROUGH 1d: we only read its output
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        if not wandb_enabled and es is None:
            continue
        parsed = _parse_train_line(line)
        if parsed is not None:
            iteration, metrics = parsed
            last_iteration = iteration
            if wandb_enabled:
                # WALKTHROUGH 3b: send train loss
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=iteration)
            continue
        val_metrics = _parse_val_line(line)
        if val_metrics:
            if last_iteration is None:
                continue
            if wandb_enabled:
                # WALKTHROUGH 3c: send validation loss
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=last_iteration)
            if es is None:
                continue
            value = val_metrics.get(es.monitor)
            if value is None:
                continue
            if best_value is None or value < best_value:
                best_value, best_step = value, last_iteration
            elif last_iteration - best_step >= es.patience_steps and last_iteration >= es.min_steps:
                early_stopped = True
                msg = (
                    f"EARLY STOP at step {last_iteration}: no val/{es.monitor} improvement "
                    f"in {last_iteration - best_step} steps "
                    f"(best {best_value:.4f} at step {best_step}; patience {es.patience_steps})"
                )
                logger.info(msg)
                print(msg, flush=True)
                if wandb_enabled:
                    wandb.run.summary["early_stop/triggered"] = True
                    wandb.run.summary["early_stop/step"] = last_iteration
                    wandb.run.summary["early_stop/best_step"] = best_step
                    wandb.run.summary[f"early_stop/best_{es.monitor}"] = best_value
                proc.terminate()
                break

    if early_stopped:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 0

    proc.wait()
    return proc.returncode


def _watch_and_offload_checkpoints(
    cfg: PipelineConfig, output_dir: Path, stop_event: threading.Event, poll_interval: float = 15.0
) -> None:
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
                continue
            if size_before != size_after:
                continue

            remote = f"{cfg.storage.checkpoint_prefix}/{cfg.run_id}/{ckpt.name}"
            logger.info("uploading checkpoint %s -> %s (then deleting local copy)", ckpt, remote)
            # WALKTHROUGH 5a: upload each checkpoint as it's written
            store.upload(ckpt, remote)
            ckpt.unlink()
            uploaded.add(ckpt.name)

    while not stop_event.wait(poll_interval):
        _sweep()
    _sweep()


def run(config_path: Path, resume: ResumeConfig | None = None) -> int:
    try:
        cfg = load_config(config_path, resume=resume)
    except ConfigError as e:
        logger.error("config validation failed: %s", e)
        return 2

    logger.info("run_id=%s config_hash=%s git_commit=%s", cfg.run_id, cfg.config_hash, cfg.git_commit)

    if cfg.wandb.enabled:
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

    stop_watcher = threading.Event()
    watcher = threading.Thread(
        target=_watch_and_offload_checkpoints,
        args=(cfg, local_ckpt_dir, stop_watcher),
        daemon=True,
    )
    watcher.start()

    returncode = 1
    try:
        returncode = _run_training_subprocess(
            cmd,
            cwd=str(RADTTS_REPO),
            wandb_enabled=cfg.wandb.enabled,
            early_stopping=cfg.train.early_stopping,
        )
    finally:
        stop_watcher.set()
        watcher.join(timeout=120)
        if cfg.wandb.enabled:
            import wandb

            wandb.finish(exit_code=returncode)

    if returncode != 0:
        logger.error("training process exited with code %s", returncode)
    return returncode


def _init_wandb(cfg: PipelineConfig) -> None:
    import wandb

    # WALKTHROUGH 3a: start the W&B run
    wandb.init(
        project=cfg.wandb.project,
        id=cfg.run_id,
        name=cfg.run_id,
        resume="allow" if cfg.resume.enabled else "never",
        tags=cfg.wandb.tags,
        config=cfg.raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RADTTS production training entrypoint")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    sys.exit(run(args.config))


if __name__ == "__main__":
    main()
