"""Production inference entrypoint.

Sibling to src/training/train.py, same philosophy: a thin wrapper *around*
the vendored RADTTS `inference.py` rather than a rewrite of it, controlling
it from the outside through CLI args and process boundaries. What this adds
on top of a bare `python3 inference.py` call:

  1. Downloads the checkpoint + vocoder named in the request from S3 instead
     of assuming they're already sitting on disk somewhere.
  2. Resolves the same relative-path issues training.train hit (RADTTS's
     bundled config_ljs_dap.json has paths relative to the repo root) by
     running the subprocess with cwd=RADTTS_REPO.
  3. Supplies a minimal static speaker-roster filelist so RADTTS can rebuild
     the same speaker->id lookup table used at training time, without
     needing the full training dataset — inference never reads audio through
     that filelist, it's only used to look up 'lupefiasco''s speaker id.
  4. Uploads the resulting wav to S3 and returns a signed URL, instead of
     leaving it on local (ephemeral, per-job) worker disk.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import uuid
from pathlib import Path

from inference.config import InferenceConfig
from blob_storage.blob_storage import build_blob_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("radtts_infer")

RADTTS_REPO = Path("/opt/RADTTS")  # baked into the Docker image, not git-cloned at runtime

# Baked into the image by src/runpod/inference/Dockerfile from
# src/configs/speaker_roster.txt. A fixed, tiny, checked-in asset — not
# something that varies per request, so it isn't part of InferenceConfig.
SPEAKER_ROSTER_DIR = Path("/app/configs")
SPEAKER_ROSTER_FILENAME = "speaker_roster.txt"


def resolve_checkpoint(cfg: InferenceConfig, local_dir: Path) -> Path:
    store = build_blob_store("s3", cfg.storage_bucket)
    local_path = local_dir / "checkpoint.pt"
    logger.info("downloading checkpoint %s -> %s", cfg.checkpoint_s3_key, local_path)
    store.download(cfg.checkpoint_s3_key, local_path)
    return local_path


def resolve_vocoder(cfg: InferenceConfig, local_dir: Path) -> tuple[Path, Path]:
    store = build_blob_store("s3", cfg.storage_bucket)
    local_ckpt = local_dir / "vocoder.pt"
    local_config = local_dir / "vocoder_config.json"
    logger.info("downloading vocoder checkpoint %s -> %s", cfg.vocoder_checkpoint_key, local_ckpt)
    store.download(cfg.vocoder_checkpoint_key, local_ckpt)
    logger.info("downloading vocoder config %s -> %s", cfg.vocoder_config_key, local_config)
    store.download(cfg.vocoder_config_key, local_config)
    return local_config, local_ckpt


def build_infer_command(
    cfg: InferenceConfig,
    checkpoint: Path,
    vocoder_config: Path,
    vocoder_ckpt: Path,
    text_file: Path,
    output_dir: Path,
) -> list[str]:
    """Translate InferenceConfig into vendored inference.py's flat CLI
    dialect — the inference-side equivalent of train.py's
    build_train_command. This is the only place that speaks RADTTS's
    argparse flags for inference."""
    overrides = {
        # Must match what training used (train.py's build_train_command),
        # since it determines the model architecture the checkpoint's
        # weights were actually trained against.
        "model_config.n_speakers": 1,
        # Points RADTTS's Data loader at the static speaker-roster filelist
        # instead of its own bundled demo filelist — same override key
        # train.py uses for the real dataset, just aimed at a fixed asset
        # baked into this image instead of something downloaded per-run.
        "data_config.training_files.LJS.basedir": f"{SPEAKER_ROSTER_DIR}/",
        "data_config.training_files.LJS.filelist": SPEAKER_ROSTER_FILENAME,
    }
    p_args = [f"{k}={v}" for k, v in overrides.items()]

    return [
        "python3",
        str(RADTTS_REPO / "inference.py"),
        "-c",
        str(RADTTS_REPO / "configs" / "config_ljs_dap.json"),
        "-r",
        str(checkpoint),
        "-v",
        str(vocoder_ckpt),
        "-k",
        str(vocoder_config),
        "-t",
        str(text_file),
        "-s",
        cfg.speaker,
        "--speaker_attributes",
        cfg.speaker,
        "--speaker_text",
        cfg.speaker,
        "--token_dur_scaling",
        str(cfg.token_dur_scaling),
        "-o",
        str(output_dir),
        "-p",
        *p_args,
    ]


def _upload_and_sign(cfg: InferenceConfig, wav_path: Path) -> str:
    import boto3  # imported lazily, same reasoning as storage.S3BlobStore

    s3 = boto3.client("s3")
    key = f"{uuid.uuid4()}.wav"
    logger.info("uploading %s -> s3://%s/%s", wav_path, cfg.output_bucket, key)
    s3.upload_file(str(wav_path), cfg.output_bucket, key)
    return s3.generate_presigned_url("get_object", Params={"Bucket": cfg.output_bucket, "Key": key})


def run(cfg: InferenceConfig) -> str:
    """Runs one inference request end-to-end, returns a signed URL to the
    generated wav. Raises on any failure — the caller (handler.py) is
    responsible for turning that into a clean {"error": ...} response."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        checkpoint = resolve_checkpoint(cfg, tmp_path)
        vocoder_config, vocoder_ckpt = resolve_vocoder(cfg, tmp_path)

        text_file = tmp_path / "text-input.txt"
        text_file.write_text(cfg.text)

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_infer_command(cfg, checkpoint, vocoder_config, vocoder_ckpt, text_file, output_dir)
        logger.info("launching: %s", " ".join(cmd))

        # Same reason as training's cwd=RADTTS_REPO fix: config_ljs_dap.json's
        # heteronyms_path/phoneme_dict_path/betabinom_cache_path are relative
        # to the RADTTS repo root, not wherever this process's cwd happens to
        # be (/app, per the Dockerfile's WORKDIR).
        proc = subprocess.run(cmd, cwd=str(RADTTS_REPO))
        if proc.returncode != 0:
            raise RuntimeError(f"inference process exited with code {proc.returncode}")

        wav_files = sorted(output_dir.glob("*.wav"))
        if not wav_files:
            raise RuntimeError("inference process exited cleanly but produced no .wav output")

        return _upload_and_sign(cfg, wav_files[0])
