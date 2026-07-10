"""Typed config loading for the RADTTS training pipeline.

The notebook's "config" was a JSON blob in a markdown cell with an instruction
to hand-copy it into place before running. That has two failure modes: (1) you
forget, and train against a stale config, or (2) the JSON is malformed and you
find out 45 minutes into a training run instead of at launch time.

This module loads configs/train.yaml, validates it against a schema, and fails
fast with a clear error before any GPU is ever allocated. That's the whole
point of "config governance" in a platform-engineering JD (see Assured's JD:
"environment and configuration management systems, ensuring consistency,
traceability, and reproducibility").
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config fails validation. Meant to be caught at the CLI
    boundary and reported clearly, not to bubble up as a stack trace after
    a GPU job has already started."""


@dataclass(frozen=True)
class StorageConfig:
    backend: str
    bucket: str
    data_prefix: str
    checkpoint_prefix: str
    vocoder_checkpoint: str
    vocoder_config: str
    warmstart_checkpoint: str

    def __post_init__(self) -> None:
        if self.backend not in {"local", "s3", "gcs"}:
            raise ConfigError(
                f"storage.backend must be one of local/s3/gcs, got {self.backend!r}"
            )


@dataclass(frozen=True)
class ResumeConfig:
    enabled: bool
    from_checkpoint: str | None
    override_iteration: int | None

    def __post_init__(self) -> None:
        if self.enabled and not self.from_checkpoint:
            raise ConfigError(
                "resume.enabled=true requires resume.from_checkpoint to be set"
            )


@dataclass(frozen=True)
class RetryConfig:
    max_restarts: int
    restart_backoff_seconds: int

    def __post_init__(self) -> None:
        if self.max_restarts < 0:
            raise ConfigError("retry.max_restarts must be >= 0")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    optim_algo: str
    use_amp: bool
    grad_clip_val: float
    iters_per_checkpoint: int
    unfreeze_modules: str
    loss_weights: dict[str, float]

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError("train.batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ConfigError("train.learning_rate must be > 0")


@dataclass(frozen=True)
class PipelineConfig:
    run_name: str
    seed: int
    storage: StorageConfig
    resume: ResumeConfig
    retry: RetryConfig
    train: TrainConfig
    raw: dict[str, Any] = field(repr=False)

    @property
    def config_hash(self) -> str:
        """Short hash of the raw config, logged with every run so any
        checkpoint can be traced back to exactly what produced it."""
        blob = yaml.safe_dump(self.raw, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def git_commit(self) -> str:
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    @property
    def run_id(self) -> str:
        """Stable, human-diffable identifier for this run: name + config hash
        + commit. This is what gets logged to W&B and stamped into the
        checkpoint prefix, so 'what config trained model_1400' is always
        answerable."""
        return f"{self.run_name}-{self.config_hash}-{self.git_commit}"

    def tracking_wandb_enabled(self) -> bool:
        return bool(self.raw.get("tracking", {}).get("wandb", {}).get("enabled", False))


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a training config. Raises ConfigError on anything
    malformed, missing, or internally inconsistent."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    try:
        run = raw["run"]
        storage = StorageConfig(**run["storage"])
        resume = ResumeConfig(**raw["resume"])
        retry = RetryConfig(**raw["retry"])
        train = TrainConfig(**raw["train"])
    except KeyError as e:
        raise ConfigError(f"missing required config key: {e}") from e
    except TypeError as e:
        raise ConfigError(f"config section has unexpected/missing fields: {e}") from e

    return PipelineConfig(
        run_name=run["name"],
        seed=run["seed"],
        storage=storage,
        resume=resume,
        retry=retry,
        train=train,
        raw=raw,
    )
