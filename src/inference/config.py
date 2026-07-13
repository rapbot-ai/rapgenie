"""Typed config for a single inference request.

Unlike training/config.py, there's no YAML file here — an inference job is a
single request/response, not a long-running run someone wants to version and
diff over time. The whole config comes straight from the RunPod job payload
(see src/runpod-inference/handler.py), which is why this is just a validated
dataclass instead of a load_config()-from-file step.

`checkpoint_s3_key` is required and must be the FULL S3 key (e.g.
'checkpoints/lupefiasco-radtts-warmstart-v5/<run_id>/model_10800'), not just a
checkpoint name — inference has no training config in scope to reconstruct a
run_id from, so the caller has to say exactly where the checkpoint lives.
Explicit over implicit: no "use whatever's latest" default that could
silently point at a different checkpoint than the one you tested against.
"""

from __future__ import annotations

from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when an inference request is missing something required or
    internally inconsistent. Caught at the handler boundary and reported as
    a clean {"error": ...} response, not a raw traceback."""


# Same S3 bucket/keys src/configs/train.yaml's storage block points
# training at for these same two pretrained files. Duplicated as literals
# here rather than importing training.config: training's StorageConfig is
# tied to loading a whole train.yaml, which has nothing to do with a single
# inference request, and reaching into "training" from "inference" for two
# constants would be a stranger coupling than just repeating two S3 keys
# that rarely change (this is a pinned pretrained asset, not config that
# needs to stay in sync with a training run).
DEFAULT_STORAGE_BUCKET = "martinconnor-radtts-training-artifacts"
DEFAULT_OUTPUT_BUCKET = "rapbot-rapgenie-outputs"
DEFAULT_VOCODER_CHECKPOINT_KEY = "pretrained/hifigan_libritts100360_generator0p5.pt"
DEFAULT_VOCODER_CONFIG_KEY = "pretrained/hifigan_22khz_config.json"
DEFAULT_SPEAKER = "lupefiasco"
DEFAULT_TOKEN_DUR_SCALING = 1.5


@dataclass(frozen=True)
class InferenceConfig:
    text: str
    checkpoint_s3_key: str
    storage_bucket: str = DEFAULT_STORAGE_BUCKET
    output_bucket: str = DEFAULT_OUTPUT_BUCKET
    vocoder_checkpoint_key: str = DEFAULT_VOCODER_CHECKPOINT_KEY
    vocoder_config_key: str = DEFAULT_VOCODER_CONFIG_KEY
    speaker: str = DEFAULT_SPEAKER
    token_dur_scaling: float = DEFAULT_TOKEN_DUR_SCALING

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ConfigError("'text' must be a non-empty string")
        if not self.checkpoint_s3_key:
            raise ConfigError(
                "'checkpoint_s3_key' must be set to the full S3 key of a "
                "trained checkpoint, e.g. "
                "'checkpoints/lupefiasco-radtts-warmstart-v5/<run_id>/model_10800'"
            )
        if self.token_dur_scaling <= 0:
            raise ConfigError("'token_dur_scaling' must be > 0")
