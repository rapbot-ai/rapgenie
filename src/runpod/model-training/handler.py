"""RunPod Serverless handler — wraps training.train.run() as a queue-based job.

Follows RunPod's documented handler pattern (runpod.serverless.start with a
handler function reading job["input"]): https://docs.runpod.io/serverless/workers/handler-functions

Job input shape (see src/runpod/model-training/scripts/submit_training_job.js for how this
gets built and sent):

    {
      "input": {
        "config_yaml": "<the full literal text of a src/configs/train.yaml file>",
        "resume": false,
        "resume_from": null
      }
    }

Why ship the whole config as a string instead of just an S3 key or a config
name: it keeps the submitted job self-contained and guarantees whatever you
submit is exactly what gets hashed into config_hash / run_id by
training/config.py — no risk of "the config on the worker doesn't match the
config I meant to submit" drift.

'resume'/'resume_from' are deliberately kept out of that string. Resume is a
per-job decision made at submit time, never a property of a committed
train.yaml — so it travels as its own top-level input field and is passed
straight into training.train.run() as the sole source of resume state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import runpod

from training.config import ConfigError, ResumeConfig
from training.train import run as run_training


def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    config_yaml = job_input.get("config_yaml")

    if not config_yaml:
        return {
            "error": "job input must include 'config_yaml': the full text "
            "content of a train.yaml config. See src/runpod/model-training/scripts/submit_training_job.js."
        }

    try:
        resume = ResumeConfig(
            enabled=job_input.get("resume", False),
            from_checkpoint=job_input.get("resume_from"),
            override_iteration=job_input.get("resume_override_iteration"),
        )
    except ConfigError as e:
        return {"error": f"invalid resume fields in job input: {e}"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = Path(f.name)

    try:
        runpod.serverless.progress_update(job, "config validated, starting training")
        exit_code = run_training(config_path, resume=resume)
    except Exception as e:  # noqa: BLE001 - deliberately broad: report, don't crash the worker
        return {"error": f"training run raised an exception: {e}"}
    finally:
        config_path.unlink(missing_ok=True)

    if exit_code != 0:
        return {"error": f"training run exited with non-zero code {exit_code}"}

    return {"status": "completed"}


runpod.serverless.start({"handler": handler})
