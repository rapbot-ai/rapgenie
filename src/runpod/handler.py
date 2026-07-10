"""RunPod Serverless handler — wraps training.train.run() as a queue-based job.

Follows RunPod's documented handler pattern (runpod.serverless.start with a
handler function reading job["input"]): https://docs.runpod.io/serverless/workers/handler-functions

Job input shape (see src/runpod/scripts/submit_training_job.js for how this
gets built and sent):

    {
      "input": {
        "config_yaml": "<the full literal text of a src/configs/train.yaml file>"
      }
    }

Why ship the whole config as a string instead of just an S3 key or a config
name: it keeps the submitted job self-contained and guarantees whatever you
submit is exactly what gets hashed into config_hash / run_id by
training/config.py — no risk of "the config on the worker doesn't match the
config I meant to submit" drift.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import runpod

from training.train import run as run_training


def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    config_yaml = job_input.get("config_yaml")

    if not config_yaml:
        return {
            "error": "job input must include 'config_yaml': the full text "
            "content of a train.yaml config. See src/runpod/scripts/submit_training_job.js."
        }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = Path(f.name)

    try:
        runpod.serverless.progress_update(job, "config validated, starting training")
        exit_code = run_training(config_path)
    except Exception as e:  # noqa: BLE001 - deliberately broad: report, don't crash the worker
        return {"error": f"training run raised an exception: {e}"}
    finally:
        config_path.unlink(missing_ok=True)

    if exit_code != 0:
        return {"error": f"training run exited with non-zero code {exit_code}"}

    return {"status": "completed"}


runpod.serverless.start({"handler": handler})
