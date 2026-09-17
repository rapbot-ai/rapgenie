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
            run_id=job_input.get("resume_run_id"),
        )
    except ConfigError as e:
        return {"error": f"invalid resume fields in job input: {e}"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = Path(f.name)

    try:
        runpod.serverless.progress_update(job, "config validated, starting training")
        # WALKTHROUGH 2a: the handler calls run() for each job
        exit_code = run_training(config_path, resume=resume)
    except Exception as e:
        return {"error": f"training run raised an exception: {e}"}
    finally:
        config_path.unlink(missing_ok=True)

    if exit_code != 0:
        return {"error": f"training run exited with non-zero code {exit_code}"}

    return {"status": "completed"}


# WALKTHROUGH 4a: register the handler with RunPod
runpod.serverless.start({"handler": handler})
