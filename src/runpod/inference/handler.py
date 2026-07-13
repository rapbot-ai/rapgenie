"""RunPod Serverless handler — wraps inference.infer.run() as a queue-based
job. Separate endpoint from src/runpod/model-training/handler.py (training);
see ./Dockerfile for why they're split.

Follows RunPod's documented handler pattern (runpod.serverless.start with a
handler function reading job["input"]): https://docs.runpod.io/serverless/workers/handler-functions

Job input shape (see scripts/submit_inference_job.js for
how this gets built and sent):

    {
      "input": {
        "text": "<text to synthesize>",
        "checkpoint_s3_key": "checkpoints/lupefiasco-radtts-warmstart-v5/<run_id>/model_10800",
        "storage_bucket": "...",           (optional, defaults to the training artifacts bucket)
        "vocoder_checkpoint_key": "...",   (optional, defaults to the pretrained HiFi-GAN checkpoint)
        "vocoder_config_key": "...",       (optional)
        "speaker": "lupefiasco",           (optional)
        "token_dur_scaling": 1.5           (optional)
      }
    }

Only 'text' and 'checkpoint_s3_key' are required — everything else has a
sensible default in InferenceConfig. 'checkpoint_s3_key' is deliberately the
one thing with no default: which checkpoint to run is a per-request decision,
never an implicit "latest" the worker guesses at (see config.py's docstring).
"""

from __future__ import annotations

import runpod

from inference.config import ConfigError, InferenceConfig
from inference.infer import run as run_inference


def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    if not job_input.get("text"):
        return {"error": "job input must include 'text': the text to synthesize."}
    if not job_input.get("checkpoint_s3_key"):
        return {
            "error": "job input must include 'checkpoint_s3_key': the full S3 key of a "
            "trained checkpoint, e.g. 'checkpoints/<run>/model_10800'. See "
            "src/runpod/inference/scripts/submit_inference_job.js."
        }

    # Only pass through fields the request actually set, so anything omitted
    # falls back to InferenceConfig's own defaults rather than this handler
    # re-declaring what those defaults are.
    optional_fields = (
        "storage_bucket",
        "output_bucket",
        "vocoder_checkpoint_key",
        "vocoder_config_key",
        "speaker",
        "token_dur_scaling",
    )
    kwargs = {k: job_input[k] for k in optional_fields if k in job_input}

    try:
        cfg = InferenceConfig(text=job_input["text"], checkpoint_s3_key=job_input["checkpoint_s3_key"], **kwargs)
    except ConfigError as e:
        return {"error": f"invalid inference request: {e}"}

    try:
        runpod.serverless.progress_update(job, "request validated, starting inference")
        wav_signed_url = run_inference(cfg)
    except Exception as e:  # noqa: BLE001 - deliberately broad: report, don't crash the worker
        return {"error": f"inference run raised an exception: {e}"}

    return {"wavSignedUrl": wav_signed_url, "text": cfg.text}


runpod.serverless.start({"handler": handler})
