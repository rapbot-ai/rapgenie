"""Plain HTTP server — wraps inference.infer.run() as a synchronous request.

The second front door on the same image as handler.py (see ./Dockerfile):
the RunPod Serverless endpoint uses the image's default CMD (handler.py),
while an always-on Pod overrides its container start command to
`python3 -u /app/server.py` and gets this server instead. Both call the
same InferenceConfig -> inference.infer.run() path; the ONLY difference is
the invocation protocol (RunPod's queue vs. plain HTTP).

Deliberately dumb on purpose:

- No queue. Queueing/retry/assignment is the gpu-dispatcher's job (in the
  rapbot-mobile repo). This server does one job at a time and says "busy"
  (503) if asked to do two — the dispatcher should never let that happen.
- Synchronous /infer. The dispatcher holds the request open and measures
  executionTime itself; no callback/webhook machinery on the Pod.

Endpoints:

    GET  /health  -> 200 {"status": "ok", "busy": <bool>}
    POST /infer   -> body is the same shape as the serverless job "input"
                     (see handler.py's docstring for the field reference):
                     200 {"wavSignedUrl": ..., "text": ...}
                     400 {"error": ...}   validation / ConfigError
                     401 {"error": ...}   bad or missing bearer token
                     503 {"error": ...}   GPU already running a job
                     500 {"error": ...}   inference raised

Auth: the Pod's proxy URL (https://<podId>-<port>.proxy.runpod.net) is
publicly reachable, and this endpoint triggers paid GPU work plus S3 reads/
writes — so if the AUTH_TOKEN env var is set on the Pod, /infer requires a
matching `Authorization: Bearer <token>` header. /health stays open (it
leaks nothing and Batch 2's dispatcher health checks shouldn't need a
secret to ask "are you alive?").
"""

from __future__ import annotations

import os
import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from inference.config import ConfigError, InferenceConfig
from inference.infer import run as run_inference

app = FastAPI()

# One GPU, one job. Non-blocking acquire in /infer: a second concurrent
# request gets an immediate 503 rather than silently queueing here —
# queueing is the dispatcher's job, and a hidden second queue on the Pod
# would make the dispatcher's busy-tracking wrong.
_gpu_lock = threading.Lock()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "busy": _gpu_lock.locked()}


@app.post("/infer")
def infer(body: dict, request: Request) -> JSONResponse:
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if request.headers.get("authorization") != f"Bearer {auth_token}":
            return JSONResponse(status_code=401, content={"error": "missing or invalid bearer token"})

    # Same validation and error messages as handler.py, so the dispatcher
    # and the serverless path return interchangeable shapes downstream.
    if not body.get("text"):
        return JSONResponse(
            status_code=400,
            content={"error": "job input must include 'text': the text to synthesize."},
        )
    if not body.get("checkpoint_s3_key"):
        return JSONResponse(
            status_code=400,
            content={
                "error": "job input must include 'checkpoint_s3_key': the full S3 key of a "
                "trained checkpoint, e.g. 'checkpoints/<run>/model_10800'. See "
                "src/runpod/inference/scripts/submit_inference_job.js."
            },
        )

    # Only pass through fields the request actually set, so anything omitted
    # falls back to InferenceConfig's own defaults rather than this server
    # re-declaring what those defaults are (same passthrough as handler.py).
    optional_fields = (
        "storage_bucket",
        "output_bucket",
        "vocoder_checkpoint_key",
        "vocoder_config_key",
        "speaker",
        "token_dur_scaling",
    )
    kwargs = {k: body[k] for k in optional_fields if k in body}

    try:
        cfg = InferenceConfig(text=body["text"], checkpoint_s3_key=body["checkpoint_s3_key"], **kwargs)
    except ConfigError as e:
        return JSONResponse(status_code=400, content={"error": f"invalid inference request: {e}"})

    if not _gpu_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=503,
            content={"error": "worker is busy with another inference job"},
        )
    try:
        wav_signed_url = run_inference(cfg)
    except Exception as e:  # noqa: BLE001 - deliberately broad: report, don't crash the worker
        return JSONResponse(status_code=500, content={"error": f"inference run raised an exception: {e}"})
    finally:
        _gpu_lock.release()

    return JSONResponse(status_code=200, content={"wavSignedUrl": wav_signed_url, "text": cfg.text})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
