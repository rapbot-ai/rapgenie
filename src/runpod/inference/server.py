"""Plain HTTP server — wraps inference.infer.run() as submit + poll.

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
- ASYNC /infer (v1.1.1, changed from synchronous): POST /infer validates,
  kicks the inference off on a background thread, and answers 202 with an
  inferJobId immediately; the dispatcher then polls GET /result/<id>.
  The sync version held the HTTP request open for the whole inference —
  but Pod traffic rides https://<podId>-<port>.proxy.runpod.net, which is
  Cloudflare-fronted, and Cloudflare kills any request that stays quiet
  for ~100s with a 524 (seen in prod 2026-07-29: worker finished fine at
  125s, dispatcher got a 524 at ~100s, job reported failed). Submit+poll
  keeps every HTTP exchange sub-second, so inference duration is
  unbounded. Must be paired with the matching dispatcher (attemptDispatch
  submit+poll) in rapbot-mobile.

Endpoints:

    GET  /health            -> 200 {"status": "ok", "busy": <bool>}
    POST /infer             -> body is the same shape as the serverless job
                               "input" (see handler.py's docstring):
                               202 {"inferJobId": ...}   accepted, running
                               400 {"error": ...}   validation / ConfigError
                               401 {"error": ...}   bad or missing bearer token
                               503 {"error": ...}   GPU already running a job
    GET  /result/<id>       -> 200 {"status": "RUNNING"}
                               200 {"status": "COMPLETED", "wavSignedUrl": ..., "text": ...}
                               200 {"status": "FAILED", "error": ...}
                               401 {"error": ...}   bad or missing bearer token
                               404 {"error": ...}   unknown id (e.g. the
                                   server restarted and lost in-memory
                                   state — the dispatcher treats this as a
                                   worker failure and retries elsewhere)

Auth: the Pod's proxy URL is publicly reachable, and these endpoints
trigger paid GPU work plus S3 reads/writes — so if the AUTH_TOKEN env var
is set on the Pod, /infer AND /result require a matching
`Authorization: Bearer <token>` header. /health stays open (it leaks
nothing and Batch 2's dispatcher health checks shouldn't need a secret to
ask "are you alive?").
"""

from __future__ import annotations

import os
import threading
import uuid

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

# In-memory only, by design: results die with the Pod, matching the fleet's
# "worker registry is transient, S3/pg-boss are the only durable pieces"
# rule. A dispatcher polling for an id this dict doesn't know gets a 404
# and treats it as a worker failure. Pruned to the newest few entries —
# one-job-at-a-time means this never meaningfully grows, but a long-lived
# Pod shouldn't leak memory either.
_results: dict[str, dict] = {}
_results_order: list[str] = []
_RESULTS_KEPT = 20


def _store_result(infer_job_id: str, result: dict) -> None:
    _results[infer_job_id] = result
    if infer_job_id not in _results_order:
        _results_order.append(infer_job_id)
    while len(_results_order) > _RESULTS_KEPT:
        _results.pop(_results_order.pop(0), None)


def _authorized(request: Request) -> bool:
    auth_token = os.environ.get("AUTH_TOKEN")
    if not auth_token:
        return True
    return request.headers.get("authorization") == f"Bearer {auth_token}"


def _run_job(infer_job_id: str, cfg: InferenceConfig) -> None:
    """Background-thread body. Owns releasing the GPU lock (acquired by
    /infer before the thread starts, so the busy check and the work are
    one atomic claim)."""
    try:
        wav_signed_url = run_inference(cfg)
        _store_result(infer_job_id, {"status": "COMPLETED", "wavSignedUrl": wav_signed_url, "text": cfg.text})
    except Exception as e:  # noqa: BLE001 - deliberately broad: report, don't crash the worker
        _store_result(infer_job_id, {"status": "FAILED", "error": f"inference run raised an exception: {e}"})
    finally:
        _gpu_lock.release()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "busy": _gpu_lock.locked()}


@app.post("/infer")
def infer(body: dict, request: Request) -> JSONResponse:
    if not _authorized(request):
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

    infer_job_id = uuid.uuid4().hex
    _store_result(infer_job_id, {"status": "RUNNING"})
    threading.Thread(target=_run_job, args=(infer_job_id, cfg), daemon=True).start()

    return JSONResponse(status_code=202, content={"inferJobId": infer_job_id})


@app.get("/result/{infer_job_id}")
def result(infer_job_id: str, request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse(status_code=401, content={"error": "missing or invalid bearer token"})

    stored = _results.get(infer_job_id)
    if stored is None:
        return JSONResponse(status_code=404, content={"error": f"unknown inferJobId {infer_job_id}"})
    return JSONResponse(status_code=200, content=stored)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
