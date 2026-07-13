# RADTTS Training Pipeline — Colab → Production

This is the training half of rapgenie: `src/training/` and `src/runpod/`
rebuild your `radtts_train_model_and_infer.ipynb` notebook as a production
training pipeline, living alongside rapgenie's existing inference code
(`src/express/`, `src/aws/`, `src/gpt/`, etc.) as its own domain. It trains
the exact same model (NVIDIA RADTTS, warm-started, pitch/energy
conditioning) — the difference is entirely in how the job is configured,
launched, tracked, and deployed.

## What the notebook actually did

Reading the notebook, the real pattern was:

1. Mount Google Drive for all inputs/outputs/checkpoints (`/content/drive/MyDrive/...`).
2. `git clone` **upstream** `NVIDIA/RADTTS` fresh into the Colab VM and
   `pip install -r requirements.txt` on every session. (Note: this worked in
   Colab because Colab ships a pre-baked, already-compatible numpy/numba/etc.
   environment that upstream's unpinned requirements.txt happens not to
   disturb. The dependency breakage only surfaces on a clean install — which
   is exactly what happens in a Docker build or a fresh EC2 box, which is why
   rapgenie's production inference side (`src/bash/bash.js`, pointing at
   `~/radtts/.venv`) clones your fork (`rapbot-ai/radtts@a517713`) instead.
   This pipeline does the same, for the same reason — see the table below.)
3. Launch `train.py` as a subprocess via `!python3 train.py -c config.json -p key=value ...`,
   overriding hyperparameters as flat CLI key-value pairs.
4. Track progress with a manually-launched TensorBoard pointed at a Drive path.
5. To change a hyperparameter or resume, you'd hand-edit the CLI invocation, or
   edit `train.py` itself with `sed` (see cell 16 — patching the resume iteration
   directly in RADTTS source).
6. The "config file" was a JSON blob pasted into a markdown cell with an instruction
   to manually copy it into `RADTTS/configs/config_ljs_dap_.json` before running.
7. Inference was another manual `!python3 inference.py ...` call against a
   hardcoded checkpoint path you'd update by hand each time.

This works for one person iterating live in a notebook. It breaks down the moment
you need: reproducibility (what config produced checkpoint `model_1400`?),
unattended/scheduled runs, multi-GPU, cost tracking, or handing the job to a
teammate or a scheduler.

## What changed, and why (mapped to what interviewers will actually ask)

| Notebook approach | Production approach here | Why it matters |
|---|---|---|
| Google Drive as the filesystem | Object storage (S3/GCS) for data + checkpoints, config-driven paths | Drive isn't durable/auditable at scale; blob storage is what every JD in your search (brightwheel's "data foundations," Assured's "artifact management, promotion strategies") is actually asking about. |
| `git clone`-ing **upstream** `NVIDIA/RADTTS` fresh every run | Clones **your fork**, `rapbot-ai/radtts`, pinned to commit `a5177136f37fa55c74ad16c1a2cbd705a543a5ad` | Upstream's `requirements.txt` is unpinned and breaks under a modern pip resolver: numpy/numba/llvmlite have to be version-matched as a trio or the import fails, matplotlib>=3.8 breaks against contourpy, and setuptools>=81 drops a `distutils` shim this codebase imports. Your fork's `a517713` commit already fixed this once — cloning upstream in the Dockerfile would silently reintroduce the exact bug you already solved. This is also why the Dockerfile installs the fork's `requirements.txt` *before* this pipeline's own — one file owns the fragile ML/audio stack (numpy/numba/llvmlite/scipy/scikit-learn/pandas/matplotlib/contourpy/setuptools/torch/torchaudio), the other only owns what this pipeline adds (`wandb`, `boto3`, `pyyaml`), so they never fight over a version. |
| `pip install` every run | Pinned `Dockerfile` (frozen deps, CUDA version pinned) | Reproducibility. "It worked in Colab last month" is not an answer in a staff interview. |
| Hyperparams as ad hoc `-p key=value` CLI strings | Structured config in `src/configs/train.yaml`, loaded and validated by `src/training/config.py` (dataclass + schema check) | Config governance — literally Assured's JD ("environment and configuration management systems, ensuring consistency, traceability, reproducibility"). |
| Manually pasted JSON config file | Config is versioned in git alongside code, one file per experiment, referenced by name/hash in every run's logs | Traceability: you can always answer "what config trained this checkpoint." |
| `sed`-patching `train.py` to change the resume iteration | `train.py` never gets hand-edited; resume (`resume_from`, `override_iteration`) is sent as its own field on the job payload, never stored in `train.yaml` — one source of truth, not a config value that could disagree with a submit-time override | Never mutate vendored/third-party source in place — patch through an explicit, single-sourced parameter, not config file state. |
| TensorBoard pointed at a Drive folder you remember to launch | Structured experiment tracking (Weights & Biases run, tagged with git commit + config hash) alongside TensorBoard | Observability into the training lifecycle — brightwheel's JD calls this out directly ("evaluation harnesses," "monitoring that ties system health to output quality"). |
| One process, one GPU, launched by hand in a notebook cell | `torchrun`-based launch (`src/training/launch.py`) that works identically on 1 GPU or N, dispatched to RunPod Serverless on demand | This is the GPU fleet/job-orchestration story from your rapBot work, generalized — the same muscle Luma-style and brightwheel-style ("durable job execution system: retries, explicit budgets, idempotency, monitoring") roles are hiring for. |
| No retry/failure handling — a crashed Colab cell just... stops | Checkpointing on every eval interval + automatic resume-from-last-checkpoint on restart | Idempotency and retries under a budget — this is verbatim brightwheel JD language. Currently a manual per-job payload field (`resume`/`resume_from` on the RunPod request, see `src/runpod/handler.py`); a job-queue-enforced retry budget like the durable-job-queue interview-prep exercise you built separately is the natural next step once this is running. |
| No automated validation at all | `src/training/config.py` + `src/training/data_validation.py` fail fast, on every run, before a GPU is ever touched | Config validated before it's ever promoted to a real GPU run — matches Assured's "release and configuration layer" framing. No CI yet on purpose (see "Deploying it for real" below) — validation happens locally / in the container at run time instead of in a pipeline. |

## Layout

This follows rapgenie's existing `src/<domain>/` convention (same shape as
`src/aws/`, `src/bash/`, `src/express/`, `src/gpt/`) rather than being a
separate top-level project:

```
rapgenie/
  src/
    configs/
      config_ljs_dap.json    # pre-existing: RADTTS inference config
      train.yaml              # the one thing you edit per training experiment
    training/
      config.py               # typed config loading + validation (fails fast on bad config)
      data_validation.py      # sanity-checks the aligned dataset before a run ever starts a GPU
      train.py                # thin, typed wrapper around RADTTS train.py — resolves paths,
                               # sets up W&B + structured logging, handles resume, uploads
                               # checkpoints to blob storage on each save
      launch.py               # torchrun-compatible entrypoint, works for 1 or N GPUs
      storage.py               # local/S3/GCS blob-storage abstraction
      requirements.txt        # this pipeline's own deps (NOT torch/numpy/etc — see the file)
      Dockerfile               # pinned CUDA/PyTorch base + RADTTS deps, plain docker-run image
      README.md                # this file
      RUNBOOK.md               # the actual deploy-it-for-real commands
    runpod/                   # the actual deployment target: handler, worker Dockerfile, submit/status scripts
      handler.py
      Dockerfile
      scripts/
        build_worker.js         # node, not bash — see the file for why
        submit_training_job.js
        check_job_status.js
```

The `training` package is importable as `training.config`, `training.train`,
etc. — both Dockerfiles set `PYTHONPATH=/app/src` so it resolves the same
way a local run does from `rapgenie/src/` (see "Running it" below).

## Deploying it for real

Kept deliberately simple, on purpose: a few plain `aws` CLI commands to
create the S3 bucket and a scoped IAM user, a `docker build`/`push`, and
creating the RunPod Serverless endpoint by hand in the console (five
minutes, RunPod's own documented flow). No Terraform, no Kubernetes. **See
`RUNBOOK.md` for the exact command sequence** — using your own AWS and
RunPod accounts; I can't run these myself (they require your credentials
and spend real money on GPU time), but every command has been verified for
syntax/logic, and the JSON/YAML payload construction is tested end-to-end
against your actual `src/configs/train.yaml`.

The philosophy here: simple first, enterprise/abstracted later. Right now
that means no automated tests and no CI either — you run `build_worker.js`
and `submit_training_job.js` by hand and eyeball the results, and that's
fine until it isn't. Once the plain-CLI version is actually running a real
job end to end, the natural next steps — in rough order — are: add back a
`tests/` directory covering `training/config.py` and
`training/data_validation.py` once those modules are changing often enough
that manual eyeballing stops being reliable; add CI to run those tests +
validate config on every PR; codify the AWS side as Terraform once you're
touching it often enough that clicking/typing it by hand gets tedious or
error-prone; automate image build/push/deploy in CI instead of running
`build_worker.js` by hand; and only reach for Kubernetes if you end up
needing multiple concurrent training jobs sharing a GPU fleet rather than
one job at a time (RunPod Serverless already gets you scale-to-zero without
it). Doing it simply first is also the better interview story — it shows
you know *when* to reach for CI/Terraform/k8s, not just that you can.

## Running it

Local (single GPU, same as the notebook's warm-start cell) — build and run
from the **rapgenie repo root**:

```bash
docker build -f src/training/Dockerfile -t radtts-train .
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/checkpoints:/checkpoints \
  radtts-train python -m training.train --config configs/train.yaml
```

Multi-GPU, without Docker — run from **rapgenie/src/** so the `training`
package resolves:

```bash
cd src
torchrun --nproc_per_node=4 -m training.launch --config configs/train.yaml
```

RunPod Serverless (unattended, on-demand GPU — see `RUNBOOK.md`):

```bash
cd src/runpod/scripts && node submit_training_job.js
```

## What to say in an interview about this

The honest framing: "I trained this in a Colab notebook originally because I was
a solo founder iterating fast — that was the right call at the time. Here's how
I'd rebuild it as a production pipeline a team could operate: versioned config,
containerized environment, blob storage instead of Drive, checkpoint-based resume,
and fail-fast validation before anything ever touches a GPU — deliberately kept to
plain CLI commands and no CI/Terraform/Kubernetes yet, because those earn their
keep once there's a team or a recurring workflow around this, not before."
That's a stronger answer than pretending the notebook was already production-grade,
or than reaching for infrastructure the problem doesn't need yet.
