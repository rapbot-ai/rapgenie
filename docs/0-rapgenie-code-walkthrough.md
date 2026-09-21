# Automating TTS Model Training

rapgenie - `~/src/training/train.py`

I converted an open Nvidia TTS model from a brittle and manual touchpoint-heavy Google Colab interface to a scalable container running in a serverless cloud environment with visibility in Weights & Biases.

## Prep:

1. Warmstart the GPUs 1 hour before.
2. Generate outputs beforehand.
3. Make sure that always on servers scale with multiple concurrent jobs.
4. Log into github/aws/runpod/weights and biases.

## 1. Use Case

Section 1: why automate
├── rapBot: users rap in real rappers' voices
├── each rapper is its own fine-tuned RADTTS model
└── 133 artists, retrained on bigger datasets, so 1000+ runs

## 2. Problem

Not reproducible & not auditable
├── not reproducible: hyperparameters typed in a Google Colab notebook, data on a Drive mount that dies with the session
├── not auditable: no metrics dashboard, each run overwrote the last run's logs
└── brittle: repo cloned fresh every run, so upstream changes broke us

## 3. Solution

We built an automated, containerized training job
├── vendored black box: forked repo, pinned deps, archived
├── RunPod Serverless: job queue and GPU provisioning
├── W&B: metrics
└── S3: checkpoints uploaded as written

## 4. Implementation

Files: `src/training/train.py` and `src/runpod/model-training/handler.py`. Search for `WALKTHROUGH 1a` to jump.

| Statement | Code | What's there | File | Line |
| --- | --- | --- | --- | --- |
| **Vendored black box we never touch** | | | | |
| | 1a | The model's repo is baked into the Docker image. | `train.py` | 23 |
| | 1b | We call the model's own training script. | `train.py` | 105 |
| | 1c | We start it as a subprocess. | `train.py` | 164 |
| | 1d | We only read its output. | `train.py` | 173 |
| **Automated as a queued job** | | | | |
| | 2a | The handler calls `run()` for each job. | `handler.py` | 39 |
| **Metrics stream to W&B** | | | | |
| | 3a | Start the W&B run. | `train.py` | 320 |
| | 3b | Send train loss. | `train.py` | 184 |
| | 3c | Send validation loss. | `train.py` | 192 |
| **GPUs scale through RunPod Serverless** | | | | |
| | 4a | Register the handler with RunPod. | `handler.py` | 52 |
| **Checkpoints upload to blob storage** | | | | |
| | 5a | Upload each checkpoint as it's written. | `train.py` | 251 |

## 5. Future Optimizations

- Resume can't fetch a checkpoint from blob storage yet. It expects the file on local disk. The fix is to download it before training starts.
- Restarts are manual. If a run dies, someone has to find its run id in W&B. Then they find the last checkpoint in blob storage. Then they resubmit the job with both. The fix is automatic retries. The config already has a retry setting, but nothing reads it yet.
- We read metrics by parsing printed output. If that format changes on us, our metrics stream breaks, and nothing alerts us. The fix is to read the TensorBoard files the model already writes.
