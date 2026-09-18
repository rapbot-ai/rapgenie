# Automating TTS Model Training

rapgenie - `~/src/training/train.py`

I converted an open Nvidia TTS model from a brittle and manual touchpoint-heavy Google Colab interface to an automated, scalable container running in a serverless cloud environment with visibility in Weights & Biases.

## PreP:

1. Warmstart the GPUs
2. Have prepared outputs at martinconnor.com/rapgenie

## 1. Use Case

rapBot lets users rap in their favorite rappers' voices using TTS models fine-tuned from Nvidia's open-source RADTTS. We automated model training because:
- each rapper needs a separate voice model
- those models are the core business value of the platform

## 2. Problem

Model training was run by hand in a Google Colab notebook. The data team ran each cell and watched the model's loss. This made it hard to reproduce because hyperparameters were typed into a cell, the model's code was cloned fresh each run, and data sat on a Drive mount that dropped with the Colab session. This was also hard to audit. There was no dashboard for loss metrics and each new run erased the last run's logs.

## 3. Solution

We treat the open-source model's repo as a vendored black box that we never touch. That saves us from maintaining a custom fork heavily coupled to our tech stack. We automated model training as a queued job around it. Model metrics stream to Weights and Biases. GPUs scale through RunPod Serverless. Checkpoints upload to blob storage as they're written.

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
