---
marp: true
theme: gaia
class: invert
paginate: true
style: |
  section { font-size: 30px; padding: 60px 80px; }
  h1 { color: #f4b860; }
  h2 { color: #f4b860; font-size: 44px; }
  li { margin-bottom: 0.4em; }
  ul ul li { font-size: 26px; opacity: 0.85; }
  code { font-size: 0.9em; }
---

<!-- _class: lead invert -->

# Automating TTS Model Training

Google Colab notebook → containerized serverless training

`~/src/training/train.py`

<!--
## PREP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Do these before the interview:
1. Warmstart the GPUs 1 hour before.
2. Generate outputs beforehand.
3. Make sure always-on servers scale with multiple concurrent jobs.
4. Log into GitHub, AWS, RunPod, Weights & Biases, App Store Connect, Android Store
5. Open rapgenie and rapbot-mobile repos in VSCode
-->

---

# Walkthrough Structure

1. Business use case
2. Motivation (why automate?)
3. Technical problem
4. Technical solution
5. Code Walkthrough
6. App demo
7. Future state
8. Wrap-up

---

## USE CASE: what rapBot does

- Users write lyrics that get synthesized via rappers' voices
- Each voice is a TTS model fine-tuned from Nvidia's RADTTS
- The voice models constitute core business value

<!--
SOURCES:
- Apple App Store: https://appstoreconnect.apple.com/apps/1553540835/distribution/info
- Google Play: https://play.google.com/console/u/0/developers/7015808534271269202/app/4973394726403743674/app-dashboard
- https://martinconnor.com/rapgenie
-->

---

## MOTIVATION: why automate training?

- 133 artists in the catalog → 133 voice models
- Every dataset upgrade means retraining all of them
- That's 1000+ training runs, too many to babysit by hand

<!--
SOURCES:
- Lightsail: https://us-east-1.console.aws.amazon.com/lightsail/webapp/home?region=us-east-1#
- TablePlus, Postman collection
-->

---

## PROBLEM: bad reproducibility/auditability

- Notebook cloned RADTTS fresh each run: unpinned, any upstream change could break training
- Hyperparameters typed into a Google Colab notebook
- One GPU, one run at a time
- No metrics dashboard
- Checkpoints and logs on a Drive mount that dies with the session

<!--
SOURCES:
- Fixed colab notebook: https://colab.research.google.com/drive/159klw_hkpt5imC80Tkoi752G_XNZdNe7#scrollTo=My8jR5wx7abv
- Upstream RADTTS PR #35: https://github.com/NVIDIA/radtts/pull/35
- My fix for Google Colab notebook: https://github.com/NVIDIA/radtts/issues/16
- Upstream issues I commented on: https://github.com/NVIDIA/radtts/issues?q=commenter%3A%40me
- Our RadTTS Fork: https://github.com/rapbot-ai/radtts
-->

---

## SOLUTION: automated, containerized training jobs

- **Maintainable:** RADTTS is a vendored black box
- **Scalable:** job queue and GPU provisioning via RunPod
- **Visible:** metrics dashboard via Weights & Biases
- **Durable:** checkpoints saved to S3 as they're created
- **Portable:** containerized via Docker

<!--
SOURCES:
- RunPod: https://console.runpod.io/pods
- W&B: https://wandb.ai/rapbot-ai/radtts-voice-clone/table
- S3: https://us-east-1.console.aws.amazon.com/s3/buckets/martinconnor-radtts-training-artifacts?region=us-east-1
- Docker Hub: https://hub.docker.com/repository/docker/skygamer313/radtts-train-worker
-->

---

## Code Walkthrough

*▶ Submit training job now*

| Approach | File | Code |
| --- | --- | --- |
| Vendored black box | `train.py` | 1a-1d |
| Queued job | `handler.py` | 2a |
| Metrics to W&B | `train.py` | 3a-3c |
| RunPod Serverless | `handler.py` | 4a |
| Checkpoints to S3 | `train.py` | 5a |

---

## App Demo

1. Send training job via `POST /run`
   - Postman
2. Check queue/logs/metrics
   - RunPod, Weights & Biases
3. Create checkpoint
   - S3

<!--
SOURCES:
- POSTman: /train cURL
- RunPod: https://console.runpod.io/pods
- W&B: https://wandb.ai/rapbot-ai/radtts-voice-clone/table
- S3: https://us-east-1.console.aws.amazon.com/s3/buckets/martinconnor-radtts-training-artifacts?region=us-east-1
-->

---

## Future State

1. Warm restart
   - Checkpoint download to the worker isn't built
   - Then automatic retries, the config already has the setting
2. Log parser
   - Safe while pinned to one commit
   - Sturdier source: TensorBoard files the model already writes

---

## Wrap-Up

- martinconnor.com/rapgenie
- Questions/comments?