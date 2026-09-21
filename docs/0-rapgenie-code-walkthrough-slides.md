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
PREP, before the interview:
1. Warmstart the GPUs 1 hour before.
2. Generate outputs beforehand.
3. Make sure always-on servers scale with multiple concurrent jobs.
4. Log into GitHub, AWS, RunPod, Weights & Biases, App Store Connect, Android Store
-->

---

## 1. USE CASE: what rapBot does

- Users write lyrics that get synthesized via rappers' voices
- Each voice is a TTS model fine-tuned from Nvidia's RADTTS
- The voice models constitute core business value

<!--
SOURCES:
- Apple App Store: https://appstoreconnect.apple.com/apps/1553540835/distribution/info
- Google Play: https://play.google.com/console/u/0/developers/7015808534271269202/app/4973394726403743674/app-dashboard
- martinconnor.com/rapgenie
-->

---

## 2. MOTIVATION: why automate training?

- 133 artists in the catalog → 133 voice models
- Every dataset upgrade means retraining all of them
- That's 1000+ training runs, too many to babysit by hand

<!--
SOURCES:
- Lightsail: https://us-east-1.console.aws.amazon.com/lightsail/webapp/home?region=us-east-1#
- TablePlus, Postman collection
-->

---

## 3. PROBLEM: bad reproducibility/auditability

- Hyperparameters typed into a Google Colab notebook
- Data on a Drive mount that dies with the session
- No metrics dashboard
- Unpersisted logs

<!--
SOURCES:
- Colab notebook: https://colab.research.google.com/drive/159klw_hkpt5imC80Tkoi752G_XNZdNe7#scrollTo=My8jR5wx7abv
- Upstream RADTTS PR #35: https://github.com/NVIDIA/radtts/pull/35
- Upstream issues I commented on: https://github.com/NVIDIA/radtts/issues?q=commenter%3A%40me
- Fork: https://github.com/rapbot-ai/radtts
-->

---

## 4. SOLUTION: automated, containerized training jobs

- **Maintainable:** RADTTS is a vendored black box
- **Scalable:** job queue and GPU provisioning via RunPod Serverless
- **Visible:** metrics dashboard via Weights & Biases
- **Durable:** checkpoints saved to S3 as they're written
- **Portable:** containerized via Docker

<!--
SOURCES:
- RunPod: https://console.runpod.io/pods
- W&B: https://wandb.ai/rapbot-ai/radtts-voice-clone/table
- S3: https://us-east-1.console.aws.amazon.com/s3/buckets/martinconnor-radtts-training-artifacts?region=us-east-1
- Docker Hub: https://hub.docker.com/repository/docker/skygamer313/radtts-train-worker
-->

---

## 5. Code Walkthrough

| Approach | File | Code |
| --- | --- | --- |
| Vendored black box | `train.py` | 1a-1d |
| Queued job | `handler.py` | 2a |
| Metrics to W&B | `train.py` | 3a-3c |
| RunPod Serverless | `handler.py` | 4a |
| Checkpoints to S3 | `train.py` | 5a |

---

## 6. Future Improvements

1. Warm restart
   - Checkpoint download to the worker isn't built
   - Then automatic retries, the config already has the setting
2. Log parser
   - Safe while pinned to one commit
   - Sturdier source: TensorBoard files the model already writes
