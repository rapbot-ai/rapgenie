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

From a notebook someone babysat to a job we submit

`rapgenie/src/training/train.py`

<!--
PREP, before the interview:
1. Warmstart the GPUs 1 hour before.
2. Generate outputs beforehand.
3. Make sure always-on servers scale with multiple concurrent jobs.
4. Log into GitHub, AWS, RunPod, Weights & Biases.
-->

---

## 1. MOTIVATION: Why automate?

- rapBot lets users rap in real rappers' voices
- Each rapper is its own fine-tuned RADTTS model
- 133 artists, retrained on bigger datasets, so 1000+ training runs

---

## 2. PROBLEM: bad reproducibility/auditability

- Hyperparameters typed into a Google Colab notebook
- Data on a Drive mount that dies with the session
- No metrics dashboard
- Unpersisted logs

---

## 3. SOLUTION: automated, containerized training jobs

- Low maintenance cost: RadTTS is vendored black box
- Job queues and GPU provisioning (RunPod)
- Metrics dashboard (Weights & Biases)
- Checkpoints saved as written (S3)
- Portable container (Docker)

---

## 4. Code Walkthrough

| Approach | File | Code |
| --- | --- | --- |
| Vendored black box | `train.py` | 1a-1d |
| Queued job | `handler.py` | 2a |
| Metrics to W&B | `train.py` | 3a-3c |
| RunPod Serverless | `handler.py` | 4a |
| Checkpoints to S3 | `train.py` | 5a |

---

## 5. Future Improvements

1. Warm restart
   - Checkpoint download to the worker isn't built
   - Then automatic retries, the config already has the setting
2. Log parser
   - Safe while pinned to one commit
   - Sturdier source: TensorBoard files the model already writes
