# Runbook: Actually Deploying This

Everything below is real — every command provisions or touches a real
resource under your own AWS and RunPod accounts, using your own credentials.
I can't run these for you (I don't have your credentials, and provisioning
GPU capacity costs real money that only you should authorize), but every
command here is copy-pasteable. Run them from your own terminal, in order.

Target architecture, per your choices: AWS for storage (S3 + IAM), RunPod
Serverless for GPU capacity. Kept deliberately simple — plain `aws` CLI
commands, no Terraform, no Kubernetes. That's on purpose: get one real job
running end to end by hand first, and only reach for IaC/orchestration once
you're doing this often enough that doing it by hand is actually the
bottleneck. `src/runpod/` (relative to the rapgenie repo root) is the whole
deployment target here.

## 0. Prerequisites

- AWS account with billing enabled, and the AWS CLI configured (`aws configure`
  or equivalent env vars) with credentials that can create S3 buckets and IAM
  users. **Not** the credentials this runbook creates for you later — those
  are scoped-down and come *from* this step.
- A [Docker Hub](https://hub.docker.com/) account, and Docker installed locally.
- A [RunPod](https://www.runpod.io/) account with a payment method on file
  (Serverless GPU workers cost real money per second while running — see
  Cost Control at the bottom before you submit a real job).
- A [Weights & Biases](https://wandb.ai/) account and API key, if you want
  `tracking.wandb.enabled: true` in your config (optional — set it to
  `false` to skip this entirely).

## 1. Create the S3 bucket + IAM user (AWS)

Plain CLI, no Terraform state to manage for something this small:

```bash
BUCKET_NAME=mepc36-radtts-training-artifacts   # must be globally unique across all of AWS
REGION=us-east-1

aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
aws s3api put-bucket-versioning --bucket "$BUCKET_NAME" \
  --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption --bucket "$BUCKET_NAME" \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
aws s3api put-public-access-block --bucket "$BUCKET_NAME" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

Create a least-privilege IAM user scoped to just this bucket — not your
root/admin credentials, so a leaked key's blast radius is "one training
bucket," not your whole AWS account:

```bash
USER_NAME="${BUCKET_NAME}-worker"

aws iam create-user --user-name "$USER_NAME"

cat > /tmp/radtts-bucket-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "ListBucket", "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::${BUCKET_NAME}"},
    {"Sid": "ReadWriteObjects", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], "Resource": "arn:aws:s3:::${BUCKET_NAME}/*"}
  ]
}
EOF

aws iam put-user-policy --user-name "$USER_NAME" \
  --policy-name radtts-training-bucket-access \
  --policy-document file:///tmp/radtts-bucket-policy.json

aws iam create-access-key --user-name "$USER_NAME"
```

That last command prints `AccessKeyId` and `SecretAccessKey` once — save
them now, AWS won't show the secret again.

Update `src/configs/train.yaml`: set `storage.bucket` to `$BUCKET_NAME`.
**Don't** put the access key/secret in the config file itself — they get
set as RunPod runtime environment variables in step 4, never committed to git.

Upload your warmstart checkpoint, vocoder checkpoint/config, and training
data to the new bucket at the paths your config references. These already
exist on disk at `~/Desktop/rapgenie/radtts-assets/` — `cd` there first so
the relative paths below resolve:

```bash
cd ~/Desktop/rapgenie/radtts-assets

aws s3 cp ./radtts++ljs-dap.pt s3://$BUCKET_NAME/pretrained/radtts++ljs-dap.pt
aws s3 cp ./hifigan_libritts100360_generator0p5.pt s3://$BUCKET_NAME/pretrained/hifigan_libritts100360_generator0p5.pt
aws s3 cp ./hifigan_22khz_config.json s3://$BUCKET_NAME/pretrained/hifigan_22khz_config.json
aws s3 sync ./8-formatted-lupe-lines-second-pass-22khz-mono-465/ \
  s3://$BUCKET_NAME/datasets/lupefiasco/8-formatted-lupe-lines-second-pass-22khz-mono-465/
```

If you ever need to re-download these (new machine, deleted local copy),
they're not something we produced — they're NVIDIA's published RADTTS
checkpoints, linked from the fork's own README (`~/Desktop/radtts/README.md`):

- `radtts++ljs-dap.pt` — the "RADTTS++DAP-LJS" checkpoint: https://drive.google.com/file/d/1Rb2VMUwQahGrnpFSlAhCPh7OpDN3xgOr/view
- `hifigan_libritts100360_generator0p5.pt` — HiFi-GAN vocoder checkpoint (LibriTTS 100+360): https://drive.google.com/file/d/1lD62jl5hF6T5AkGoWKOcgMZuMR4Ir76d/view
- `hifigan_22khz_config.json` — that vocoder's config: https://drive.google.com/file/d/1WRtyvkmQxlYShkeTwWmlj7_WiS70R7Jb/view

The training dataset (`8-formatted-lupe-lines-second-pass-22khz-mono-465/`)
is yours, not NVIDIA's — there's no public source for that one; the copy in
`radtts-assets/` is the only copy.

## 2. Build and push the worker image (Docker Hub)

Run from the **rapgenie repo root**. `DOCKERHUB_USERNAME` is read from the
repo-root `.env` (see that file), not passed inline:

```bash
node src/runpod/model-training/scripts/build_worker.js v1.0.0
```

No local smoke test before push right now — kept simple, same as no unit
tests (see "Deploying it for real" in `README.md`). The first real signal on
whether the image works is the actual job in step 4, watched via
`check_job_status.js` / the RunPod console logs. Worth knowing the tradeoff:
a broken image means finding out after a worker's already spun up and
started billing, instead of catching it locally first. If that starts
costing real money, a `test_input.json` fixture + `docker run --test_input`
(RunPod's documented local-testing pattern) is the natural thing to add back.

Push:

```bash
docker push docker.io/$DOCKERHUB_USERNAME/radtts-train-worker:v1.0.0
```

## 3. Create the RunPod Serverless endpoint

The community Terraform provider for RunPod (`decentralized-infrastructure/runpod`)
does support a `runpod_endpoint` resource, but I couldn't fully verify the
exact schema of the `runpod_template` resource it depends on before handing
this to you (the registry docs are JS-rendered and I don't want to hand you
Terraform with guessed attribute names). The console path below is short —
five minutes — and is RunPod's own documented flow. If you want it fully as
code later, verify the template resource schema at
`registry.terraform.io/providers/decentralized-infrastructure/runpod/latest/docs`
first.

1. Go to the [Serverless console](https://www.console.runpod.io/serverless) → **New Endpoint** → **Import from Docker Registry**.
2. Container image: `docker.io/YOUR_DOCKERHUB_USERNAME/radtts-train-worker:v1.0.0`.
3. Endpoint type: **Queue**.
4. GPU: pick based on VRAM needs — RADTTS warmstart training fits comfortably
   on a single 24GB card (RTX 4090 / A5000-class); pick the cheapest tier
   that offers one under **GPU Configuration**.
5. Workers: **min workers = 0** (this is the entire point of serverless —
   pay nothing while idle), **max workers = 1** (you're running one training
   job at a time; raise this only if you're running multiple experiments
   concurrently).
6. Environment variables (Settings tab → Environment Variables — these are
   *runtime* variables, not baked into the image, exactly the distinction
   in RunPod's docs about secrets vs. build-time config):

   | Key | Value |
   |---|---|
   | `AWS_ACCESS_KEY_ID` | the `AccessKeyId` printed by `create-access-key` in step 1 |
   | `AWS_SECRET_ACCESS_KEY` | the `SecretAccessKey` printed by `create-access-key` in step 1 |
   | `WANDB_API_KEY` | from your W&B account settings, only if `tracking.wandb.enabled: true` |

7. Click **Deploy Endpoint**. Copy the **Endpoint ID** from the endpoint's page.

## 4. Submit a real training job

`RUNPOD_API_KEY` / `RUNPOD_ENDPOINT_ID` (from step 3) are read from the
repo-root `.env`, not passed inline. Run from the **rapgenie repo root**:

```bash
node src/runpod/model-training/scripts/submit_training_job.js   # defaults to src/configs/train.yaml
```

This prints a job id. Check on it:

```bash
node src/runpod/model-training/scripts/check_job_status.js <job_id>
```

Status will move through `IN_QUEUE` → (worker cold-starts, can take a
minute or two the first time) → `IN_PROGRESS` → `COMPLETED` or `FAILED`.
Logs are visible in the RunPod console under the endpoint's **Requests** tab.

## 5. Cost control — read this before step 4

RunPod Serverless bills per second of active GPU time. A few things that
matter:

- With `workers_min = 0`, you pay nothing while no job is running. This is
  the actual advantage over the Kubernetes path — no idle GPU node pool.
- The `executionTimeout`/`ttl` policy in `submit_training_job.js` is set to
  a 48-hour execution budget by default. If your run should take 2 hours,
  lower it — a runaway job is a runaway bill, and RunPod's TTL is a hard
  cutoff (see the comment in the script).
- Set `workers_max = 1` so a bug can't accidentally fan out into multiple
  concurrent billed workers.
- To tear down: delete the endpoint from the RunPod console when you're
  done (stops it from accepting new jobs). To remove the AWS side, copy out
  any checkpoints you care about first, then:

  ```bash
  aws s3 rm "s3://$BUCKET_NAME" --recursive
  aws s3api delete-bucket --bucket "$BUCKET_NAME"
  aws iam delete-access-key --user-name "$USER_NAME" --access-key-id <the AccessKeyId from step 1>
  aws iam delete-user-policy --user-name "$USER_NAME" --policy-name radtts-training-bucket-access
  aws iam delete-user --user-name "$USER_NAME"
  ```

## What's real vs. what you still have to do

Real, in this repo, ready to run: the Dockerfile, the handler, the
submit/status scripts, the CLI commands above.

Still requires your action, because it requires your credentials/money and
I don't have or want either: actually running the `aws` commands, actually
building/pushing the image, actually clicking Deploy Endpoint in the RunPod
console, actually submitting a job. Once you've done steps 1-3 once, step 4
(submitting a training job) is the only one you'll repeat.

## When to reach for Terraform/Kubernetes instead

Not now, but worth knowing the trigger conditions, since "knowing when" is
the actual staff-level signal, not "knowing how": codify steps 1-3 as
Terraform once you're recreating this environment often enough that doing
it by hand is error-prone (a second environment, a teammate needs the same
setup, disaster recovery). Reach for Kubernetes instead of RunPod Serverless
if you need multiple concurrent training jobs sharing a fixed GPU fleet with
priority/preemption between them — a single job at a time, on-demand, is
exactly what serverless is for.
