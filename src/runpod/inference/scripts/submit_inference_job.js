#!/usr/bin/env node
/**
 * Submits a real inference job to your deployed RunPod Serverless INFERENCE
 * endpoint via POST /run. https://docs.runpod.io/serverless/endpoints/send-requests
 *
 * Usage (RUNPOD_INFER_API_KEY / RUNPOD_INFER_ENDPOINT_ID come from the
 * repo-root .env — a separate pair from RUNPOD_TRAIN_API_KEY /
 * RUNPOD_TRAIN_ENDPOINT_ID that submit_training_job.js uses, so both
 * credentials can be live in .env at once with no manual toggling):
 *   node submit_inference_job.js "text to synthesize" --checkpoint checkpoints/<run>/model_10800 [--speaker lupefiasco] [--tempo 1.5]
 *
 * checkpoint_s3_key has no default on purpose (see src/inference/config.py's
 * docstring) — you always say exactly which checkpoint you're running
 * against, copy-pasted from wherever you tracked the training run's output.
 */

const path = require('path')
const axios = require('axios');
const e = require('cors');
// Explicit path, not dotenv's default (cwd-relative) lookup — this way it
// finds the repo-root .env regardless of which directory you run this
// script from, instead of only working when you happen to be cd'd there.
require('dotenv').config({ path: path.resolve(__dirname, '../../../../.env') })

// Single source of truth for usage — both --help and every validation error
// below print this same string, so it can't drift out of sync with what the
// script actually accepts. Runnable via `npm run submit:infer-job -- <args>`
// or directly with `node submit_inference_job.js <args>`.
const USAGE = `Usage: node submit_inference_job.js "<text>" --checkpoint <s3-key> [--speaker <name>] [--tempo <n>]

Required (env, from repo-root .env):
  RUNPOD_INFER_API_KEY       from https://www.runpod.io/console/user/settings
  RUNPOD_INFER_ENDPOINT_ID   the INFERENCE endpoint's ID (not the training endpoint's)

Required (args):
  "<text>"             positional, the text to synthesize
  --checkpoint <key>   full S3 key, e.g. checkpoints/lupefiasco-radtts-warmstart-v5/<run_id>/model_10800

Optional (args):
  --speaker <name>     defaults to whatever InferenceConfig defaults to (see src/inference/config.py)
  --tempo <n>          token_dur_scaling, e.g. 1.5

Example:
  npm run submit:infer-job -- "let's go" --checkpoint checkpoints/lupefiasco-radtts-warmstart-v5/<run_id>/model_10800 --tempo 1.5`

const rawArgs = process.argv.slice(2)

if (rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.log(USAGE)
  process.exit(0)
}

const { RUNPOD_INFER_API_KEY, RUNPOD_INFER_ENDPOINT_ID } = process.env

if (!RUNPOD_INFER_API_KEY || !RUNPOD_INFER_ENDPOINT_ID) {
  console.error('Missing RUNPOD_INFER_API_KEY and/or RUNPOD_INFER_ENDPOINT_ID in your repo-root .env.\n')
  console.error(USAGE)
  process.exit(1)
}

const flagValue = (flag) => {
  const i = rawArgs.indexOf(flag)
  return i !== -1 ? rawArgs[i + 1] : null
}
const consumedIndices = new Set()
  ;['--checkpoint', '--speaker', '--tempo'].forEach((flag) => {
    const i = rawArgs.indexOf(flag)
    if (i !== -1) {
      consumedIndices.add(i)
      consumedIndices.add(i + 1)
    }
  })
const positionalArgs = rawArgs.filter((_, i) => !consumedIndices.has(i))

const text = positionalArgs[0]
const checkpoint = flagValue('--checkpoint')
const speaker = flagValue('--speaker')
const tempo = flagValue('--tempo')

if (!text || !checkpoint) {
  console.error(!text ? 'Missing required "<text>" argument.\n' : 'Missing required --checkpoint <s3-key>.\n')
  console.error(USAGE)
  process.exit(1)
}

const buildPayload = (text, checkpoint, { speaker, tempo } = {}) => ({
  input: {
    text,
    checkpoint_s3_key: checkpoint,
    ...(speaker ? { speaker } : {}),
    ...(tempo ? { token_dur_scaling: Number(tempo) } : {}),
  },
})

const submitJob = async () => {
  const url = `https://api.runpod.ai/v2/${RUNPOD_INFER_ENDPOINT_ID}/run`
  const headers = {
    Authorization: `Bearer ${RUNPOD_INFER_API_KEY}`,
    'Content-Type': 'application/json',
  }
  const payload = buildPayload(text, checkpoint, { speaker, tempo })

  console.log(`Submitting inference request to ${url}...`)
  console.log(JSON.stringify(payload, null, 2))
  const { data } = await axios.post(url, payload, { headers })
  console.log(JSON.stringify(data, null, 2))

  if (data.id) {
    console.log(`\nJob id: ${data.id}`)
    console.log(`\nCheck status with:\nnpm run check:infer-status -- ${data.id}`)
    console.log(`\nKill it with:\nnpm run kill:infer-job -- ${data.id}\n`)
  }
}

if (require.main === module) {
  submitJob().catch((error) => {
    console.error('Failed to submit inference job:', error.response ? error.response.data : error.message)
    process.exit(1)
  })
}

module.exports = { buildPayload }
