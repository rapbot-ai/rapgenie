#!/usr/bin/env node
/**
 * Submits a real inference job to your deployed RunPod Serverless INFERENCE
 * endpoint via POST /run. https://docs.runpod.io/serverless/endpoints/send-requests
 *
 * Usage:
 *   export RUNPOD_API_KEY=...        # from https://www.runpod.io/console/user/settings
 *   export RUNPOD_ENDPOINT_ID=...    # the INFERENCE endpoint's ID — NOT the training
 *                                     # endpoint's. Same env var name as
 *                                     # submit_training_job.js uses, different value:
 *                                     # these are two separate RunPod endpoints.
 *   node submit_inference_job.js "text to synthesize" --checkpoint checkpoints/<run>/model_10800 [--speaker lupefiasco] [--tempo 1.5]
 *
 * checkpoint_s3_key has no default on purpose (see src/inference/config.py's
 * docstring) — you always say exactly which checkpoint you're running
 * against, copy-pasted from wherever you tracked the training run's output.
 */

const axios = require('axios')
require('dotenv').config()

const { RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID } = process.env

if (!RUNPOD_API_KEY) {
  throw new Error('Set RUNPOD_API_KEY in your environment')
}
if (!RUNPOD_ENDPOINT_ID) {
  throw new Error('Set RUNPOD_ENDPOINT_ID in your environment (the inference endpoint\'s ID)')
}

const rawArgs = process.argv.slice(2)

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

if (!text) {
  throw new Error(
    'Usage: node submit_inference_job.js "<text>" --checkpoint <s3-key> [--speaker <name>] [--tempo <n>]'
  )
}
if (!checkpoint) {
  throw new Error(
    '--checkpoint <s3-key> is required, e.g. checkpoints/lupefiasco-radtts-warmstart-v5/<run_id>/model_10800'
  )
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
  const url = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`
  const headers = {
    Authorization: `Bearer ${RUNPOD_API_KEY}`,
    'Content-Type': 'application/json',
  }
  const payload = buildPayload(text, checkpoint, { speaker, tempo })

  console.log(`Submitting inference request to ${url}...`)
  console.log(JSON.stringify(payload, null, 2))
  const { data } = await axios.post(url, payload, { headers })
  console.log(JSON.stringify(data, null, 2))

  if (data.id) {
    console.log(`\nJob id: ${data.id}`)
    console.log(`Check status with: node ../../runpod/scripts/check_job_status.js ${data.id}`)
  }
}

if (require.main === module) {
  submitJob().catch((error) => {
    console.error('Failed to submit inference job:', error.response ? error.response.data : error.message)
    process.exit(1)
  })
}

module.exports = { buildPayload }
