#!/usr/bin/env node
/**
 * Checks the status of a job submitted via submit_training_job.js or
 * submit_inference_job.js, using GET /status/{job_id}.
 * https://docs.runpod.io/serverless/endpoints/send-requests
 *
 * Shared between both workers on purpose — it's just a GET against RunPod's
 * generic job-status endpoint, nothing training- or inference-specific about
 * it, so it lives at src/runpod/scripts/ rather than duplicated (or picked
 * arbitrarily) into one of the two worker subdirs. Which credentials it uses
 * is never inferred or defaulted — you say --endpoint train or --endpoint
 * infer explicitly, same "explicit over implicit" rule as
 * checkpoint_s3_key having no default in src/inference/config.py.
 *
 * Usage:
 *   node check_job_status.js <job_id> --endpoint train|infer
 */

const path = require('path')
const axios = require('axios')
// Explicit path, not dotenv's default (cwd-relative) lookup — this way it
// finds the repo-root .env regardless of which directory you run this
// script from, instead of only working when you happen to be cd'd there.
require('dotenv').config({ path: path.resolve(__dirname, '../../../.env') })

// Single source of truth for usage — both --help and every validation error
// below print this same string, so it can't drift out of sync with what the
// script actually accepts. Runnable via `npm run check:train-status -- <job_id>`
// / `npm run check:infer-status -- <job_id>`, or directly with
// `node check_job_status.js <job_id> --endpoint train|infer`.
const USAGE = `Usage: node check_job_status.js <job_id> --endpoint train|infer

Required (env, from repo-root .env):
  --endpoint train   uses RUNPOD_TRAIN_API_KEY / RUNPOD_TRAIN_ENDPOINT_ID
  --endpoint infer   uses RUNPOD_INFER_API_KEY / RUNPOD_INFER_ENDPOINT_ID

Required (args):
  <job_id>             printed by submit_training_job.js / submit_inference_job.js on submit
  --endpoint <which>   train or infer — no default, always explicit about which endpoint's job this is

Example:
  npm run check:train-status -- abc123
  npm run check:infer-status -- abc123`

const rawArgs = process.argv.slice(2)

if (rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.log(USAGE)
  process.exit(0)
}

const endpointFlagIndex = rawArgs.indexOf('--endpoint')
const endpoint = endpointFlagIndex !== -1 ? rawArgs[endpointFlagIndex + 1] : null
const jobId = rawArgs.filter((_, i) => i !== endpointFlagIndex && i !== endpointFlagIndex + 1)[0]

if (endpoint !== 'train' && endpoint !== 'infer') {
  console.error(`Missing or invalid --endpoint (got ${JSON.stringify(endpoint)}) — must be exactly "train" or "infer".\n`)
  console.error(USAGE)
  process.exit(1)
}
if (!jobId) {
  console.error('Missing required <job_id> argument.\n')
  console.error(USAGE)
  process.exit(1)
}

const RUNPOD_API_KEY = endpoint === 'train' ? process.env.RUNPOD_TRAIN_API_KEY : process.env.RUNPOD_INFER_API_KEY
const RUNPOD_ENDPOINT_ID = endpoint === 'train' ? process.env.RUNPOD_TRAIN_ENDPOINT_ID : process.env.RUNPOD_INFER_ENDPOINT_ID

if (!RUNPOD_API_KEY || !RUNPOD_ENDPOINT_ID) {
  console.error(`Missing RUNPOD_${endpoint.toUpperCase()}_API_KEY and/or RUNPOD_${endpoint.toUpperCase()}_ENDPOINT_ID in your repo-root .env.\n`)
  console.error(USAGE)
  process.exit(1)
}

const checkStatus = async () => {
  const url = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${jobId}`
  const headers = { Authorization: `Bearer ${RUNPOD_API_KEY}` }

  const { data } = await axios.get(url, { headers })
  console.log(JSON.stringify(data, null, 2))
}

if (require.main === module) {
  checkStatus().catch((error) => {
    console.error('Failed to check job status:', error.response ? error.response.data : error.message)
    process.exit(1)
  })
}
