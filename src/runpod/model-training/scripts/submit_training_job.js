#!/usr/bin/env node
/**
 * Submits a real, asynchronous training job to your deployed RunPod
 * Serverless endpoint via POST /run.
 * https://docs.runpod.io/serverless/endpoints/send-requests
 *
 * Usage (RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID come from the repo-root .env,
 * not passed on the command line — see the dotenv.config() call below):
 *   node submit_training_job.js [path/to/train.yaml] [--resume-from model_600]
 *
 * Rewritten from bash on purpose: the original shelled out to `python3 -c
 * "import json; ..."` just to escape the config file's text into a JSON
 * string, then built the whole request body as a bash heredoc. Here that's
 * one call to JSON.stringify() and a real JS object — no second language
 * standing in for structured data.
 */

const fs = require('fs')
const path = require('path')
const axios = require('axios')

// Explicit path, not dotenv's default (cwd-relative) lookup — this way it
// always finds the repo-root .env regardless of which directory this script
// is run from, instead of depending on the caller happening to be cd'd there.
require('dotenv').config({ path: path.resolve(__dirname, '../../../../.env') })

const { RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID } = process.env

if (!RUNPOD_API_KEY) {
  throw new Error('Set RUNPOD_API_KEY in your environment')
}
if (!RUNPOD_ENDPOINT_ID) {
  throw new Error('Set RUNPOD_ENDPOINT_ID in your environment (from the RunPod console)')
}

const rawArgs = process.argv.slice(2)
const resumeFlagIndex = rawArgs.indexOf('--resume-from')
const resumeFrom = resumeFlagIndex !== -1 ? rawArgs[resumeFlagIndex + 1] : null
if (resumeFlagIndex !== -1 && !resumeFrom) {
  throw new Error('--resume-from requires a checkpoint name, e.g. --resume-from model_600')
}
const positionalArgs = rawArgs.filter((_, i) => i !== resumeFlagIndex && i !== resumeFlagIndex + 1)

// Default: scripts -> model-training -> runpod -> src, then down into configs -> src/configs/train.yaml
const defaultConfigPath = path.resolve(__dirname, '../../../configs/train.yaml')
const configPath = positionalArgs[0] || defaultConfigPath

if (!fs.existsSync(configPath)) {
  throw new Error(`config file not found: ${configPath}`)
}

const configYaml = fs.readFileSync(configPath, 'utf-8')

// executionTimeout / ttl: training runs far longer than RunPod's defaults
// (10 min execution timeout, 24h ttl) — see
// https://docs.runpod.io/serverless/endpoints/send-requests#long-running-jobs
// 172800000 ms = 48h execution budget, 259200000 ms = 72h total lifespan
// (48h execution + 24h headroom for queue time). Adjust to your actual
// expected training duration.
const buildPayload = (configYamlText, resumeFromCheckpoint = null) => ({
  input: {
    config_yaml: configYamlText,
    resume: Boolean(resumeFromCheckpoint),
    resume_from: resumeFromCheckpoint,
  },
  policy: {
    executionTimeout: 172800000,
    ttl: 259200000,
  },
})

const submitJob = async () => {
  const url = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`
  const headers = {
    Authorization: `Bearer ${RUNPOD_API_KEY}`,
    'Content-Type': 'application/json',
  }
  const payload = buildPayload(configYaml, resumeFrom)

  console.log(`Submitting ${configPath} to ${url}...`)
  const { data } = await axios.post(url, payload, { headers })
  console.log(JSON.stringify(data, null, 2))

  if (data.id) {
    console.log(`\nJob id: ${data.id}`)
    console.log(`Check status with: node check_job_status.js ${data.id}`)
  }
}

if (require.main === module) {
  submitJob().catch((error) => {
    console.error('Failed to submit training job:', error.response ? error.response.data : error.message)
    process.exit(1)
  })
}

module.exports = { buildPayload }
