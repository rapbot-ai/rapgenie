#!/usr/bin/env node
/**
 * Submits a real, asynchronous training job to your deployed RunPod
 * Serverless endpoint via POST /run.
 * https://docs.runpod.io/serverless/endpoints/send-requests
 *
 * Usage (RUNPOD_TRAIN_API_KEY / RUNPOD_TRAIN_ENDPOINT_ID come from the
 * repo-root .env, not passed on the command line — see the dotenv.config()
 * call below):
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
const { appendJobLogEntry } = require('../../scripts/job_log.js')

// Explicit path, not dotenv's default (cwd-relative) lookup — this way it
// always finds the repo-root .env regardless of which directory this script
// is run from, instead of depending on the caller happening to be cd'd there.
require('dotenv').config({ path: path.resolve(__dirname, '../../../../.env') })

// Single source of truth for usage — both --help and every validation error
// below print this same string, so it can't drift out of sync with what the
// script actually accepts. Runnable via `npm run submit:train-job -- <args>`
// or directly with `node submit_training_job.js <args>`.
const USAGE = `Usage: node submit_training_job.js [path/to/train.yaml] [--resume-from <checkpoint>]

Required (env, from repo-root .env):
  RUNPOD_TRAIN_API_KEY       from https://www.runpod.io/console/user/settings
  RUNPOD_TRAIN_ENDPOINT_ID   the TRAINING endpoint's ID (not the inference endpoint's)

Optional (args):
  [path/to/train.yaml]     defaults to src/configs/train.yaml
  --resume-from <ckpt>     e.g. --resume-from model_600 — resumes training from that checkpoint;
                            omit entirely to start a fresh run (resume defaults to false)

Example:
  npm run submit:train-job -- --resume-from model_600`

const rawArgs = process.argv.slice(2)

if (rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.log(USAGE)
  process.exit(0)
}

const { RUNPOD_TRAIN_API_KEY, RUNPOD_TRAIN_ENDPOINT_ID } = process.env

if (!RUNPOD_TRAIN_API_KEY || !RUNPOD_TRAIN_ENDPOINT_ID) {
  console.error('Missing RUNPOD_TRAIN_API_KEY and/or RUNPOD_TRAIN_ENDPOINT_ID in your repo-root .env.\n')
  console.error(USAGE)
  process.exit(1)
}

const resumeFlagIndex = rawArgs.indexOf('--resume-from')
const resumeFrom = resumeFlagIndex !== -1 ? rawArgs[resumeFlagIndex + 1] : null
if (resumeFlagIndex !== -1 && !resumeFrom) {
  console.error('--resume-from requires a checkpoint name, e.g. --resume-from model_600.\n')
  console.error(USAGE)
  process.exit(1)
}
const positionalArgs = rawArgs.filter((_, i) => i !== resumeFlagIndex && i !== resumeFlagIndex + 1)

// Default: scripts -> model-training -> runpod -> src, then down into configs -> src/configs/train.yaml
const defaultConfigPath = path.resolve(__dirname, '../../../configs/train.yaml')
const configPath = positionalArgs[0] || defaultConfigPath

if (!fs.existsSync(configPath)) {
  console.error(`config file not found: ${configPath}\n`)
  console.error(USAGE)
  process.exit(1)
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
  const url = `https://api.runpod.ai/v2/${RUNPOD_TRAIN_ENDPOINT_ID}/run`
  const headers = {
    Authorization: `Bearer ${RUNPOD_TRAIN_API_KEY}`,
    'Content-Type': 'application/json',
  }
  const payload = buildPayload(configYaml, resumeFrom)

  console.log(`Submitting ${configPath} to ${url}...`)
  const { data } = await axios.post(url, payload, { headers })
  console.log(JSON.stringify(data, null, 2))

  if (data.id) {
    console.log(`\nJob id: ${data.id}`)
    console.log(`Check status with: npm run check:train-status -- ${data.id}`)
    console.log(`Kill it with: npm run kill:train-job -- ${data.id}`)
  }
}

if (require.main === module) {
  submitJob().catch((error) => {
    console.error('Failed to submit training job:', error.response ? error.response.data : error.message)
    process.exit(1)
  })
}

module.exports = { buildPayload }
