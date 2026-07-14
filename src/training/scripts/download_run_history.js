#!/usr/bin/env node
/**
 * Downloads a W&B run's logged data to local files — history (every metric
 * logged over every step), config, and summary — instead of just printing a
 * sampled preview in a notebook cell.
 *
 * JS rewrite of the original download_run_history.py. Reason for the
 * rewrite: that script needed its own local Python venv (wandb/pandas/pyyaml),
 * and just getting it importable burned two rounds of real debugging —
 * wandb 0.17.0 crashing on numpy 2.0's removed np.float_, then protobuf's
 * C-extension not supporting Python 3.14 at all. Every other local dev
 * script in this repo is already Node (this repo's actual runtime), so this
 * sidesteps that whole class of problem — no venv, no C-extension
 * compatibility roulette, just `npm install`.
 *
 * wandb has no official Node SDK with read/history support — @wandb/sdk
 * (github.com/wandb/wandb-js) is write-only (init/log/finish), last released
 * 2023. So this talks to wandb's GraphQL API directly
 * (https://api.wandb.ai/graphql, HTTP Basic auth with username "api" and
 * WANDB_API_KEY as the password) — the same API the Python SDK's
 * wandb.Api()/run.scan_history() call under the hood.
 *
 * Usage (from the rapgenie repo root):
 *   node src/training/scripts/download_run_history.js <run_id> [--project <name>] [--entity <name>] [--output-dir <path>] [--sampled]
 * or:
 *   npm run wandb:history -- <run_id>
 *
 * Auth (WANDB_API_KEY) and --entity's default both come from the repo-root
 * .env, the same file every RunPod script already reads from.
 *
 * Always writes to <repo-root>/logs/weights-and-biases/<run_id>/
 * (history.csv + history.meta.json) unless --output-dir overrides it.
 * (logs/ is gitignored — these are local exports, not checked-in artifacts.)
 *
 * Full vs. sampled history: same reasoning as the Python version. wandb's
 * `history` field silently downsamples when you ask for fewer samples than
 * there are steps in the range you query — fine for eyeballing a chart,
 * wrong for a "download the data" script, since a 10,000-iteration run
 * would quietly come back as 500 rows with no error. This script pages
 * through history in fixed-size windows, asking for as many samples as
 * steps in each window (so nothing gets downsampled), and defaults to that
 * full/unsampled mode. --sampled instead does one unpaginated query across
 * the whole run capped at 500 samples — fast, capped, wandb decides which
 * steps you get.
 */

'use strict'

const fs = require('fs')
const path = require('path')
const axios = require('axios')
const yaml = require('js-yaml')

const REPO_ROOT = path.resolve(__dirname, '../../../')
const TRAIN_YAML = path.resolve(__dirname, '../../configs/train.yaml')

// Your own example command used this entity — kept as the built-in fallback
// so you don't have to retype it every run, but --entity or WANDB_ENTITY (in
// .env) always override it.
const DEFAULT_ENTITY = 'rapbot-ai'

const WANDB_GRAPHQL_URL = 'https://api.wandb.ai/graphql'

// Single source of truth for usage — both --help and every validation error
// below print this same string.
const USAGE = `Usage: node download_run_history.js <run_id> [--project <name>] [--entity <name>] [--output-dir <path>] [--sampled]
  or: npm run wandb:history -- <run_id>

Required:
  <run_id>   the run's id, e.g. lupefiasco-radtts-warmstart-v5-86f7089a5b26-77e3051
             (read it off the W&B run you want, or the run_id= line train.py logs at startup)

Required (env, from repo-root .env):
  WANDB_API_KEY   from https://wandb.ai/authorize

Optional (args):
  --project <name>      defaults to tracking.wandb.project in src/configs/train.yaml
  --entity <name>       defaults to $WANDB_ENTITY (.env) if set, else ${DEFAULT_ENTITY}
  --output-dir <path>   defaults to <repo-root>/logs/weights-and-biases/<run_id>/
  --sampled              use wandb's fast, capped-at-500-points history instead of the full,
                          unsampled, paginated scan

Example:
  npm run wandb:history -- lupefiasco-radtts-warmstart-v5-86f7089a5b26-77e3051`

// Explicit path, not dotenv's default (cwd-relative) lookup — this way it
// always finds the repo-root .env regardless of which directory this script
// is run from.
require('dotenv').config({ path: path.join(REPO_ROOT, '.env') })

function parseArgs(argv) {
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(USAGE)
    process.exit(0)
  }

  const flagValue = (flag) => {
    const i = argv.indexOf(flag)
    if (i === -1) return undefined
    const value = argv[i + 1]
    if (!value || value.startsWith('--')) {
      console.error(`${flag} requires a value.\n`)
      console.error(USAGE)
      process.exit(1)
    }
    return value
  }

  const sampled = argv.includes('--sampled')
  const project = flagValue('--project')
  const entity = flagValue('--entity')
  const outputDir = flagValue('--output-dir')

  const consumed = new Set(['--sampled'])
  ;[['--project', project], ['--entity', entity], ['--output-dir', outputDir]].forEach(([flag, value]) => {
    if (value !== undefined) {
      consumed.add(flag)
      consumed.add(value)
    }
  })
  const positional = argv.filter((a) => !consumed.has(a))

  const runId = positional[0]
  if (!runId) {
    console.error('Missing required <run_id>.\n')
    console.error(USAGE)
    process.exit(1)
  }

  return { runId, project, entity, outputDir, sampled }
}

function defaultProject() {
  if (!fs.existsSync(TRAIN_YAML)) return undefined
  const raw = yaml.load(fs.readFileSync(TRAIN_YAML, 'utf-8')) || {}
  return ((raw.tracking || {}).wandb || {}).project
}

async function graphql(apiKey, query, variables) {
  const { data } = await axios.post(
    WANDB_GRAPHQL_URL,
    { query, variables },
    { auth: { username: 'api', password: apiKey }, headers: { 'Content-Type': 'application/json' } }
  )
  if (data.errors && data.errors.length > 0) {
    throw new Error(`wandb GraphQL error: ${data.errors.map((e) => e.message).join('; ')}`)
  }
  return data.data
}

const RUN_META_QUERY = `
  query Run($project: String!, $entity: String!, $name: String!) {
    project(name: $project, entityName: $entity) {
      run(name: $name) {
        name
        displayName
        state
        createdAt
        config
        summaryMetrics
      }
    }
  }
`

const RUN_HISTORY_PAGE_QUERY = `
  query RunHistoryPage($project: String!, $entity: String!, $name: String!, $minStep: Int64!, $maxStep: Int64!, $samples: Int!) {
    project(name: $project, entityName: $entity) {
      run(name: $name) {
        history(minStep: $minStep, maxStep: $maxStep, samples: $samples)
      }
    }
  }
`

async function fetchRunMeta(apiKey, entity, project, runId) {
  const result = await graphql(apiKey, RUN_META_QUERY, { project, entity, name: runId })
  const run = result && result.project && result.project.run
  if (!run) {
    throw new Error(
      `run not found: ${entity}/${project}/${runId} — check --entity/--project, or that this run_id exists in that project`
    )
  }
  return {
    runId,
    name: run.displayName || run.name,
    state: run.state,
    createdAt: run.createdAt,
    config: JSON.parse(run.config || '{}'),
    summary: JSON.parse(run.summaryMetrics || '{}'),
  }
}

// Full, unsampled: page through in fixed-size step windows, asking for as
// many samples as there are steps in each window — the "samples" parameter
// only downsamples when you ask for fewer samples than the window contains,
// so a window-sized samples count guarantees every row in that window comes
// back untouched. Stops once a page comes back empty.
async function scanHistoryFull(apiKey, entity, project, runId, lastStep, pageSize) {
  const rows = []
  let offset = 0
  const upperBound = lastStep != null ? lastStep + 1 : Infinity

  while (offset < upperBound) {
    const maxStep = Math.min(offset + pageSize, upperBound === Infinity ? offset + pageSize : upperBound)
    const result = await graphql(apiKey, RUN_HISTORY_PAGE_QUERY, {
      project,
      entity,
      name: runId,
      minStep: offset,
      maxStep,
      samples: pageSize,
    })
    const run = result && result.project && result.project.run
    if (!run) {
      throw new Error(`run not found while paging history: ${entity}/${project}/${runId}`)
    }
    const pageRows = (run.history || []).map((r) => JSON.parse(r))
    if (pageRows.length === 0) break
    rows.push(...pageRows)
    offset += pageSize
    if (upperBound === Infinity && pageRows.length < pageSize) break // no lastStep known — stop once a short page signals the end
  }
  return rows
}

// Sampled/capped: one query spanning the whole run, samples capped — wandb
// downsamples server-side across that whole range. Same query shape as the
// full scan, just unpaginated with a wide window.
async function scanHistorySampled(apiKey, entity, project, runId, lastStep, sampleCount) {
  const maxStep = lastStep != null ? lastStep + 1 : 1000000
  const result = await graphql(apiKey, RUN_HISTORY_PAGE_QUERY, {
    project,
    entity,
    name: runId,
    minStep: 0,
    maxStep,
    samples: sampleCount,
  })
  const run = result && result.project && result.project.run
  if (!run) {
    throw new Error(`run not found while fetching sampled history: ${entity}/${project}/${runId}`)
  }
  return (run.history || []).map((r) => JSON.parse(r))
}

function rowsToCsv(rows) {
  // Union of keys across every row, in first-seen order (mirrors how the
  // metrics naturally appear step to step, rather than an arbitrary sort).
  const keys = []
  const seen = new Set()
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key)
        keys.push(key)
      }
    }
  }

  const escape = (value) => {
    if (value === undefined || value === null) return ''
    const s = String(value)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }

  const lines = [keys.map(escape).join(',')]
  for (const row of rows) {
    lines.push(keys.map((k) => escape(row[k])).join(','))
  }
  return lines.join('\n') + '\n'
}

async function main() {
  const { runId, project: projectArg, entity: entityArg, outputDir: outputDirArg, sampled } = parseArgs(
    process.argv.slice(2)
  )

  const apiKey = process.env.WANDB_API_KEY
  if (!apiKey) {
    console.error('Missing WANDB_API_KEY in your repo-root .env.\n')
    console.error(USAGE)
    process.exit(1)
  }

  const entity = entityArg || process.env.WANDB_ENTITY || DEFAULT_ENTITY
  const project = projectArg || defaultProject()
  if (!project) {
    console.error(
      `Couldn't determine --project (no tracking.wandb.project in ${TRAIN_YAML}). Pass --project explicitly.\n`
    )
    console.error(USAGE)
    process.exit(1)
  }

  const runPath = `${entity}/${project}/${runId}`
  console.log(`Fetching ${runPath} ...`)
  const meta = await fetchRunMeta(apiKey, entity, project, runId)

  const lastStep = typeof meta.summary._step === 'number' ? meta.summary._step : null

  const rows = sampled
    ? await scanHistorySampled(apiKey, entity, project, runId, lastStep, 500)
    : await scanHistoryFull(apiKey, entity, project, runId, lastStep, 1000)

  const outputDir = outputDirArg ? path.resolve(outputDirArg) : path.join(REPO_ROOT, 'logs', 'weights-and-biases', runId)
  fs.mkdirSync(outputDir, { recursive: true })

  const historyPath = path.join(outputDir, 'history.csv')
  fs.writeFileSync(historyPath, rowsToCsv(rows))
  console.log(`Wrote ${rows.length} rows -> ${historyPath}`)

  const metaPath = path.join(outputDir, 'history.meta.json')
  fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2))
  console.log(`Wrote run config/summary -> ${metaPath}`)
}

if (require.main === module) {
  main().catch((error) => {
    console.error('Failed to download run history:', error.response ? error.response.data : error.message)
    process.exit(1)
  })
}

module.exports = { buildRunPath: (entity, project, runId) => `${entity}/${project}/${runId}`, rowsToCsv }
