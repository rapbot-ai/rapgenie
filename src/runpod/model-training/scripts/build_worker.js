#!/usr/bin/env node
/**
 * Builds the RunPod worker image, per RunPod's documented deploy flow:
 * https://docs.runpod.io/serverless/workers/deploy
 *
 * Usage:
 *   export DOCKERHUB_USERNAME=your-username
 *   node build_worker.js v1.0.0
 *
 * Named build_worker, not build_and_push: it does NOT push automatically
 * (same behavior as the original bash script this was ported from, which is
 * where the old build_and_push name came from) — it builds, then prints the
 * exact test/push commands so you can look at the image before it leaves
 * your machine and decide yourself when to push.
 *
 * Reuses execComm from ../../../bash/bash.js — the same subprocess-spawning
 * helper rapgenie's own RADTTS inference calls already use — instead of
 * introducing a second way to shell out to a command.
 */

const path = require('path')
const { execComm } = require('../../../bash/bash.js')
// Explicit path, not dotenv's default (cwd-relative) lookup — this way it
// always finds the repo-root .env regardless of which directory this script
// is run from, instead of depending on the caller happening to be cd'd there.
require('dotenv').config({ path: path.resolve(__dirname, '../../../../.env') })

// Single source of truth for usage — both --help and every validation error
// below print this same string, so it can't drift out of sync with what the
// script actually accepts. Runnable via `npm run build:train -- v1.0.0` or
// directly with `node build_worker.js v1.0.0`.
const USAGE = `Usage: node build_worker.js <version-tag>

Required (env, from repo-root .env):
  DOCKERHUB_USERNAME   your Docker Hub username, e.g. martinconnor

Required (args):
  <version-tag>        e.g. v1.0.0 — avoid :latest, see src/training/RUNBOOK.md for why

Example:
  npm run build:train -- v1.0.5`

const rawArgs = process.argv.slice(2)

if (rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.log(USAGE)
  process.exit(0)
}

const { DOCKERHUB_USERNAME } = process.env
const version = rawArgs[0]

if (!DOCKERHUB_USERNAME) {
  console.error('Missing DOCKERHUB_USERNAME in your repo-root .env.\n')
  console.error(USAGE)
  process.exit(1)
}
if (!version) {
  console.error('Missing required <version-tag> argument.\n')
  console.error(USAGE)
  process.exit(1)
}

// scripts -> model-training -> runpod -> src -> rapgenie (repo root, the Docker build context)
const repoRoot = path.resolve(__dirname, '../../../..')
const dockerfile = path.join(repoRoot, 'src/runpod/model-training/Dockerfile')
const image = `docker.io/${DOCKERHUB_USERNAME}/radtts-train-worker:${version}`

const buildDockerBuildCommand = () =>
  `docker build --platform linux/amd64 -f ${dockerfile} -t ${image} ${repoRoot}`

const main = async () => {
  const buildCommand = buildDockerBuildCommand()
  console.log(`Building ${image} (context: ${repoRoot})...`)
  console.log(buildCommand)
  await execComm(buildCommand, { printLogs: true })

  console.log('')
  console.log('Built. Test locally before pushing:')
  console.log(`  docker run -it ${image}`)
  console.log('')
  console.log('To push:')
  console.log(`  docker push ${image}`)
  console.log('')
  console.log('Avoid :latest for production — see src/training/RUNBOOK.md for why.')
}

if (require.main === module) {
  main().catch((error) => {
    console.error('Build failed:', error.message || error)
    process.exit(1)
  })
}

module.exports = { buildDockerBuildCommand }
