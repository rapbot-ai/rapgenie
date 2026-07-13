#!/usr/bin/env node
/**
 * Builds the RunPod INFERENCE worker image, per RunPod's documented deploy
 * flow: https://docs.runpod.io/serverless/workers/deploy
 *
 * Mirrors src/runpod/model-training/scripts/build_worker.js (training's
 * version) exactly, just pointed at the inference Dockerfile and a different
 * image name — this is a separate endpoint, not the same image. See
 * ../Dockerfile for why.
 *
 * Usage:
 *   export DOCKERHUB_USERNAME=your-username
 *   node build_worker.js v1.0.0
 *
 * Named build_worker, not build_and_push: it does NOT push automatically —
 * it builds, then prints the exact test/push commands so you can look at
 * the image before it leaves your machine and decide yourself when to push.
 *
 * Reuses execComm from ../../../bash/bash.js — the same subprocess-spawning
 * helper rapgenie's own RADTTS inference calls already use — instead of
 * introducing a second way to shell out to a command.
 */

const path = require('path')
const { execComm } = require('../../../bash/bash.js')
// Explicit path, not dotenv's default (cwd-relative) lookup — this way it
// finds the repo-root .env regardless of which directory you run this
// script from, instead of only working when you happen to be cd'd there.
require('dotenv').config({ path: path.resolve(__dirname, '../../../../.env') })

const { DOCKERHUB_USERNAME } = process.env
const version = process.argv[2]

if (!DOCKERHUB_USERNAME) {
  throw new Error('Set DOCKERHUB_USERNAME in your environment')
}
if (!version) {
  throw new Error('Usage: node build_worker.js <version-tag>, e.g. v1.0.0')
}

// scripts -> inference -> runpod -> src -> rapgenie (repo root, the Docker build context)
const repoRoot = path.resolve(__dirname, '../../../..')
const dockerfile = path.join(repoRoot, 'src/runpod/inference/Dockerfile')
const image = `docker.io/${DOCKERHUB_USERNAME}/radtts-infer-worker:${version}`

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
