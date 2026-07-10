#!/usr/bin/env node
/**
 * Checks the status of a job submitted via submit_training_job.js, using
 * GET /status/{job_id}.
 * https://docs.runpod.io/serverless/endpoints/send-requests
 *
 * Usage:
 *   export RUNPOD_API_KEY=...
 *   export RUNPOD_ENDPOINT_ID=...
 *   node check_job_status.js <job_id>
 */

const axios = require('axios')

const { RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID } = process.env
const jobId = process.argv[2]

if (!RUNPOD_API_KEY) {
  throw new Error('Set RUNPOD_API_KEY in your environment')
}
if (!RUNPOD_ENDPOINT_ID) {
  throw new Error('Set RUNPOD_ENDPOINT_ID in your environment')
}
if (!jobId) {
  throw new Error('Usage: node check_job_status.js <job_id>')
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
