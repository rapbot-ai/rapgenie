/**
 * Shared local job log — the one place a submitted job's id gets recorded
 * alongside what it actually was, so a job id is never just an opaque UUID
 * you have to remember or cross-reference by memory. Not a runnable script
 * itself (no shebang) — imported by submit_training_job.js,
 * submit_inference_job.js, and list_jobs.js, the same way blob_storage.js is
 * shared rather than duplicated per-worker.
 *
 * RunPod's API has no "list all jobs you've submitted to this endpoint"
 * endpoint — GET /status only works if you already have the job id. This
 * file is what makes that not a problem: every submit appends one line here,
 * list_jobs.js reads it back and cross-references each entry against live
 * status.
 *
 * Format: JSON Lines (one JSON object per submission) at repo-root
 * .runpod-jobs.log — append-only, gitignored (see .gitignore), local state
 * only, never sent anywhere.
 */

const fs = require('fs')
const path = require('path')

const LOG_PATH = path.resolve(__dirname, '../../../.runpod-jobs.log')

/**
 * @param {{ endpoint: 'train' | 'infer', jobId: string, note: string }} entry
 */
const appendJobLogEntry = ({ endpoint, jobId, note }) => {
  const line = JSON.stringify({ timestamp: new Date().toISOString(), endpoint, jobId, note })
  fs.appendFileSync(LOG_PATH, line + '\n')
}

/**
 * @returns {Array<{ timestamp: string, endpoint: string, jobId: string, note: string }>}
 *   Most recent first. Malformed lines (e.g. from an interrupted write) are
 *   skipped rather than throwing — a corrupt log entry shouldn't block you
 *   from seeing every other job.
 */
const readJobLogEntries = () => {
  if (!fs.existsSync(LOG_PATH)) return []
  return fs
    .readFileSync(LOG_PATH, 'utf-8')
    .split('\n')
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line)
      } catch {
        return null
      }
    })
    .filter(Boolean)
    .reverse()
}

module.exports = { LOG_PATH, appendJobLogEntry, readJobLogEntries }
