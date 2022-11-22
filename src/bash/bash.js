const { spawn } = require('child_process')
const { appendFileSync } = require('fs')

const execComm = (comm, opts = {}, logFileName) => {
  const { needsData, printLogs, saveLogs } = opts

  return new Promise((resolve, reject) => {
    const terminal = spawn('bash')

    terminal.stdout.on('data', (data) => {
      if (needsData) {
        resolve(data.toString())
      }
      printLogs && console.log('### data:', data)
      saveLogs && appendFileSync(`./output/bash/${logFileName}.txt`, JSON.stringify(data))
    });

    terminal.on('exit', async (code) => {
      if (code) {
        reject(new Error(`Failed running comm:\n${comm}`))
      } else {
        resolve()
      }
    });

    // FFMPEG LOGS ITS PRINTOUTS TO .STDERR, SO WE COMMENT IT OUT: 
    // eslint-disable-next-line no-unused-vars
    terminal.stderr.on('data', (data) => {
      printLogs && console.log(`###########\n${data}`);
      saveLogs && appendFileSync(`./output/bash/${logFileName}.txt`, JSON.stringify(data.toString()))
    });

    terminal.stdin.write(comm, async (err) => {
      if (err) {
        reject(err)
      } else {
        terminal.stdin.end();
      }
    });
  })
}

const execPythonComm = (args, opts = {}) => {
  const { printLogs } = opts
  return new Promise((resolve, reject) => {
    const terminal = spawn('/usr/bin/python3', args)

    // eslint-disable-next-line no-unused-vars
    // ALWAYS MUST BE LEFT ON, OR ELSE PROCESS WON'T EXECUTE FOR SOME REASON:
    terminal.stdout.on('data', (data) => {
      printLogs && console.log('$$$ data:', data.toString())
      console.log('resolve 1')
      // return resolve(data.toString())
    });

    terminal.on('exit', async (code) => {
      if (code) {
        return reject(new Error(`Failed running comm:\n${args}`))
      } else {
        console.log('resolve 4')
        return resolve()
      }
    });

    terminal.stderr.on('data', (error) => {
      console.log('STDERR > error:', error.toString())

      // TODO: actually rethrow errors, i.e.:
      // const isWarning = error.toString().includes('UserWarning')
      // console.log('isWarning:', isWarning)
      // if (!isWarning) {
      // console.log('Rethrowing...')
      // const rethrownError = new Error(error.toString())
      // return reject(rethrownError)
      // }
    });

    terminal.stdin.write('', async (err) => {
      if (err) {
        console.log('resolve 2')
        return reject(new Error(`Error creating file 2\ncode: ${code} signal:${signal}`))
      } else {
        console.log('resolve 3')
        terminal.stdin.end();
      }
    });
  })
}

module.exports = {
  execComm,
  execPythonComm
}
