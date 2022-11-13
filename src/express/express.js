require('dotenv').config()
const express = require('express')
const bodyParser = require('body-parser');
const CORS = require('cors');
const { execPythonComm } = require('../bash/bash')

const app = express()

app.use(CORS());
app.use(bodyParser.json({ strict: false, limit: '50mb' }));

app.get('/', async (req, res) => {
  return res.send('Hello world!')
})

app.post(`/infer`, async (req, res) => {
  try {
    const inferPath = `/home/ubuntu/1-radtts-repo/inference.py`
    const modelConfigPath = `-c /home/ubuntu/1-radtts-repo/2-configs/1-radtts-configs/config_ljs_dap.json`
    const modelPath = `-r /home/ubuntu/1-radtts-repo/1-models/1-radtts-models/lupe-fiasco-radtts-model`
    const vocoderPath = `-v /home/ubuntu/1-radtts-repo/1-models/2-hifigan-models/hifigan_libritts100360_generator0p5.pt`
    const vocoderConfigPath = `-k /home/ubuntu/1-radtts-repo/2-configs/2-hifigan-configs/hifigan_22khz_config.json`
    const textInput = `-t /home/ubuntu/1-radtts-repo/5-tts-input-text/catch-me-like-spalding.txt`
    const speaker = `-s lupefiasco`
    const speakerAttributes = `--speaker_attributes lupefiasco`
    const speakerText = `--speaker_text lupefiasco`
    const outputDir = `-o /home/ubuntu/1-radtts-repo/6-training-output`
    const radttsInferCommand = `${inferPath} ${modelConfigPath} ${modelPath} ${vocoderPath} ${vocoderConfigPath} ${textInput} ${speaker} ${speakerAttributes} ${speakerText} ${outputDir}`.split(' ')
    const result = await execPythonComm(radttsInferCommand, { printLogs: true })
    console.log('radttsInferCommand:', [`python3`, inferPath, modelConfigPath, modelPath, vocoderPath, vocoderConfigPath, textInput, speaker, speakerAttributes, speakerText, outputDir].join(` \\\n`))
    return res.send(result)
  } catch (error) {
    console.log('error:', error)
    const stringifiedError = JSON.stringify(error, Object.getOwnPropertyNames(error))
    res.status(500).send(stringifiedError)
  }
})

app.listen(3020, async () => {
  try {
    console.log('listening on 3020')
  } catch (error) {
    console.error('app.listen error:', error)
    console.log(error)
  }
})
