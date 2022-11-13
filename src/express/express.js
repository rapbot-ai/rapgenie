require('dotenv').config()
const express = require('express')
const bodyParser = require('body-parser');
const CORS = require('cors');
const { v4 } = require('uuid')
const { writeFileSync, rmSync, mkdirSync, existsSync, createReadStream } = require('fs')
const { uploadToS3, s3Client } = require('../aws/aws')
const { execPythonComm } = require('../bash/bash')
const { rmDirsRecursively } = require('../util/util')

const app = express()

app.use(CORS());
app.use(bodyParser.json({ strict: false, limit: '50mb' }));

app.get('/', async (req, res) => {
  return res.send('Hello world!')
})

app.post(`/infer`, async (req, res) => {
  try {
    const { text } = req.body
    const jobId = v4()
    console.log('jobId:', jobId)
    const textInputPath = `/home/ubuntu/1-radtts-repo/5-tts-input-text/${jobId}.txt`
    console.log('textInputPath:', textInputPath)
    const outputDir = `/home/ubuntu/1-radtts-repo/6-training-output/${jobId}`
    console.log('outputDir:', outputDir)

    const inferPath = `/home/ubuntu/1-radtts-repo/inference.py`
    const modelConfigPath = `-c /home/ubuntu/1-radtts-repo/2-configs/1-radtts-configs/config_ljs_dap.json`
    const modelPath = `-r /home/ubuntu/1-radtts-repo/1-models/1-radtts-models/lupe-fiasco-radtts-model`
    const vocoderPath = `-v /home/ubuntu/1-radtts-repo/1-models/2-hifigan-models/hifigan_libritts100360_generator0p5.pt`
    const vocoderConfigPath = `-k /home/ubuntu/1-radtts-repo/2-configs/2-hifigan-configs/hifigan_22khz_config.json`
    const textInput = `-t ${textInputPath}`
    const speaker = `-s lupefiasco`
    const speakerAttributes = `--speaker_attributes lupefiasco`
    const speakerText = `--speaker_text lupefiasco`
    const outputDirArg = `-o ${outputDir}`
    const radttsInferCommand = `${inferPath} ${modelConfigPath} ${modelPath} ${vocoderPath} ${vocoderConfigPath} ${textInput} ${speaker} ${speakerAttributes} ${speakerText} ${outputDirArg}`.split(' ')
    console.log('radttsInferCommand:', radttsInferCommand)

    writeFileSync(textInputPath, text)
    mkdirSync(outputDir)

    await execPythonComm(radttsInferCommand, { printLogs: true })

    const wavName = `0_0_lupefiasco_durscaling1.0_sigma0.8_sigmatext0.666_sigmaf01.0_sigmaenergy1.0_denoised_0.0.wav`
    const wavPath = `${outputDir}/${wavName}`
    console.log('wavPath:', wavPath)
    const wavContent = createReadStream(wavPath)
    const uploadResponse = await uploadToS3(`${jobId}.wav`, 'rapbot-rapgenie-outputs', wavContent, 'audio/wav')
    console.log('uploadResponse:', uploadResponse)
    const params = {
      Bucket: 'rapbot-rapgenie-outputs',
      Key: `${jobId}.wav`
    }
    const wavSignedUrl = s3Client.getSignedUrl('getObject', params)

    rmSync(textInputPath)
    await rmDirsRecursively(outputDir)
    return res.send({ wavSignedUrl })
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
