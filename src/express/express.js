require('dotenv').config()
const express = require('express')
const bodyParser = require('body-parser');
const CORS = require('cors');
const { v4 } = require('uuid')
const { writeFileSync, rmSync, mkdirSync, existsSync, createReadStream } = require('fs')
const { uploadToS3, s3Client } = require('../aws/aws')
const { execPythonComm } = require('../bash/bash')

const app = express()

app.use(CORS());
app.use(bodyParser.json({ strict: false, limit: '50mb' }));

app.get('/', async (req, res) => {
  return res.send('Hello world!')
})

app.post(`/infer`, async (req, res) => {
  try {
    const { inferenceBody, inferenceType } = req.body
    const jobId = v4()
    console.log('jobId:', jobId)
    const radttsTextInput = `/home/ubuntu/1-radtts-repo/5-tts-input-text/${jobId}.txt`

    if (inferenceType === 'text' && inferenceBody) {
      console.log("Text-based STT...")
    } else if (inferenceType === 'topic' && inferenceBody) {
      console.log("Topic-based STT...")
      const getCoupletsCommand = `/home/ubuntu/rapgenie/src/gpt/get-couplet.py ${inferenceBody} ${radttsTextInput}`.split(' ')
      await execPythonComm(getCoupletsCommand, { printLogs: true })
    } else if (inferenceType !== 'topic' || !inferenceType === 'text' || '') {
      throw new Error(`inferenceType must be 'topic' or 'text'`)
    } else if (!inferenceBody) {
      throw new Error(`'inferenceBody' must be defined!`)
    }

    const inferFunc = `/home/ubuntu/1-radtts-repo/inference.py`
    const radttsModelConfig = `-c /home/ubuntu/1-radtts-repo/2-configs/1-radtts-configs/config_ljs_dap.json`
    const radttsModel = `-r /home/ubuntu/1-radtts-repo/1-models/1-radtts-models/lupe-fiasco-radtts-model`
    const vocoder = `-v /home/ubuntu/1-radtts-repo/1-models/2-hifigan-models/hifigan_libritts100360_generator0p5.pt`
    const vocoderConfig = `-k /home/ubuntu/1-radtts-repo/2-configs/2-hifigan-configs/hifigan_22khz_config.json`
    const textInput = `-t ${radttsTextInput}`
    const speaker = `-s lupefiasco`
    const speakerAttributes = `--speaker_attributes lupefiasco`
    const speakerText = `--speaker_text lupefiasco`
    const radttsOutputDir = `/home/ubuntu/1-radtts-repo/6-training-output/${jobId}`
    const radttsOutputDirArg = `-o ${radttsOutputDir}`
    const radttsInferCommand = [
      inferFunc,
      radttsModelConfig,
      radttsModel,
      vocoder,
      vocoderConfig,
      textInput,
      speaker,
      speakerAttributes,
      speakerText,
      radttsOutputDirArg,
    ]
    console.log('radttsInferCommand:', radttsInferCommand.join(`\n`))

    inferenceType === 'text' && writeFileSync(radttsTextInput, inferenceBody)
    !existsSync(radttsOutputDir) && mkdirSync(radttsOutputDir)

    await execPythonComm(radttsInferCommand, { printLogs: true })

    const radttsOutputWav = `0_0_lupefiasco_durscaling1.0_sigma0.8_sigmatext0.666_sigmaf01.0_sigmaenergy1.0_denoised_0.0.wav`
    console.log('wavPath:', `${radttsOutputDir}/${radttsOutputWav}`)
    const wavContent = createReadStream(`${radttsOutputDir}/${radttsOutputWav}`)
    await uploadToS3(`${jobId}.wav`, 'rapbot-rapgenie-outputs', wavContent, 'audio/wav')
    const params = {
      Bucket: 'rapbot-rapgenie-outputs',
      Key: `${jobId}.wav`
    }
    const wavSignedUrl = s3Client.getSignedUrl('getObject', params)

    rmSync(radttsTextInput)
    rmSync(radttsOutputDir, { recursive: true, force: true });
    return res.send({ wavSignedUrl })
  } catch (error) {
    console.log('error:', error)
    const stringifiedError = JSON.stringify(error, Object.getOwnPropertyNames(error))
    res.status(500).send(stringifiedError)
  }
})

app.post(`/typecast-callback`, async (req, res) => {
  try {
    const { body } = req
    console.log('body:', body)
    return res.send(body)
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
