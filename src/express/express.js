require('dotenv').config()
const express = require('express')
const bodyParser = require('body-parser');
const CORS = require('cors');
const { v4 } = require('uuid')
const axios = require('axios')
const { writeFileSync, rmSync, mkdirSync, existsSync, createReadStream, createWriteStream, readFileSync, readdirSync } = require('fs')
const { Configuration, OpenAIApi } = require("openai")

const { uploadToS3, s3Client } = require('../aws/aws.js')
const { execPythonComm, execComm } = require('../bash/bash.js')

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

const app = express()

app.use(CORS());
app.use(bodyParser.json({ strict: false, limit: '50mb' }));

app.get('/', async (_, res) => {
  return res.send('Hello world!')
})

app.post(`/infer`, async (req, res) => {
  try {
    const { inferenceBody, tempo = 1.5 } = req.body

    if (!inferenceBody) {
      throw new Error(`'inferenceBody' must be defined!`)
    }

    const jobId = v4()
    const jobDir = `/home/ubuntu/jobs/${jobId}`
    mkdirSync(jobDir)
    const textInputFile = `${jobDir}/text-input.txt`

    writeFileSync(textInputFile, inferenceBody)

    const inferFunc = `../radtts/inference.py`
    const radttsModelConfig = `/home/ubuntu/rapgenie/src/configs/config_ljs_dap.json`
    const radttsModel = `../models/lupe-fiasco-radtts-model`
    const vocoder = `../models/hifigan_libritts100360_generator0p5.pt`
    const vocoderConfig = `../models/hifigan_22khz_config.json`
    const speaker = `lupefiasco`
    const speakerAttributes = `lupefiasco`
    const speakerText = `lupefiasco`
    const radttsInferCommand = [
      inferFunc,
      `-r`,
      radttsModel,
      `-c`,
      radttsModelConfig,
      `-v`,
      vocoder,
      `-k`,
      vocoderConfig,
      `-t`,
      textInputFile,
      `-s`,
      speaker,
      `--speaker_attributes`,
      speakerAttributes,
      `--speaker_text`,
      speakerText,
      `-o`,
      jobDir,
      `--token_dur_scaling`,
      1.5
    ]
    console.log('radttsInferCommand:', radttsInferCommand.join(` \\\n`))

    await execPythonComm(radttsInferCommand, { printLogs: true })

    const [inferOutput] = readdirSync(jobDir).filter(file => file.split('.').pop() === 'wav')
    console.log('wavPath:', `${jobDir}/${inferOutput}`)
    const wavContent = createReadStream(`${jobDir}/${inferOutput}`)
    await uploadToS3(`${jobId}.wav`, 'rapbot-rapgenie-outputs', wavContent, 'audio/wav')
    const params = {
      Bucket: 'rapbot-rapgenie-outputs',
      Key: `${jobId}.wav`
    }
    const wavSignedUrl = s3Client.getSignedUrl('getObject', params)

    const text = inferenceBody
    rmSync(jobDir, { recursive: true, force: true });
    return res.send({ wavSignedUrl, text })
  } catch (error) {
    console.log('error:', error)
    const stringifiedError = JSON.stringify(error, Object.getOwnPropertyNames(error))
    res.status(500).send(stringifiedError)
  }
})

app.post(`/infer-typecast`, async (req, res) => {
  try {
    const {
      inferenceBody,
      tempo = 1,
      style_label = 'toneup-1',
      actor_id = '61b007392f2010f2aa1a052a',
      max_seconds = 20,
      lang = 'en'
    } = req.body

    if (!inferenceBody) {
      throw new Error(`'inferenceBody' must be defined!`)
    }

    const jobId = v4()
    const jobDir = `/home/ubuntu/jobs/${jobId}`
    mkdirSync(jobDir)
    mkdirSync(`${jobDir}/wavs`)
    mkdirSync(`${jobDir}/typecast`)
    const textInputFile = `${jobDir}/text-input.txt`

    writeFileSync(textInputFile, inferenceBody)

    const text = inferenceBody

    const body = {
      text,
      lang,
      actor_id,
      max_seconds,
      tempo,
      style_label
    }
    const headers = {
      Authorization: `Bearer ${process.env.TYPECAST_TOKEN}`
    }
    const { data: { result: { speak_url } } } = await axios.post(`https://typecast.ai/api/speak`, body, { headers })
    console.log('speak_url:', speak_url)

    const pollForTypecastJob = async (speak_url) => {
      const { data: { result: { status, audio } } } = await axios.get(speak_url, { headers })

      if (status === 'done') {
        const { url } = audio
        return url
      } else if (status !== 'done') {
        return new Promise((resolve, reject) => {
          return setTimeout(async () => {
            try {
              return resolve(await pollForTypecastJob(speak_url))
            } catch (error) {
              return reject(error)
            }
          }, 1000)
        })
      } else {
        // TODO: handle whatever value for 'status' that typecast uses to indicate errors
      }
    }

    console.log("Starting polling...")
    const audioFileUrl = await pollForTypecastJob(speak_url)
    console.log('audioFileUrl:', audioFileUrl)

    const typecastWavStereo = `typecast-output-stereo-16-khz.wav`
    const writer = createWriteStream(`${jobDir}/typecast/${typecastWavStereo}`);
    const streamResponse = await axios.get(audioFileUrl, { headers, responseType: 'stream' });
    streamResponse.data.pipe(writer);

    await new Promise((resolve, reject) => {
      writer.on('finish', () => {
        console.log("typecast wav downloaded")
        return resolve()
      });
      writer.on('error', (error) => {
        console.error("Error while downloading typecast wav:", error)
        return reject(error)
      });
    })

    const typecastWavMono = `typecast-output-mono-22-khz.wav`
    const convertToMonoAnd225KhzComm = `ffmpeg -i ${jobDir}/typecast/${typecastWavStereo} -ar 22050 -ac 1 ${jobDir}/wavs/${typecastWavMono}`
    await execComm(convertToMonoAnd225KhzComm)

    const validationString = `${typecastWavMono}|${text.replace(`\n`, ' ')}.|lupefiasco`
    writeFileSync(`${jobDir}/validation.txt`, validationString)

    const conversionFunc = `../radtts/inference_voice_conversion.py`
    const radttsModel = `../models/lupe-fiasco-radtts-model`
    const radttsModelConfig = `/home/ubuntu/rapgenie/src/configs/config_ljs_dap.json`
    const vocoder = `../models/hifigan_libritts100360_generator0p5.pt`
    const vocoderConfig = `../models/hifigan_22khz_config.json`
    const dataConfigParams = `data_config.validation_files="{'Dummy': {'basedir': '${jobDir}', 'audiodir':'wavs', 'filelist': 'validation.txt'}}"`
    const radttsVoiceTransferCommand = [
      conversionFunc,
      `-r`,
      radttsModel,
      `-c`,
      radttsModelConfig,
      `-v`,
      vocoder,
      `-k`,
      vocoderConfig,
      `-o`,
      jobDir,
      `-p`,
      dataConfigParams
    ]
    console.log('radttsVoiceTransferCommand:', radttsVoiceTransferCommand.join(' \\\n'))
    await execPythonComm(radttsVoiceTransferCommand, { printLogs: true })
    console.log('Voice transfer done!')

    const voiceTransferOutput = readdirSync(jobDir).filter(file => file.split('.').pop() === 'wav')
    const wavContent = createReadStream(`${jobDir}/${voiceTransferOutput}`)
    await uploadToS3(`${jobId}.wav`, 'rapbot-rapgenie-outputs', wavContent, 'audio/wav')
    const params = {
      Bucket: 'rapbot-rapgenie-outputs',
      Key: `${jobId}.wav`
    }
    const wavSignedUrl = s3Client.getSignedUrl('getObject', params)

    // rmSync(jobDir, { recursive: true, force: true });

    return res.send({ wavSignedUrl, text })
  } catch (error) {
    console.log('error:', error)
    error && error.response && error.response.data && error.response.data.message && console.log('error:', error.response.data.message)
    const stringifiedError = JSON.stringify(error, Object.getOwnPropertyNames(error))
    console.log('stringifiedError:', stringifiedError)
    res.status(500).send(stringifiedError)
  }
})

app.post('/gpt/lyrics', async (req, res) => {
  try {
    const { topic } = req.body

    if (!topic) {
      throw new Error(`'topic' must be defined!`)
    }

    const jobId = v4()
    console.log('new jobId:', jobId)
    const jobDir = `/home/ubuntu/jobs/${jobId}`
    mkdirSync(jobDir)

    const textInputFile = `${jobDir}/text-input.txt`
    const getCoupletsFunc = `/home/ubuntu/rapgenie/src/gpt/get-couplet.py`
    const getCoupletsCommand = [getCoupletsFunc, topic, textInputFile]
    console.log('About to execute GPT...')
    await execPythonComm(getCoupletsCommand, { printLogs: true })
    console.log('Done executing GPT')
    const lineChoices = readFileSync(textInputFile, 'utf-8').split(`\n\n`).map(el => el.split(`\n`))
    rmSync(jobDir, { recursive: true, force: true });
    console.log('About to return')
    return res.status(200).send(lineChoices)
  } catch (error) {
    console.log('error:', error)
    error && error.response && error.response.data && error.response.data.message && console.log('error:', error.response.data.message)
    const stringifiedError = JSON.stringify(error, Object.getOwnPropertyNames(error))
    console.log('stringifiedError:', stringifiedError)
    res.status(500).send(stringifiedError)
  }
})

app.post('/openai', async (req, res) => {
  if (req.method === 'POST') {
    const {
      topic,
    } = req.body

    try {
      const { data: completions } = await openai.createCompletion({
        model: "text-davinci-003",
        prompt: `Give me a stanza of rap about: ${topic}`,
        temperature: 1,
        n: 1,
        presence_penalty: 0,
        frequency_penalty: 0,
        max_tokens: 3700
      });
      const { choices: [{ text }] } = completions

      return res.status(200).send(text.split(`\n`).map(el => [el]))
    } catch (error) {
      console.log('error:', error)
      logger.handleErr(error, req.query)
      const formattedError = JSON.parse(JSON.stringify(error, Object.getOwnPropertyNames(error)))
      return res.status(error.statusCode || 500).send(formattedError)
    }
  }
})

app.listen(3020, async () => {
  try {
    console.log('listening on 3020')
    !existsSync(`../jobs`) && mkdirSync(`../jobs`)
  } catch (error) {
    console.error('app.listen error:', error)
    console.log(error)
  }
})
