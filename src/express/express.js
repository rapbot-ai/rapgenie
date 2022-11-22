require('dotenv').config()
const express = require('express')
const bodyParser = require('body-parser');
const CORS = require('cors');
const { v4 } = require('uuid')
const axios = require('axios')
const { writeFileSync, rmSync, mkdirSync, existsSync, createReadStream, createWriteStream } = require('fs')

const { uploadToS3, s3Client } = require('../aws/aws.js')
const { execPythonComm, execComm } = require('../bash/bash.js')

const app = express()

app.use(CORS());
app.use(bodyParser.json({ strict: false, limit: '50mb' }));

app.get('/', async (_, res) => {
  return res.send('Hello world!')
})

app.post(`/infer`, async (req, res) => {
  try {
    const { inferenceBody, inferenceType } = req.body
    const jobId = v4()
    console.log('jobId:', jobId)
    const outputDir = `/home/ubuntu/jobs/${jobId}`
    mkdirSync(outputDir)
    const radttsTextInput = `${outputDir}/text-input.txt`

    if (inferenceType === 'text' && inferenceBody) {
      console.log("Text-based STT...")
      writeFileSync(radttsTextInput, inferenceBody)
    } else if (inferenceType === 'topic' && inferenceBody) {
      console.log("Topic-based STT...")
      const getCoupletsCommand = `/home/ubuntu/rapgenie/src/gpt/get-couplet.py ${inferenceBody} ${radttsTextInput}`.split(' ')
      await execPythonComm(getCoupletsCommand, { printLogs: true })
    } else if (inferenceType !== 'topic' || !inferenceType === 'text' || '') {
      throw new Error(`inferenceType must be 'topic' or 'text'`)
    } else if (!inferenceBody) {
      throw new Error(`'inferenceBody' must be defined!`)
    }

    const inferFunc = `/home/ubuntu/radtts/inference.py`
    const radttsModelConfig = `/home/ubuntu/models/config_ljs_dap.json`
    const radttsModel = `/home/ubuntu/models/lupe-fiasco-radtts-model`
    const vocoder = `/home/ubuntu/models/hifigan_libritts100360_generator0p5.pt`
    const vocoderConfig = `/home/ubuntu/models/hifigan_22khz_config.json`
    const speaker = `lupefiasco`
    const speakerAttributes = `lupefiasco`
    const speakerText = `lupefiasco`
    const radttsInferCommand = [
      inferFunc,
      `-c`,
      radttsModel,
      `-k`,
      radttsModelConfig,
      `-v`,
      vocoder,
      `-k`,
      vocoderConfig,
      `-t`,
      radttsTextInput,
      `-s`,
      speaker,
      `--speaker_attributes`,
      speakerAttributes,
      `--speaker_text`,
      speakerText,
      `-o`,
      outputDir,
    ]
    console.log('radttsInferCommand:', radttsInferCommand.join(`\n`))

    !existsSync(outputDir) && mkdirSync(outputDir)

    await execPythonComm(radttsInferCommand, { printLogs: true })

    const inferOutput = `0_0_lupefiasco_durscaling1.0_sigma0.8_sigmatext0.666_sigmaf01.0_sigmaenergy1.0_denoised_0.0.wav`
    console.log('wavPath:', `${outputDir}/${inferOutput}`)
    const wavContent = createReadStream(`${outputDir}/${inferOutput}`)
    await uploadToS3(`${jobId}.wav`, 'rapbot-rapgenie-outputs', wavContent, 'audio/wav')
    const params = {
      Bucket: 'rapbot-rapgenie-outputs',
      Key: `${jobId}.wav`
    }
    const wavSignedUrl = s3Client.getSignedUrl('getObject', params)

    rmSync(radttsTextInput)
    rmSync(outputDir, { recursive: true, force: true });
    return res.send({ wavSignedUrl })
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
      inferenceType,
      tempo = 1,
      style_label = 'normal-1',
      actor_id = '61b007392f2010f2aa1a052a',
      max_seconds = 20,
      lang = 'en'
    } = req.body

    const jobId = v4()
    const outputDir = `/home/ubuntu/jobs/${jobId}`
    mkdirSync(outputDir)
    const gptLyricsFile = `${outputDir}/text-input.txt`

    if (inferenceType === 'text' && inferenceBody) {
      console.log("Text-based STT...")
      writeFileSync(gptLyricsFile, inferenceBody)
    } else if (inferenceType === 'topic' && inferenceBody) {
      console.log("Topic-based STT...")
      const getCoupletsFunc = `/home/ubuntu/radtts/src/gpt/get-couplet.py`
      const getCoupletsCommand = [getCoupletsFunc, inferenceBody, gptLyricsFile]
      await execPythonComm(getCoupletsCommand, { printLogs: true })
    } else if (inferenceType !== 'topic' || !inferenceType === 'text' || '') {
      throw new Error(`inferenceType must be 'topic' or 'text'`)
    } else if (!inferenceBody) {
      throw new Error(`'inferenceBody' must be defined!`)
    }

    !existsSync(`${outputDir}/wavs`) && mkdirSync(`${outputDir}/wavs`)

    const text = inferenceType === 'text' ?
      inferenceBody :
      readFileSync(gptLyricsFile, 'utf-8')

    const body = {
      text,
      lang,
      actor_id,
      max_seconds,
      tempo,
      style_label
    }
    console.log('body:', body)
    const headers = {
      Authorization: `Bearer ${process.env.TYPECAST_TOKEN}`
    }
    console.log('headers:', headers)
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
    const writer = createWriteStream(`${outputDir}/wavs/${typecastWavStereo}`);
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
    const convertToMonoAnd225KhzComm = `ffmpeg -i ${outputDir}/wavs/${typecastWavStereo} -ar 22050 -ac 1 ${outputDir}/wavs/${typecastWavMono}`
    await execComm(convertToMonoAnd225KhzComm)

    const validationString = `${typecastWavMono}|${text.replace(`\n`, ' ')}.|lupefiasco`
    writeFileSync(`${outputDir}/validation.txt`, validationString)

    const conversionFunc = `/home/ubuntu/radtts/inference_voice_conversion.py`
    const radttsModel = `/home/ubuntu/models/lupe-fiasco-radtts-model`
    const radttsModelConfig = `/home/ubuntu/models/config_ljs_dap.json`
    const vocoder = `/home/ubuntu/models/hifigan_libritts100360_generator0p5.pt`
    const vocoderConfig = `/home/ubuntu/models/hifigan_22khz_config.json`
    const validationParams = `"{'Dummy': {'basedir': '${outputDir}', 'audiodir':'wavs', 'filelist': 'validation.txt'}}"`
    const validationParamsArg = `data_config.validation_files=${validationParams}`
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
      outputDir,
      `-p`,
      validationParamsArg
    ]
    console.log('radttsVoiceTransferCommand:', radttsVoiceTransferCommand.join(' \\\n'))
    await execPythonComm(radttsVoiceTransferCommand, { printLogs: true })
    console.log('Voice transfer done!')

    const voiceTransferOutput = `2_0_sid0_sigma0.8.wav`
    const wavContent = createReadStream(`${outputDir}/${voiceTransferOutput}`)
    await uploadToS3(`${jobId}.wav`, 'rapbot-rapgenie-outputs', wavContent, 'audio/wav')
    const params = {
      Bucket: 'rapbot-rapgenie-outputs',
      Key: `${jobId}.wav`
    }
    const wavSignedUrl = s3Client.getSignedUrl('getObject', params)

    rmSync(gptLyricsFile)
    rmSync(outputDir, { recursive: true, force: true });

    return res.send({ wavSignedUrl })
  } catch (error) {
    console.log('error:', error)
    const stringifiedError = JSON.stringify(error, Object.getOwnPropertyNames(error))
    console.log('stringifiedError:', stringifiedError)
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
