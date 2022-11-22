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

    const inferOutput = `0_0_lupefiasco_durscaling1.0_sigma0.8_sigmatext0.666_sigmaf01.0_sigmaenergy1.0_denoised_0.0.wav`
    console.log('wavPath:', `${radttsOutputDir}/${inferOutput}`)
    const wavContent = createReadStream(`${radttsOutputDir}/${inferOutput}`)
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

app.post(`/infer-typecast`, async (req, res) => {
  try {
    const { inferenceBody, inferenceType } = req.body
    const jobId = v4()
    console.log('jobId:', jobId)
    const gptLyricsFile = `/home/ubuntu/1-radtts-repo/5-tts-input-text/${jobId}.txt`

    if (inferenceType === 'text' && inferenceBody) {
      console.log("Text-based STT...")
    } else if (inferenceType === 'topic' && inferenceBody) {
      console.log("Topic-based STT...")
      const getCoupletsFunc = `/home/ubuntu/1-radtts-repo/src/gpt/get-couplet.py`
      const getCoupletsCommand = [getCoupletsFunc, inferenceBody, gptLyricsFile,]
      await execPythonComm(getCoupletsCommand, { printLogs: true })
    } else if (inferenceType !== 'topic' || !inferenceType === 'text' || '') {
      throw new Error(`inferenceType must be 'topic' or 'text'`)
    } else if (!inferenceBody) {
      throw new Error(`'inferenceBody' must be defined!`)
    }

    const radttsOutputDir = `/home/ubuntu/1-radtts-repo/6-training-output/${jobId}`

    inferenceType === 'text' && writeFileSync(gptLyricsFile, inferenceBody)
    !existsSync(radttsOutputDir) && mkdirSync(radttsOutputDir)
    !existsSync(`${radttsOutputDir}/wavs`) && mkdirSync(`${radttsOutputDir}/wavs`)

    const dollarJrActorId = '61b007392f2010f2aa1a052a'
    const tempo = 0.5
    const style_label = 'tonedown-1'
    const lang = 'en'
    const max_seconds = 20
    const text = inferenceType === 'text' ?
      inferenceBody :
      readFileSync(gptLyricsFile, 'utf-8')

    const body = {
      text,
      lang,
      actor_id: dollarJrActorId,
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

    const typecastWavStereo = `1.wav`
    const writer = createWriteStream(`${radttsOutputDir}/wavs/${typecastWavStereo}`);
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

    // TODO: convert to mono and 22.5 khz
    const typecastWavMono = `2.wav`
    const convertToMonoAnd225KhzComm = `ffmpeg -i ${radttsOutputDir}/wavs/${typecastWavStereo} -ar 22050 -ac 1 ${radttsOutputDir}/wavs/${typecastWavMono}`
    await execComm(convertToMonoAnd225KhzComm)

    const validationString = `${typecastWavMono}|${text.replace(`\n`, ' ')}.|lupefiasco`
    writeFileSync(`${radttsOutputDir}/validation.txt`, validationString)

    const conversionFunc = `/home/ubuntu/1-radtts-repo/inference_voice_conversion.py`
    const radttsModel = `/home/ubuntu/1-radtts-repo/1-models/1-radtts-models/lupe-fiasco-radtts-model`
    const radttsModelConfig = `/home/ubuntu/1-radtts-repo/2-configs/1-radtts-configs/config_ljs_dap.json`
    const vocoder = `/home/ubuntu/1-radtts-repo/1-models/2-hifigan-models/hifigan_libritts100360_generator0p5.pt`
    const vocoderConfig = `/home/ubuntu/1-radtts-repo/2-configs/2-hifigan-configs/hifigan_22khz_config.json`
    const radttsOutputDirArg = `${radttsOutputDir}`
    const escapeSlashes = ``
    const validationParams = `${escapeSlashes}"{'Dummy': {'basedir': '${radttsOutputDir}', 'audiodir':'wavs', 'filelist': 'validation.txt'}}${escapeSlashes}"`
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
      radttsOutputDirArg,
      `-p`,
      validationParamsArg
    ]
    console.log('radttsVoiceTransferCommand:', radttsVoiceTransferCommand.join(' '))
    await execPythonComm(radttsVoiceTransferCommand, { printLogs: true })
    console.log('Voice transfer done!')

    // Change this code to use fs.readdir and find the file ending in .wav
    const voiceTransferOutput = `2_0_sid0_sigma0.8.wav`
    const wavContent = createReadStream(`${radttsOutputDir}/${voiceTransferOutput}`)
    await uploadToS3(`${jobId}.wav`, 'rapbot-rapgenie-outputs', wavContent, 'audio/wav')
    const params = {
      Bucket: 'rapbot-rapgenie-outputs',
      Key: `${jobId}.wav`
    }
    const wavSignedUrl = s3Client.getSignedUrl('getObject', params)

    rmSync(gptLyricsFile)
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
