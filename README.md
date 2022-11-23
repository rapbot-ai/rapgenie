RapGenie: An end-to-end solution for synthesizing rap songs. Includes TTS voice generation and GPT lyrics generation.

# SYNTHESIZED RAP:

## VOICE:

Inference:

```
python3 \
/radtts/inference.py \
-c /config_ljs_dap.json \
-r ./lupe-fiasco-radtts-model \
-v ./hifigan_libritts100360_generator0p5.pt \
-k ./hifigan_22khz_config.json \
-t ./tts-input-text.txt \
-s lupefiasco \
--speaker_attributes lupefiasco \
--speaker_text lupefiasco \
-o /home/ubuntu/1-radtts-repo/6-training-output
```

Voice Transfer:

```
/home/ubuntu/1-radtts-repo/inference_voice_conversion.py
-r /home/ubuntu/1-radtts-repo/1-models/1-radtts-models/lupe-fiasco-radtts-model
-c /home/ubuntu/1-radtts-repo/2-configs/1-radtts-configs/config_ljs_dap.json
-v /home/ubuntu/1-radtts-repo/1-models/2-hifigan-models/hifigan_libritts100360_generator0p5.pt
-k /home/ubuntu/1-radtts-repo/2-configs/2-hifigan-configs/hifigan_22khz_config.json
-o /home/ubuntu/1-radtts-repo/6-training-output/f7bda518-b4f5-4c67-9a7d-9c79784ddec3
-p data_config.validation_files="{'Dummy': {'basedir': '/home/ubuntu/jobs/f7bda518-b4f5-4c67-9a7d-9c79784ddec3', 'audiodir':'wavs', 'filelist': 'validation.txt'}}"'
```

## LYRICS:

# COMMON COMMANDS:

## COPY FILE FROM AWS EC2 TO LOCAL DISK:

scp \
-i ~/rapbot-gpu-1.pem ubuntu@3.80.180.112:/home/ubuntu/1-radtts-repo/6-training-output/0_0_lupefiasco_durscaling1.0_sigma0.8_sigmatext0.666_sigmaf01.0_sigmaenergy1.0_denoised_0.0.wav ./

# EXPECTED SERVER FILE TREE:

~/jobs <-- create this manually

~/models <-- TODO: create script to download these items from my google drive using gdown
- lupe-fiasco-radtts-model
- gpt-j-8bit_couplets_generator.pt
- hifigan_libritts100360_generator0p5.pt
- hifigan_22khz_config.json

~/radtts <-- clone repo from github [here](https://github.com/NVIDIA/radtts)

~/rapgenie <-- clone repo from github [here](https://github.com/rapbot-ai/rapgenie)

~/tts-datasets <-- contains files used to train voice
- 8-formatted-lupe-lines-second-pass-22khz-mono-465 (on my GDrive [here](https://drive.google.com/drive/folders/1Yxj_ekL9Z_PZ7e0f9Qegq_RPUMVyoEGF?usp=share_link))
-- training.txt
-- validation.txt
-- wavs
--- 1761932.wav
--- 1761976.wav
--- 1804477.wav
--- etc.

# REQUIREMENTS:

ffmpeg
radtts
typecast API
python
ubuntu
cuda/GPU

# API:

## INFER:

Request:

```
{
    "inferenceType": "text",
    "inferenceBody": "yo what is up"
}
```

Response:

```
{
    "wavSignedUrl": "https://rapbot-rapgenie-outputs.s3.amazonaws.com/d9203cb4-7f38-41cb-8d47-f0530fe8bd92.wav?AWSAccessKeyId=AKIAIFUTGW2VXZPECBPA&Expires=1669161516&Signature=ugnUR4AId%2FtghXTjbKVrmPH87Oc%3D",
    "text": "yo what is up"
}
```

## INFER-TYPECAST

Request:

```
{
    "inferenceType": "topic",
    "inferenceBody": "beef",
    "tempo": 0.75,
    "style_label": "toneup-1",
    "actor_id": "61b007392f2010f2aa1a052a",
    "max_seconds": 20,
    "lang": "en"
}
```

Response:

```
{
    "wavSignedUrl": "https://rapbot-rapgenie-outputs.s3.amazonaws.com/d9203cb4-7f38-41cb-8d47-f0530fe8bd92.wav?AWSAccessKeyId=AKIAIFUTGW2VXZPECBPA&Expires=1669161516&Signature=ugnUR4AId%2FtghXTjbKVrmPH87Oc%3D",
    "text": "yo what is up"
}
```