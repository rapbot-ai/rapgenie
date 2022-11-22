RapGenie: An end-to-end solution for synthesizing rap songs. Includes TTS voice generation and GPT lyrics generation.

# VOICE GENERATION:

## RADTTS:

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

## GPT GENERATION:

# COMMON COMMANDS:

## COPY FROM AWS EC2:

scp -i ~/rapbot-gpu-1.pem ubuntu@3.80.180.112:/home/ubuntu/1-radtts-repo/6-training-output/0_0_lupefiasco_durscaling1.0_sigma0.8_sigmatext0.666_sigmaf01.0_sigmaenergy1.0_denoised_0.0.wav ./

# EXPECTED SERVER FILE TREE:

~/jobs <-- create this manually

~/models <-- TODO: create script to download these items from my google drive using gdown
-- lupe-fiasco-radtts-model
-- gpt-j-8bit_couplets_generator.pt
-- hifigan_libritts100360_generator0p5.pt
-- hifigan_22khz_config.json

~/radtts <-- clone repo from github

~/rapgenie <-- clone repo from github

~/tts-datasets <-- contains files used to train voice
-- 8-formatted-lupe-lines-second-pass-22khz-mono-465
---- training.txt
---- validation.txt
---- wavs
------ 1761932.wav
------ 1761976.wav
------ 1804477.wav

## REQUIREMENTS:

ffmpeg
radtts
typecast API
python
ubuntu
cuda/GPU