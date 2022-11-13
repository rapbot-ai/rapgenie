RapGenie: An end-to-end solution for synthesizing rap songs. Includes TTS voice generation and GPT lyrics generation.

# VOICE GENERATION:

## RADTTS:

python3 \
/home/ubuntu/1-radtts-repo/inference.py \
-c /home/ubuntu/1-radtts-repo/2-configs/1-radtts-configs/config_ljs_dap.json \
-r /home/ubuntu/1-radtts-repo/1-models/1-radtts-models/lupe-fiasco-radtts-model \
-v /home/ubuntu/1-radtts-repo/1-models/2-hifigan-models/hifigan_libritts100360_generator0p5.pt \
-k /home/ubuntu/1-radtts-repo/2-configs/2-hifigan-configs/hifigan_22khz_config.json \
-t /home/ubuntu/1-radtts-repo/5-tts-input-text/catch-me-like-spalding.txt \
-s lupefiasco \
--speaker_attributes lupefiasco \
--speaker_text lupefiasco \
-o /home/ubuntu/1-radtts-repo/6-training-output

## GPT GENERATION:

# COMMON COMMANDS:

## COPY FROM AWS EC2:

scp -i ~/rapbot-gpu-1.pem ubuntu@3.80.180.112:/home/ubuntu/1-radtts-repo/6-training-output/0_0_lupefiasco_durscaling1.0_sigma0.8_sigmatext0.666_sigmaf01.0_sigmaenergy1.0_denoised_0.0.wav ./