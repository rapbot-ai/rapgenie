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
-i ~/rapbot-gpu-1.pem ubuntu@3.80.180.112:/home/ubuntu/jobs/244a13bd-42ec-4042-9ebc-da8c1f4f3458/wavs/typecast-output-mono-22-khz.wav ./

scp -i ./gpt-j-8bit_065000.pt ubuntu@3.80.180.112:/home/ubuntu/models/

# EXPECTED SERVER FILE TREE:

~/jobs <-- this should becreated automatically. If it's not, just make it yourself

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

## INFERENCE COMMAND:

python3 /home/ubuntu/radtts/inference.py \
-r /home/ubuntu/models/lupe-fiasco-radtts-model \
-c /home/ubuntu/rapgenie/src/configs/config_ljs_dap.json \
-v /home/ubuntu/models/hifigan_libritts100360_generator0p5.pt \
-k /home/ubuntu/models/hifigan_22khz_config.json \
-t /home/ubuntu/jobs/e81f53a9-8250-44a5-937a-27d326b596e6/text-input.txt \
-s lupefiasco \
--speaker_attributes lupefiasco \
--speaker_text lupefiasco \
-o /home/ubuntu/jobs/e81f53a9-8250-44a5-937a-27d326b596e6 \
--token_dur_scaling 1.5

# SETTING UP INFERENCE WITH A CUSTOM RADTTS MODEL:

1. Go to AWS and Set up your instance:
- ubuntu
- 20.04
- g4dn.xlarge
- an ubuntu 22.04 g4dn.xlarge instance with 104 GB space.
- 104 GB space

2. Restrict the read-write perms of your .pem file so AWS doesn't complain about it:

```
sudo chmod 400 <PEM-FILE-NAME>.pem
```

3. Log onto the device using this command:

```
ssh -i <PEM-NAME-HERE>4.pem ubuntu@<INSTANCE-IP-ADDRESS-HERE>
```

4. Create the necessary folders: 

```
mkdir /home/ubuntu/models
mkdir /home/ubuntu/tts-datasets
```

5. Download the hifigan_libritts model, the lupe fiasco radtts model & model config, and the lupe fiasco training data to your local machine from Google drive here:

lupe fiasco + model config:
https://drive.google.com/drive/folders/1j981XSwxFh_65s3JKVdEAFX3GL-vDnJ5

hifigan_libritts:
https://drive.google.com/file/d/1gbrmWexvW3fwEM0aDxxXC18A2f5DzuED/view?usp=drive_link

lupe fiasco training data:
https://drive.google.com/drive/folders/1Yxj_ekL9Z_PZ7e0f9Qegq_RPUMVyoEGF?usp=drive_link

6. Upload your model and model config using these commands:

scp -i ~/rapbot-gpu-4.pem ./hifigan_22khz_config.json ubuntu@18.208.144.207:/home/ubuntu/models
scp -i ~/rapbot-gpu-4.pem ./lupe-fiasco-radtts-model ubuntu@18.208.144.207:/home/ubuntu/models
scp -i ~/rapbot-gpu-4.pem ./hifigan_libritts100360_generator0p5.pt ubuntu@18.208.144.207:/home/ubuntu/models
scp -i ~/rapbot-gpu-4.pem ./8-formatted-lupe-lines-second-pass-22khz-mono-465/training.txt ubuntu@18.208.144.207:/home/ubuntu/tts-datasets/8-formatted-lupe-lines-second-pass-22khz-mono-465

7. Install cuda toolkit using these directions. Input your instance config (Ubuntu, 20.04, Linux, etc.) to obtain the right installer:

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network

8. Make sure that the cuda is available on the instance:

```
python3
import torch
torch.cuda.is_available()
```
If it's not, you probably installed CUDA or your nvidia drivers incorrectly.

9. Run nvidia-smi to make sure your GPU is there:

```
nvidia-smi
```

10. Clone the `rapgenie` and `radtts` repos to the instance:

https://github.com/NVIDIA/radtts
https://github.com/rapbot-ai/rapgenie

11. Install pip:

```
sudo apt install python3-pip -y
```

11. Install dependencies for radtts:

```
python3 -m pip install -r requirements.txt
```

It might have some missing dependencies. Just install those individually:

```
python3 -m pip install <MISSING-DEPENDENCY>
```

13. Install nvm:

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash

14. Check nvm installation:

```
nvm --version
```

15. Install node:

```
nvm install lts
```

16. Check node version:

```
node --version
```

17. Install yarn:

```
npm install -g yarn
```

18. Install rapgenie dependencies:

```
yarn install
```

19. Install nodemon:

```
npm install -g nodemon
```

20. Add creds for AWS to upload the output .wav to S3:

```
cd /home/ubuntu/rapgenie
vi .env
<COPY-CREDS-INTO-FILE-AND-SAVE>
```

21. Start the rapgenie server:

```
npm run dev
```

22. Perform inference by sending a GET request to /infer:

```
curl --location 'http://IP_ADDRESS_HERE:3020/infer' \
--header 'Content-Type: application/json' \
--data '{
    "inferenceBody": "hey"
}'
```

23. That inference command might throw an error about missing packages, or have version mismatches. If those happen, just install them individually:

```
python3 -m pip install <MISSING-DEPENDENCY>
```

Numpy in particular can be tricky. v2.0.0 deprecated some methods we need. You might have to go in and manually fix the librosa packages use of numpy to make it work. This could perhaps be fixed by pinning the version of librosa in `radtts`, but I have not tried that. Off the top of my head:

  a. Change `np.float` to `float` in `.../utils/utils.py`.
  b. Change `np.complex` to `np.complex128` (or `np.complex64`, whatever number it currently has).

24. Install pm2 and then keep the process running in the background:

```
npm install -g pm2
npm run pm2
```
