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

## COPY FILE OR DIRECTORY TO/FROM AWS EC2 TO/FROM LOCAL DISK:

scp -i "$PEM_FILE" <LOCAL_SOURCE_PATH> "ubuntu@$GPU_IP_ADDRESS:<REMOTE_DESTINATION_PATH>"

*change -i to -r in the above command in order to copy folders*

### Example:

```
scp -i "$PEM_FILE" ./hifigan_22khz_config.json "ubuntu@$GPU_IP_ADDRESS:/home/ubuntu/models"
```

## 

# EXPECTED SERVER FILE TREE:

~/jobs <-- this should be created automatically. If it's not, just make it yourself.

~/models <-- where the lupe fiasco and hifigan libritts model actually lives

~/radtts <-- clone repo from github [here](https://github.com/NVIDIA/radtts)

~/rapgenie <-- clone repo from github [here](https://github.com/rapbot-ai/rapgenie)

~/tts-datasets <-- contains files used to train voice

~/tts-datasets/8-formatted-lupe-lines-second-pass-22khz-mono-465 
- get this on my GDrive [here](https://drive.google.com/drive/folders/1Yxj_ekL9Z_PZ7e0f9Qegq_RPUMVyoEGF?usp=share_link)
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

```
mkdir -p /home/ubuntu/test \
&& echo "this is a test" >> /home/ubuntu/test/text-input.txt \
&& python3 /home/ubuntu/radtts/inference.py \
-r /home/ubuntu/models/lupe-fiasco-radtts-model \
-c /home/ubuntu/rapgenie/src/configs/config_ljs_dap.json \
-v /home/ubuntu/models/hifigan_libritts100360_generator0p5.pt \
-k /home/ubuntu/models/hifigan_22khz_config.json \
-t /home/ubuntu/test/text-input.txt \
-s lupefiasco \
--speaker_attributes lupefiasco \
--speaker_text lupefiasco \
-o /home/ubuntu/test \
--token_dur_scaling 1.5
```

# SETTING UP INFERENCE WITH A CUSTOM RADTTS MODEL:

1. Go to AWS and set up an Ubuntu g4dn.xlarge instance with 104 GB of space.

2. Restrict the read-write perms of your .pem file so AWS doesn't complain about it:

```
PEM_FILE=<FULL_PATH_TO_PEM_FILE>
chmod 400 "$PEM_FILE"
```

3. Log onto the server:

```
GPU_IP_ADDRESS=<INSTANCE_IP_ADDRESS_HERE>
ssh -i "$PEM_FILE" "ubuntu@$GPU_IP_ADDRESS"
```

4. Create the necessary folders: 

```
mkdir /home/ubuntu/models
mkdir /home/ubuntu/tts-datasets && mkdir /home/ubuntu/tts-datasets/8-formatted-lupe-lines-second-pass-22khz-mono-465
mkdir /home/ubuntu/jobs
```

5. Download the hifigan_libritts model, the lupe fiasco radtts model & model config, and the lupe fiasco training.txt file to your local machine from Google drive here:

- lupe fiasco + model config [here](https://drive.google.com/drive/folders/1j981XSwxFh_65s3JKVdEAFX3GL-vDnJ5)
- hifigan_libritts [here](https://drive.google.com/file/d/1gbrmWexvW3fwEM0aDxxXC18A2f5DzuED/view?usp=drive_link)
- lupe fiasco training.txt file [here](https://drive.google.com/drive/folders/1Yxj_ekL9Z_PZ7e0f9Qegq_RPUMVyoEGF?usp=drive_link)

6. Upload those files to the EC2 instance:

```
scp -i "$PEM_FILE" ./hifigan_22khz_config.json "ubuntu@$GPU_IP_ADDRESS:/home/ubuntu/models"
scp -i "$PEM_FILE" ./lupe-fiasco-radtts-model "ubuntu@$GPU_IP_ADDRESS:/home/ubuntu/models"
scp -i "$PEM_FILE" ./hifigan_libritts100360_generator0p5.pt "ubuntu@$GPU_IP_ADDRESS:/home/ubuntu/models"
scp -i "$PEM_FILE" ./8-formatted-lupe-lines-second-pass-22khz-mono-465/training.txt "ubuntu@$GPU_IP_ADDRESS:/home/ubuntu/tts-datasets/8-formatted-lupe-lines-second-pass-22khz-mono-465"
```

7. Install cuda toolkit using these directions. Input your instance config (Ubuntu, 20.04, Linux, etc.) to obtain the right installer [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network).

8. Make sure that the GPU is available on the instance:

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

```
sudo apt update
sudo apt install -y git
git --version
cd ~
git clone https://github.com/NVIDIA/radtts
git clone https://github.com/rapbot-ai/rapgenie
```

11. Install Python tooling for virtual environments:

This project currently supports Python 3.11.x for `radtts`.

```
sudo apt install -y python3-venv python3-pip build-essential python3-dev
```

12. Create a radtts virtual environment, add lmdb to requirements, then install dependencies:

```
cd ~/radtts
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
echo "lmdb" >> requirements.txt
pip install -r requirements.txt
```

13. Install nvm:

```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
```

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

20. Populate .env so that the app can upload .wav outputs to S3:

```
cd /home/ubuntu/rapgenie
cp .env.example .env
vi .env
```

21. Start the rapgenie server:

```
npm run dev
```

22. Perform inference by sending a GET request to /infer:

```
curl --location "http://$GPU_IP_ADDRESS:3020/infer" \
--header 'Content-Type: application/json' \
--data '{
    "inferenceBody": "this is a test"
}'
```

23. That inference command might throw an error about missing packages, or have version mismatches. If those happen, just install them  individually:

```
python3 -m pip install <MISSING-DEPENDENCY>
```

Numpy in particular can be tricky. v2.0.0 deprecated some methods we need. You might have to go in and manually fix the librosa packages use of numpy to make it work. This could perhaps be fixed by pinning the version of librosa in `radtts`, but I have not tried that. From what I can remember:

  a. Change `np.float` to `float` in `.../utils/utils.py`.

  b. Change `np.complex` to `np.complex128` (or `np.complex64`, whatever number it currently has).

24. Install pm2 and then start the process running in the background:

```
npm install -g pm2
npm run pm2
```

## TODOs:

1. create script to download these items from my google drive using gdown
- lupe-fiasco-radtts-model
- 8-formatted-lupe-lines-second-pass-22khz-mono-465/training.txt
- gpt-j-8bit_couplets_generator.pt
- hifigan_libritts100360_generator0p5.pt
- hifigan_22khz_config.json

2. Fork radtts and add lmdb in the fork so the step that does this manually is not needed