# ASTRAL Quantization

A speech linguistic content quantizer that operates on Hubert-large features. This model is trained with explicit ASR supervision to preserve linguistic content while discarding speaker traits.

## Overview

This repository contains a speech quantization model that processes audio input to extract and quantize linguistic content. The model uses Hubert-large features as input and is designed to maintain high-quality linguistic information while reducing speaker-specific characteristics.

## Usage

The main script (`main.py`) provides a simple interface to run the model on audio files. Checkpoints will be automatically downloaded if using a pre-trained model.

### Running the Script

```bash
python main.py [--model_name MODEL_NAME] [--model_path MODEL_PATH] [--config_path CONFIG_PATH] [--audio_path AUDIO_PATH]
```

#### Parameters:
- `--model_name`: Name of the pre-trained model to use. Available options: "bsq32" or "bsq2048"
- `--model_path`: Path to your custom model weights (required if not using pre-trained model)
- `--config_path`: Path to your custom model configuration (required if not using pre-trained model)
- `--audio_path`: Path to the input audio file (e.g. "test_waves/cafe_0.wav")

### Example Usage

Using a pre-trained model:
```bash
python main.py --model_name bsq32 --audio_path your_audio.wav
```

Using a custom model:
```bash
python main.py --model_path path/to/model.bin --config_path path/to/config.yml --audio_path your_audio.wav
```

### Output

The script will output:
1. The quantized indices representing the linguistic content of the audio
2. A decoded transcript of the speech

example output:
```
tensor([[15, 15,  7, 14, 26, 26,  3,  5,  6,  6, 18, 23, 31,  7,  3, 15,  6,  0,
          7, 11,  9, 12,  7, 15,  2, 18, 31, 31,  2,  1, 25, 24, 11,  8,  4, 22,
         25, 25, 24,  6, 11, 12,  2,  9,  9,  6, 12, 27,  3, 15,  2,  1, 16, 25,
         25, 19,  2,  6,  2,  0, 16, 24, 18,  8, 14, 15, 27, 26, 27, 27, 27, 26,
         27, 31, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
         27, 11, 11, 11, 10, 31, 31,  8, 10, 15, 18, 18, 26,  8, 27, 17,  6,  6,
         26, 15, 11, 12,  0,  6,  2, 30, 25, 25, 13, 13, 23, 29, 28, 22, 31, 31,
          2,  1,  9, 18, 11,  8,  4, 21,  3,  9,  9, 14, 10,  0, 23,  3,  9,  9,
         30,  1,  2,  1,  9, 25, 18, 25, 25, 16, 10, 26, 14, 15,  8,  0,  5, 20,
          6,  6, 10, 22, 17,  1, 17, 16, 15, 11,  8, 21, 17, 17, 21, 30, 25,  9,
          9, 29, 29, 22, 11, 13,  2,  4, 13, 15,  8,  5, 20, 17,  0, 23, 17,  6,
          4, 20,  7, 15,  9, 12,  1,  7,  3, 22,  8, 30, 30, 27, 26, 27, 27, 27,
         27, 31, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
         27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
         27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 11, 11, 27, 27, 27, 27, 27, 11,
         11,  8, 31, 31, 11,  8, 10, 18, 24,  8,  4,  7,  1, 13,  9, 25,  8, 15,
          8,  4, 21, 15,  6,  2, 30, 31, 31,  3,  9,  9, 25, 26, 18, 15,  5,  3,
         15,  2, 10, 22,  1,  6,  5, 13, 20, 31, 11, 19,  1,  9,  9,  9,  9, 24,
         24, 26,  8, 10, 30, 31, 30, 26, 26, 27, 26, 27, 26, 26, 26, 31, 15, 10,
          6, 18, 18, 18, 15, 13,  1,  6,  5, 13, 29, 11,  8,  5, 23, 31, 29, 21,
         31,  7,  6,  6, 16, 14, 10,  8, 20, 20,  6,  2,  2, 15, 15,  7, 15, 15,
         11, 12,  2,  6, 16, 14, 14,  8, 20, 23,  1,  9,  9,  6, 15, 13,  0,  9,
          9, 24, 24,  8, 27]], device='cuda:0')
あなたの言葉は,私の心を軽くしてくれた...他の誰もできなかったことだ...
```

## Requirements

- PyTorch
- librosa
- hydra-core
- omegaconf
- fire
- transformers 

## TODO

- [ ] Add more detailed descriptsion of the model
- [ ] Publish downstream applications
- [ ] Release training code