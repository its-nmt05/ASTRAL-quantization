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