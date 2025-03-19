import librosa
import yaml
import hydra
from omegaconf import DictConfig
import torch
import fire
from hf_utils import load_custom_model_from_hf

HF_REPO_ID = "Plachta/ASTRAL-quantization"
HF_MODEL_PATH_MAPPINGS = {
    "bsq32": {
        "model_path": "bsq32/pytorch_model.bin",
        "config_path": "bsq32/config.yml",
    },
    "bsq2048": {
        "model_path": "bsq2048/pytorch_model.bin",
        "config_path": "bsq2048/config.yml",
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(model_path, config_path):
    config = DictConfig(yaml.safe_load(open(config_path, "r")))
    model = hydra.utils.instantiate(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model

@torch.no_grad()
def main(model_name=None, model_path=None, config_path=None, audio_path="test_waves/cafe_0.wav"):
    if model_name is not None:
        assert model_name in HF_MODEL_PATH_MAPPINGS, f"model_name must be one of {list(HF_MODEL_PATH_MAPPINGS.keys())}"
        model_path, config_path = load_custom_model_from_hf(
            HF_REPO_ID,
            model_filename=HF_MODEL_PATH_MAPPINGS[model_name]["model_path"],
            config_filename=HF_MODEL_PATH_MAPPINGS[model_name]["config_path"]
        )
    elif model_path is None or config_path is None:
        raise ValueError("model_path and config_path must be provided if model_name is not provided")
    model = init_model(model_path, config_path)
    model.eval()
    model.to(device)
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = torch.tensor(audio).to(device)

    waves_16k = audio.unsqueeze(0)
    waves_16k_lens = torch.tensor([len(audio)]).to(device)

    # encode audio to get speech content tokens, this can be processed in batch
    x_quantized, indices, feature_lens = model(waves_16k, waves_16k_lens)
    print(indices)

    # decode transcript from speech, this can only be processed one by one
    # this is not intended to be used but only to verify correctness, so no kv cache is implemented for now
    predicted_transcript = model.decode(waves_16k, waves_16k_lens)
    print(predicted_transcript)

if __name__ == "__main__":
    fire.Fire(main)