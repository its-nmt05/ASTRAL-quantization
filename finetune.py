import torch
import torch.nn as nn
import torch.optim as optim

import librosa
from main import init_model
from hf_utils import load_custom_model_from_hf

from datasets import load_dataset
from torch.utils.data import DataLoader

n_epochs = 100
model_name = "bsq32"

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

model_path, config_path = load_custom_model_from_hf(
            HF_REPO_ID,
            model_filename=HF_MODEL_PATH_MAPPINGS[model_name]["model_path"],
            config_filename=HF_MODEL_PATH_MAPPINGS[model_name]["config_path"]
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()

def pad_or_truncate(audio_tensor, target_length):
    """Pads or truncates audio to a fixed length."""
    length = audio_tensor.shape[-1]
    if length > target_length:
        return audio_tensor[:target_length]  # Truncate
    elif length < target_length:
        padding = torch.zeros((target_length - length))
        return torch.cat((audio_tensor, padding), dim=-1)  # Pad
    return audio_tensor

def collate_fn(batch, target_length=160000):  # Set an appropriate length
    """Collate function to pad/truncate audio in a batch."""
    audio_tensors = [pad_or_truncate(b["audio"]["array"], target_length) for b in batch]
    transcriptions = [b["transcription"] for b in batch]
    
    return {
        "audio": torch.stack(audio_tensors),
        "transcription": transcriptions
    }

def load_dataloader(batch_size=4):
    dataset = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="train")
    dataset = dataset.select(range(100)) 
    dataset.set_format(type="torch", columns=["audio", "transcription"])
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
def load_model(model_path, config_path):
    model = init_model(model_path, config_path)
    for param in model.parameters():    # freeze all other params except asr_decoder
        param.requires_grad = False
    for param in model.asr_decoder.parameters():
        param.requires_grad = True
    return model

def train(n_epochs, model, dataloader):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    for i in range(n_epochs):
        epoch_loss = 0
        epoch_losses = []
        num_batches = len(dataloader)
        for batch in dataloader:
            waves_16k, transcriptions = batch['audio'], batch['transcription']
            waves_16k = waves_16k.to(device)            
            waves_16k_lens = torch.tensor([len(w) for w in waves_16k]).to(device)

            target_tokens = model.tokenizer(transcriptions, padding=True, return_tensors='pt').input_ids
            target_tokens = target_tokens.to(device)
            text_lens = torch.tensor(list(map(len, target_tokens)))
            
            with torch. no_grad():
                x_quantized, indices, feature_lens = model.forward(waves_16k, waves_16k_lens)
            
            s2s_loss = model.asr_decoder(x_quantized, feature_lens, target_tokens, text_lens)

            s2s_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += s2s_loss.item()
        
        epoch_loss /= num_batches
        epoch_losses.append(epoch_loss          )   

        with open("losses.txt", "a") as file:
            file.write(f"Epoch {i}: Loss = {epoch_loss}\n")

        if i % 10 == 0:
            print(f"Epoch: {i}, Loss: {epoch_loss}")
            torch.save(model.state_dict(), f"saves_epoch_{i}_model.pth")
            
model = load_model(model_path, config_path).to(device)
dataloader = load_dataloader()
train(n_epochs, model, dataloader)
