"""Minimal Whisper encoder in PyTorch, loaded from local Hugging Face weights."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

WHISPER_SR = 16_000
"""Whisper-tiny expected sample rate."""
N_FFT = 400
"""STFT window size matching the Hugging Face preprocessor."""
HOP_LENGTH = 160
"""STFT hop matching the Hugging Face preprocessor (10 ms at 16 kHz)."""
ENCODER_CONV_STRIDE = 2
"""Stride of `conv2`; each encoder step covers this many STFT hops."""
ENCODER_FRAME_SECONDS = (HOP_LENGTH * ENCODER_CONV_STRIDE) / WHISPER_SR
"""Audio seconds per encoder time step (~20 ms, 50 steps/s)."""


class WhisperSelfAttention(nn.Module):
    """Multi-head self-attention used inside each Whisper encoder layer."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """Build Q/K/V projections for `n_heads` of width `d_model`."""
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Attend over time and project back to `d_model`."""
        batch, steps, width = hidden.shape
        query = self.q_proj(hidden).view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        merged = attended.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.out_proj(merged)


class WhisperEncoderLayer(nn.Module):
    """Pre-norm self-attention plus GELU feed-forward block."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int) -> None:
        """Build one encoder block with residual attention and FFN."""
        super().__init__()
        self.self_attn = WhisperSelfAttention(d_model, n_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply residual self-attention and the feed-forward network."""
        residual = hidden
        hidden = residual + self.self_attn(self.self_attn_layer_norm(hidden))
        residual = hidden
        hidden = residual + self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden))))
        return hidden


class WhisperEncoder(nn.Module):
    """Convolutional frontend plus stacked encoder layers (no decoder)."""

    def __init__(self, config: dict[str, object]) -> None:
        """Construct the encoder from a Hugging Face `config.json` dict."""
        super().__init__()
        d_model = int(config["d_model"])
        n_heads = int(config["encoder_attention_heads"])
        ffn_dim = int(config["encoder_ffn_dim"])
        n_layers = int(config["encoder_layers"])
        n_mels = int(config["num_mel_bins"])
        max_source_positions = int(config["max_source_positions"])
        self.max_source_positions = max_source_positions
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(max_source_positions, d_model)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode log-mel features to hidden states of shape `(batch, time, d_model)`."""
        hidden = F.gelu(self.conv1(input_features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)
        steps = min(hidden.shape[1], self.max_source_positions)
        hidden = hidden[:, :steps]
        hidden = hidden + self.embed_positions.weight[:steps]
        for layer in self.layers:
            hidden = layer(hidden)
        return self.layer_norm(hidden)


def log_mel_spectrogram(
    audio: torch.Tensor,
    mel_filters: torch.Tensor,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """Compute Whisper log-mel features for a 16 kHz waveform."""
    window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft.abs().pow(2)
    if magnitudes.shape[0] != mel_filters.shape[1]:
        magnitudes = magnitudes[: mel_filters.shape[1]]
    mel = mel_filters @ magnitudes
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def resample_to_16k(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Linearly resample a 1-D waveform to 16 kHz when needed."""
    if sample_rate == WHISPER_SR:
        return audio
    length = max(int(round(audio.numel() * WHISPER_SR / sample_rate)), 1)
    return F.interpolate(
        audio.view(1, 1, -1), size=length, mode="linear", align_corners=False
    ).view(-1)


def load_encoder(model_dir: Path, device: torch.device) -> tuple[WhisperEncoder, torch.Tensor]:
    """Load frozen encoder weights and mel filters from a local snapshot."""
    from safetensors.torch import load_file

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    preprocessor = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    encoder = WhisperEncoder(config)
    weights = load_file(str(model_dir / "model.safetensors"))
    encoder_weights = {
        key.removeprefix("model.encoder."): value
        for key, value in weights.items()
        if key.startswith("model.encoder.")
    }
    encoder.load_state_dict(encoder_weights)
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    mel_filters = torch.tensor(preprocessor["mel_filters"], dtype=torch.float32, device=device)
    return encoder, mel_filters
