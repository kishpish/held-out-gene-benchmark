"""Enformer backbone wrapper (pretrained, zero-shot readout).

Enformer: 196,608 bp input -> 896 output bins of 128 bp (central 114,688 bp) x 5,313 human tracks.
The window is TSS-centered, so the TSS falls in the central output bin (index 448).

Key LCL-relevant track indices (from Basenji/Enformer targets_human.txt):
  5110 = CAGE:B lymphoblastoid cell line: GM12878   <- primary expression proxy
    12 = DNASE:GM12878
   686 = CHIP:H3K4me3:GM12878
   688 = CHIP:H3K27ac:GM12878
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

N_BINS = 896
CENTER_BIN = N_BINS // 2          # 448; TSS bin for a TSS-centered window
LCL_CAGE = 5110
LCL_TRACKS = {"CAGE_GM12878": 5110, "DNASE_GM12878": 12,
              "H3K4me3_GM12878": 686, "H3K27ac_GM12878": 688}

_MODEL = None


def load_enformer(model_id: str = "EleutherAI/enformer-official-rough", device: str = "cuda"):
    """Load & cache the pretrained Enformer (weights cached under /ephemeral via ~/.cache)."""
    global _MODEL
    if _MODEL is None:
        from enformer_pytorch import Enformer
        m = Enformer.from_pretrained(model_id)
        m.eval().to(device)
        _MODEL = m
    return _MODEL


@torch.no_grad()
def predict_human(one_hot_seq: np.ndarray, device: str = "cuda",
                  use_bf16: bool = True) -> np.ndarray:
    """Forward pass on a single (196608, 4) one-hot -> (896, 5313) float32 human head."""
    model = load_enformer(device=device)
    x = torch.from_numpy(one_hot_seq).to(device).unsqueeze(0)  # (1, L, 4)
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
    else:
        out = model(x)
    human = out["human"] if isinstance(out, dict) else out
    return human.float().squeeze(0).cpu().numpy()   # (896, 5313)


@torch.no_grad()
def predict_human_batch(one_hot_batch: np.ndarray, device: str = "cuda",
                        use_bf16: bool = True) -> np.ndarray:
    """Forward pass on a batch (B, 196608, 4) one-hot -> (B, 896, 5313) float32 human head."""
    model = load_enformer(device=device)
    x = torch.from_numpy(one_hot_batch).to(device)
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
    else:
        out = model(x)
    human = out["human"] if isinstance(out, dict) else out
    return human.float().cpu().numpy()


@dataclass
class Readout:
    tracks: dict          # name -> value at TSS bin
    tss_bin: int


def read_tss(pred_human: np.ndarray, track_map: dict = LCL_TRACKS,
             bin_idx: int = CENTER_BIN) -> Readout:
    return Readout(tracks={k: float(pred_human[bin_idx, v]) for k, v in track_map.items()},
                   tss_bin=bin_idx)
