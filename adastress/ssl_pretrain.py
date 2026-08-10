"""
Stage 1: self-supervised contrastive pretraining of the physiological encoder.

Two augmented views of the same window are pulled together and pushed apart
from every other window in the batch (NT-Xent / SimCLR-style loss). No
labels are used here at all, this stage only needs the raw signal.

Usage:
    python -m adastress.ssl_pretrain --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import os

import joblib
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .datasets import ContrastiveWindowDataset, prepare_unlabelled_windows
from .models import Encoder, ProjectionHead
from .utils import ensure_dir, save_json, set_seed


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)

    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels])
    return F.cross_entropy(sim, labels)


def run(cfg) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")

    windows, scaler = prepare_unlabelled_windows(
        csv_path=cfg.data.unlabelled_csv,
        signal_columns=cfg.data.signal_columns,
        window_size=cfg.window.window_sec * cfg.window.fs,
        stride=cfg.window.stride_sec * cfg.window.fs,
    )
    print(f"Prepared {windows.shape[0]} unlabelled windows of shape {windows.shape[1:]}")

    dataset = ContrastiveWindowDataset(windows)
    loader = DataLoader(dataset, batch_size=cfg.ssl.batch_size, shuffle=True, drop_last=True)

    encoder = Encoder(in_channels=windows.shape[1], lstm_hidden=cfg.ssl.lstm_hidden).to(device)
    projector = ProjectionHead(encoder.output_dim, proj_dim=cfg.ssl.proj_dim).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=cfg.ssl.lr)

    for epoch in range(cfg.ssl.epochs):
        encoder.train()
        projector.train()
        total_loss = 0.0

        for x1, x2 in tqdm(loader, desc=f"SSL epoch {epoch + 1}/{cfg.ssl.epochs}"):
            x1, x2 = x1.to(device), x2.to(device)

            h1, h2 = encoder(x1), encoder(x2)
            z1, z2 = projector(h1), projector(h2)

            loss = nt_xent_loss(z1, z2, cfg.ssl.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: avg NT-Xent loss = {total_loss / len(loader):.4f}")

    ensure_dir(cfg.paths.checkpoints_dir)
    torch.save(encoder.state_dict(), cfg.paths.encoder_ckpt)
    joblib.dump(scaler, cfg.paths.scaler_ckpt)
    save_json(
        {
            "columns": cfg.data.signal_columns,
            "window_sec": cfg.window.window_sec,
            "stride_sec": cfg.window.stride_sec,
            "fs": cfg.window.fs,
            "window_size": cfg.window.window_sec * cfg.window.fs,
            "input_channels": windows.shape[1],
            "embed_dim": encoder.output_dim,
        },
        cfg.paths.meta_json,
    )
    print(f"Saved encoder, scaler, and meta.json to {cfg.paths.checkpoints_dir}")


def main():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining of the physiological encoder.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
