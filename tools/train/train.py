"""Small CPU U-Net next-step eta prediction training pipeline for surf-world SWE1 data."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import swe1


def split_episodes(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split manifest episodes into train (32) and val (12) by episode."""
    episodes = sorted(manifest["episodes"], key=lambda ep: int(ep["index"]))

    def is_val(ep: dict[str, Any]) -> bool:
        swell = ep["swell"]
        return math.isclose(float(swell["heightM"]), 1.8, abs_tol=1e-9) and math.isclose(
            float(swell["periodS"]), 9.0, abs_tol=1e-9
        )

    val_episodes = [ep for ep in episodes if is_val(ep)]

    train_by_kind: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        if is_val(ep):
            continue
        kind = str(ep["kind"])
        bucket = train_by_kind.setdefault(kind, [])
        if len(bucket) < 8:
            bucket.append(ep)

    train_episodes = sorted(
        (ep for bucket in train_by_kind.values() for ep in bucket),
        key=lambda ep: int(ep["index"]),
    )

    train_files = {ep["file"] for ep in train_episodes}
    val_files = {ep["file"] for ep in val_episodes}
    if not train_files.isdisjoint(val_files):
        raise RuntimeError("Train and val episode splits overlap!")

    return train_episodes, val_episodes


class EpisodeCache:
    """Small LRU cache that lazily loads only eta and depth, verifying SHA-256 on first load."""

    def __init__(self, max_size: int = 8) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._verified: dict[Path, tuple[dict[str, Any], int]] = {}

    def get(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        if path not in self._verified:
            ep = swe1.load_episode(path, fields=("eta", "depth"))
            header, payload_offset = swe1.read_header(path)
            self._verified[path] = (header, payload_offset)
            item = (ep["eta"], ep["depth"])
        else:
            header, payload_offset = self._verified[path]
            fields = swe1.load_episode_fields_fast(
                path, header, payload_offset, fields=("eta", "depth")
            )
            item = (fields["eta"], fields["depth"])

        self._cache[path] = item
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return item

    def get_num_frames(self, path: Path) -> int:
        if path in self._verified:
            header, _ = self._verified[path]
            return int(header["episode"]["numFrames"])
        header, _ = swe1.read_header(path)
        return int(header["episode"]["numFrames"])


def compute_normalization_scales(
    train_paths: list[Path], cache: EpisodeCache
) -> tuple[float, float]:
    """Compute global ETA_SCALE (std of eta) and DEPTH_SCALE (max depth) over train episodes."""
    total_count = 0
    total_mean = 0.0
    total_m2 = 0.0
    max_depth = -math.inf

    for path in train_paths:
        eta, depth = cache.get(path)
        arr = eta.astype(np.float64, copy=False).ravel()
        n_b = arr.size
        mean_b = float(arr.mean())
        m2_b = float(np.sum((arr - mean_b) ** 2))

        if total_count == 0:
            total_count = n_b
            total_mean = mean_b
            total_m2 = m2_b
        else:
            new_count = total_count + n_b
            delta = mean_b - total_mean
            total_mean += delta * (n_b / new_count)
            total_m2 += m2_b + (delta * delta) * (total_count * n_b / new_count)
            total_count = new_count

        ep_max_depth = float(depth.max())
        if ep_max_depth > max_depth:
            max_depth = ep_max_depth

    eta_scale = math.sqrt(total_m2 / total_count)
    if eta_scale <= 0.0 or not math.isfinite(eta_scale):
        raise ValueError(f"Invalid ETA_SCALE computed: {eta_scale}")
    if max_depth <= 0.0 or not math.isfinite(max_depth):
        raise ValueError(f"Invalid DEPTH_SCALE computed: {max_depth}")

    return eta_scale, max_depth


class NextStepDataset(Dataset):
    """Dataset of (input=[eta_t, eta_{t-1}, depth], target=eta_{t+1}) normalized tensors."""

    def __init__(
        self,
        episode_paths: list[Path],
        stride: int,
        eta_scale: float,
        depth_scale: float,
        cache: EpisodeCache,
    ) -> None:
        self.episode_paths = episode_paths
        self.eta_scale = float(eta_scale)
        self.depth_scale = float(depth_scale)
        self.cache = cache

        self.samples: list[tuple[int, int]] = []
        for ep_idx, path in enumerate(episode_paths):
            num_frames = cache.get_num_frames(path)
            for t in range(1, num_frames - 1, stride):
                self.samples.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ep_idx, t = self.samples[index]
        eta, depth = self.cache.get(self.episode_paths[ep_idx])

        inv_eta = np.float32(1.0 / self.eta_scale)
        inv_depth = np.float32(1.0 / self.depth_scale)

        x = np.empty((3, eta.shape[1], eta.shape[2]), dtype=np.float32)
        np.multiply(eta[t], inv_eta, out=x[0])
        np.multiply(eta[t - 1], inv_eta, out=x[1])
        np.multiply(depth, inv_depth, out=x[2])

        y = np.empty((1, eta.shape[1], eta.shape[2]), dtype=np.float32)
        np.multiply(eta[t + 1], inv_eta, out=y[0])

        return torch.from_numpy(x), torch.from_numpy(y)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with GroupNorm and SiLU activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompactUNet(nn.Module):
    """Compact 2-stage U-Net (128x96 -> 64x48 -> 32x24) with GroupNorm + SiLU."""

    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(c2, c3)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c3, c2, kernel_size=1),
        )
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c2, c1, kernel_size=1),
        )
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        b = self.bottleneck(self.pool2(s2))

        d2 = self.dec2(torch.cat([self.up2(b), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.head(d1)


@torch.no_grad()
def evaluate(
    model: nn.Module, val_loader: DataLoader, eta_scale: float
) -> tuple[float, float]:
    """Evaluate model MSE and persistence baseline MSE in physical units (m^2)."""
    model.eval()
    model_se_sum = 0.0
    persist_se_sum = 0.0
    total_elems = 0

    for x, y in val_loader:
        pred = model(x)
        model_se_sum += float(torch.sum((pred - y) ** 2).item())
        # Channel 0 of x is normalized eta_t; y is normalized eta_{t+1}.
        persist_se_sum += float(torch.sum((x[:, 0:1] - y) ** 2).item())
        total_elems += int(y.numel())

    scale_sq = eta_scale * eta_scale
    val_mse_phys = (model_se_sum / total_elems) * scale_sq
    persistence_mse_phys = (persist_se_sum / total_elems) * scale_sq
    return val_mse_phys, persistence_mse_phys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact CPU U-Net for next-step eta prediction on SWE1 episodes."
    )
    parser.add_argument(
        "--epochs", type=int, default=12, help="Number of training epochs (default: 12)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/full"),
        help="Directory containing SWE1 episodes and manifest.json (default: data/full).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tools/train/results"),
        help="Output directory for history.json, summary.json, and unet.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = swe1.load_manifest(data_dir)
    train_eps, val_eps = split_episodes(manifest)

    train_paths = [data_dir / ep["file"] for ep in train_eps]
    val_paths = [data_dir / ep["file"] for ep in val_eps]

    print(
        f"Loaded manifest from {data_dir}: "
        f"{len(train_eps)} train episodes, {len(val_eps)} val episodes"
    )

    cache = EpisodeCache(max_size=8)
    eta_scale, depth_scale = compute_normalization_scales(train_paths, cache)
    print(f"Normalization scales: ETA_SCALE={eta_scale:.6f} m, DEPTH_SCALE={depth_scale:.6f} m")

    train_dataset = NextStepDataset(
        episode_paths=train_paths,
        stride=2,
        eta_scale=eta_scale,
        depth_scale=depth_scale,
        cache=cache,
    )
    val_dataset = NextStepDataset(
        episode_paths=val_paths,
        stride=4,
        eta_scale=eta_scale,
        depth_scale=depth_scale,
        cache=cache,
    )
    print(
        f"Samples: train={len(train_dataset)} per epoch (stride=2), "
        f"val={len(val_dataset)} (stride=4)"
    )

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        generator=shuffle_gen,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    model = CompactUNet(in_channels=3, out_channels=1, base_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    history: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_se_sum = 0.0
        train_elems = 0

        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            batch_elems = int(y.numel())
            train_se_sum += float(loss.item()) * batch_elems
            train_elems += batch_elems

        train_mse_norm = train_se_sum / train_elems
        val_mse_phys, persistence_mse_phys = evaluate(model, val_loader, eta_scale)
        epoch_wall_s = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "train_mse_norm": train_mse_norm,
            "val_mse_phys": val_mse_phys,
            "persistence_mse_phys": persistence_mse_phys,
            "epoch_wall_s": epoch_wall_s,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_mse_norm={train_mse_norm:.6f} | "
            f"val_mse_phys={val_mse_phys:.6f} m^2 | "
            f"persistence_mse_phys={persistence_mse_phys:.6f} m^2 | "
            f"wall={epoch_wall_s:.2f}s",
            flush=True,
        )

    total_wall_s = time.perf_counter() - total_start
    if history:
        final_val_mse_phys = history[-1]["val_mse_phys"]
        final_persistence_mse_phys = history[-1]["persistence_mse_phys"]
    else:
        final_val_mse_phys, final_persistence_mse_phys = evaluate(model, val_loader, eta_scale)

    improvement_pct = (
        100.0 * (final_persistence_mse_phys - final_val_mse_phys) / final_persistence_mse_phys
        if final_persistence_mse_phys > 0.0
        else 0.0
    )

    summary = {
        "eta_scale": eta_scale,
        "depth_scale": depth_scale,
        "train_episodes": [ep["file"] for ep in train_eps],
        "val_episodes": [ep["file"] for ep in val_eps],
        "train_samples_per_epoch": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_run": args.epochs,
        "total_wall_s": total_wall_s,
        "final_val_mse_phys": final_val_mse_phys,
        "persistence_mse_phys": final_persistence_mse_phys,
        "improvement_pct": improvement_pct,
    }

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "unet.pt")

    print(
        f"Done in {total_wall_s:.2f}s | final_val_mse_phys={final_val_mse_phys:.6f} m^2 | "
        f"persistence={final_persistence_mse_phys:.6f} m^2 | "
        f"improvement={improvement_pct:.2f}% | saved to {out_dir}"
    )


if __name__ == "__main__":
    main()
