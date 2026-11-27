"""Evaluate the S09 U-Net checkpoint on held-out SWE1 episodes (data/heldout/)."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import swe1
from train import CompactUNet

REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data" / "heldout"
RESULTS_DIR = SCRIPT_DIR / "results"
WEIGHTS_PATH = RESULTS_DIR / "unet.pt"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
OUT_JSON_PATH = RESULTS_DIR / "s10_eval.json"

KIND_ORDER = ("beach", "sandbar", "point", "reef")
ROLLOUT_STARTS = (20, 40, 60, 80)
ROLLOUT_STEPS = 20
HORIZON_STEPS = (2, 4, 8, 20)
SAMPLE_INTERVAL_S = 0.5
ONSET_ROLLOUT_STEPS = 60
BREAKING_RATIO = 0.78
TIMING_STEPS = 120
TIMING_REPEATS = 3

MODEL: CompactUNet | None = None
ETA_SCALE: float = 1.0
DEPTH_SCALE: float = 1.0


@torch.no_grad()
def predict_next(
    eta_prev: np.ndarray,
    eta_cur: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Predict next physical eta (m) of shape (96, 128) from (eta_prev, eta_cur, depth)."""
    assert MODEL is not None, "Model must be loaded before calling predict_next"
    h, w = depth.shape
    x = np.empty((1, 3, h, w), dtype=np.float32)
    eta_scale_f32 = np.float32(ETA_SCALE)
    depth_scale_f32 = np.float32(DEPTH_SCALE)
    np.divide(eta_cur, eta_scale_f32, out=x[0, 0])
    np.divide(eta_prev, eta_scale_f32, out=x[0, 1])
    np.divide(depth, depth_scale_f32, out=x[0, 2])
    pred_norm = MODEL(torch.from_numpy(x))
    out = pred_norm[0, 0].numpy().copy()
    out *= eta_scale_f32
    return out


def compute_predicted_onset_row(
    time_max: np.ndarray,
    depth: np.ndarray,
    breaking_ratio: float = BREAKING_RATIO,
) -> np.ndarray:
    """Compute per-column breaking onset row using the exact src/data/contract.ts rule."""
    height, width = depth.shape
    depth_f64 = depth.astype(np.float64)
    ratio = np.zeros((height, width), dtype=np.float64)
    wet = depth_f64 > 0.0
    ratio[wet] = (2.0 * time_max[wet]) / depth_f64[wet]

    onset_row = np.full(width, -1, dtype=np.int16)
    for x in range(width):
        for y in range(height):
            if depth_f64[y, x] > 0.0 and ratio[y, x] >= breaking_ratio:
                onset_row[x] = y
                break
    return onset_row


@torch.no_grad()
def main() -> None:
    global MODEL, ETA_SCALE, DEPTH_SCALE

    torch.set_num_threads(2)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    ETA_SCALE = float(summary["eta_scale"])
    DEPTH_SCALE = float(summary["depth_scale"])

    model = CompactUNet(in_channels=3, out_channels=1, base_channels=16)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    MODEL = model

    manifest = swe1.load_manifest(DATA_DIR)
    ep_entries = sorted(manifest["episodes"], key=lambda e: int(e["index"]))

    episodes: list[dict[str, Any]] = []
    for entry in ep_entries:
        ep_path = DATA_DIR / entry["file"]
        loaded = swe1.load_episode(ep_path, fields=("eta", "depth", "onsetRow"))
        episodes.append(
            {
                "index": int(entry["index"]),
                "file": str(entry["file"]),
                "kind": str(entry["kind"]),
                "seed": int(entry["seed"]),
                "swell": entry["swell"],
                "eta": loaded["eta"],
                "depth": loaded["depth"],
                "onsetRow": loaded["onsetRow"],
            }
        )

    # ------------------------------------------------------------------
    # Measurement A — one-step MSE (t = 1..119)
    # ------------------------------------------------------------------
    a_se_model_by_kind: dict[str, float] = {k: 0.0 for k in KIND_ORDER}
    a_se_pers_by_kind: dict[str, float] = {k: 0.0 for k in KIND_ORDER}
    a_cells_by_kind: dict[str, int] = {k: 0 for k in KIND_ORDER}

    for ep in episodes:
        kind = ep["kind"]
        eta: np.ndarray = ep["eta"]
        depth: np.ndarray = ep["depth"]
        num_frames = eta.shape[0]
        for t in range(1, num_frames - 1):
            pred = predict_next(eta[t - 1], eta[t], depth)
            target = eta[t + 1].astype(np.float64)
            diff_m = pred.astype(np.float64) - target
            diff_p = eta[t].astype(np.float64) - target
            a_se_model_by_kind[kind] += float(np.sum(diff_m * diff_m))
            a_se_pers_by_kind[kind] += float(np.sum(diff_p * diff_p))
            a_cells_by_kind[kind] += int(target.size)

    one_step_per_kind: dict[str, dict[str, float]] = {}
    total_se_m = 0.0
    total_se_p = 0.0
    total_cells = 0
    for kind in KIND_ORDER:
        cells = a_cells_by_kind[kind]
        m_mse = a_se_model_by_kind[kind] / cells
        p_mse = a_se_pers_by_kind[kind] / cells
        imp = 100.0 * (p_mse - m_mse) / p_mse if p_mse > 0.0 else 0.0
        one_step_per_kind[kind] = {
            "model_mse": m_mse,
            "persistence_mse": p_mse,
            "improvement_pct": imp,
        }
        total_se_m += a_se_model_by_kind[kind]
        total_se_p += a_se_pers_by_kind[kind]
        total_cells += cells

    overall_m_mse = total_se_m / total_cells
    overall_p_mse = total_se_p / total_cells
    overall_imp = (
        100.0 * (overall_p_mse - overall_m_mse) / overall_p_mse
        if overall_p_mse > 0.0
        else 0.0
    )
    one_step_overall = {
        "model_mse": overall_m_mse,
        "persistence_mse": overall_p_mse,
        "improvement_pct": overall_imp,
    }

    # ------------------------------------------------------------------
    # Measurement B — 20-step autoregressive rollout from s in {20,40,60,80}
    # ------------------------------------------------------------------
    horizon_set = set(HORIZON_STEPS)
    b_model_by_kind: dict[str, dict[int, list[float]]] = {
        k: {h: [] for h in HORIZON_STEPS} for k in KIND_ORDER
    }
    b_pers_by_kind: dict[str, dict[int, list[float]]] = {
        k: {h: [] for h in HORIZON_STEPS} for k in KIND_ORDER
    }
    b_model_all: dict[int, list[float]] = {h: [] for h in HORIZON_STEPS}
    b_pers_all: dict[int, list[float]] = {h: [] for h in HORIZON_STEPS}

    for ep in episodes:
        kind = ep["kind"]
        eta = ep["eta"]
        depth = ep["depth"]
        for s in ROLLOUT_STARTS:
            prev_eta = eta[s - 1]
            cur_eta = eta[s]
            frozen_eta = eta[s].astype(np.float64)
            for step in range(1, ROLLOUT_STEPS + 1):
                pred = predict_next(prev_eta, cur_eta, depth)
                if step in horizon_set:
                    target = eta[s + step].astype(np.float64)
                    diff_m = pred.astype(np.float64) - target
                    diff_p = frozen_eta - target
                    mse_m = float(np.mean(diff_m * diff_m))
                    mse_p = float(np.mean(diff_p * diff_p))
                    b_model_by_kind[kind][step].append(mse_m)
                    b_pers_by_kind[kind][step].append(mse_p)
                    b_model_all[step].append(mse_m)
                    b_pers_all[step].append(mse_p)
                prev_eta, cur_eta = cur_eta, pred

    horizons_s = [step * SAMPLE_INTERVAL_S for step in HORIZON_STEPS]
    rollout_model_mse = [float(np.mean(b_model_all[h])) for h in HORIZON_STEPS]
    rollout_pers_mse = [float(np.mean(b_pers_all[h])) for h in HORIZON_STEPS]
    rollout_per_kind = {
        kind: {
            "model_mse": [float(np.mean(b_model_by_kind[kind][h])) for h in HORIZON_STEPS],
            "persistence_mse": [float(np.mean(b_pers_by_kind[kind][h])) for h in HORIZON_STEPS],
        }
        for kind in KIND_ORDER
    }

    # ------------------------------------------------------------------
    # Measurement C — breaking onset from 60-step rollout (frames 2..61)
    # ------------------------------------------------------------------
    onset_deltas_by_kind: dict[str, list[int]] = {k: [] for k in KIND_ORDER}
    onset_counts_by_kind: dict[str, dict[str, int]] = {
        k: {"both_crossing": 0, "missed": 0, "false_onset": 0, "neither": 0}
        for k in KIND_ORDER
    }

    for ep in episodes:
        kind = ep["kind"]
        eta = ep["eta"]
        depth = ep["depth"]
        stored_onset: np.ndarray = ep["onsetRow"]

        prev_eta = eta[0]
        cur_eta = eta[1]
        time_max = np.zeros(depth.shape, dtype=np.float64)
        for _ in range(ONSET_ROLLOUT_STEPS):
            pred = predict_next(prev_eta, cur_eta, depth)
            np.maximum(time_max, np.abs(pred.astype(np.float64)), out=time_max)
            prev_eta, cur_eta = cur_eta, pred

        pred_onset = compute_predicted_onset_row(time_max, depth, BREAKING_RATIO)
        counts = onset_counts_by_kind[kind]
        deltas = onset_deltas_by_kind[kind]
        for x in range(stored_onset.shape[0]):
            s_row = int(stored_onset[x])
            p_row = int(pred_onset[x])
            if s_row >= 0 and p_row >= 0:
                deltas.append(abs(p_row - s_row))
                counts["both_crossing"] += 1
            elif s_row >= 0 and p_row == -1:
                counts["missed"] += 1
            elif s_row == -1 and p_row >= 0:
                counts["false_onset"] += 1
            else:
                counts["neither"] += 1

    onset_per_kind: dict[str, dict[str, Any]] = {}
    for kind in KIND_ORDER:
        deltas = onset_deltas_by_kind[kind]
        counts = onset_counts_by_kind[kind]
        onset_per_kind[kind] = {
            "median_abs_delta_row": float(np.median(deltas)) if deltas else None,
            "both_crossing": counts["both_crossing"],
            "missed": counts["missed"],
            "false_onset": counts["false_onset"],
            "neither": counts["neither"],
            "total_columns": (
                counts["both_crossing"]
                + counts["missed"]
                + counts["false_onset"]
                + counts["neither"]
            ),
        }

    # ------------------------------------------------------------------
    # Measurement D — U-Net inference cost (120 steps, batch 1, ep00, 3x)
    # ------------------------------------------------------------------
    ep00 = next(ep for ep in episodes if ep["index"] == 0)
    ep00_eta = ep00["eta"]
    ep00_depth = ep00["depth"]
    runs_s: list[float] = []
    for _ in range(TIMING_REPEATS):
        prev_eta = ep00_eta[0]
        cur_eta = ep00_eta[1]
        t0 = time.perf_counter()
        for _ in range(TIMING_STEPS):
            pred = predict_next(prev_eta, cur_eta, ep00_depth)
            prev_eta, cur_eta = cur_eta, pred
        runs_s.append(time.perf_counter() - t0)

    median_s = float(np.median(runs_s))
    ms_per_step = (median_s * 1000.0) / TIMING_STEPS

    inference_cost = {
        "episode": ep00["file"],
        "steps": TIMING_STEPS,
        "repeats": TIMING_REPEATS,
        "runs_s": runs_s,
        "median_s": median_s,
        "ms_per_step": ms_per_step,
    }

    # ------------------------------------------------------------------
    # Assemble & write JSON report
    # ------------------------------------------------------------------
    result = {
        "one_step": {
            "per_kind": one_step_per_kind,
            "overall": one_step_overall,
        },
        "rollout": {
            "steps": list(HORIZON_STEPS),
            "horizons_s": horizons_s,
            "start_frames": list(ROLLOUT_STARTS),
            "model_mse": rollout_model_mse,
            "persistence_mse": rollout_pers_mse,
            "per_kind": rollout_per_kind,
        },
        "onset": {
            "rollout_steps": ONSET_ROLLOUT_STEPS,
            "breaking_ratio": BREAKING_RATIO,
            "per_kind": onset_per_kind,
        },
        "inference_cost": inference_cost,
        "meta": {
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "threads": int(torch.get_num_threads()),
            "eta_scale": ETA_SCALE,
            "depth_scale": DEPTH_SCALE,
            "checkpoint": str(WEIGHTS_PATH.relative_to(REPO_ROOT)),
            "episodes": [ep["file"] for ep in episodes],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Human-readable summary table
    # ------------------------------------------------------------------
    print("=== S10 Held-Out Evaluation Summary ===")
    print(
        f"Torch: {torch.__version__} | Threads: {torch.get_num_threads()} | "
        f"Episodes: {len(episodes)} | ETA_SCALE={ETA_SCALE:.6f} m, DEPTH_SCALE={DEPTH_SCALE:.6f} m"
    )
    print()

    print("Measurement A — One-step MSE (physical m^2, t=1..119):")
    print(f"{'Kind':<10} {'Model MSE':>12} {'Persist MSE':>12} {'Improv %':>10}")
    for kind in KIND_ORDER:
        row = one_step_per_kind[kind]
        print(
            f"{kind:<10} {row['model_mse']:>12.6f} {row['persistence_mse']:>12.6f} "
            f"{row['improvement_pct']:>9.2f}%"
        )
    print(
        f"{'OVERALL':<10} {one_step_overall['model_mse']:>12.6f} "
        f"{one_step_overall['persistence_mse']:>12.6f} "
        f"{one_step_overall['improvement_pct']:>9.2f}%"
    )
    print()

    print("Measurement B — Autoregressive Rollout MSE (physical m^2, starts={20,40,60,80}):")
    header_h = " ".join(f"{h:>10.1f}s" for h in horizons_s)
    print(f"{'Series':<18} {header_h}")
    print(
        f"{'Overall model':<18} "
        + " ".join(f"{v:>11.6f}" for v in rollout_model_mse)
    )
    print(
        f"{'Overall persist':<18} "
        + " ".join(f"{v:>11.6f}" for v in rollout_pers_mse)
    )
    for kind in KIND_ORDER:
        km = rollout_per_kind[kind]["model_mse"]
        kp = rollout_per_kind[kind]["persistence_mse"]
        print(f"{kind + ' model':<18} " + " ".join(f"{v:>11.6f}" for v in km))
        print(f"{kind + ' persist':<18} " + " ".join(f"{v:>11.6f}" for v in kp))
    print()

    print("Measurement C — Breaking Onset from 60-Step Rollout (frames 2..61):")
    print(f"{'Kind':<10} {'Med |dRow|':>12} {'Both':>8} {'Missed':>8} {'False':>8} {'Neither':>8}")
    for kind in KIND_ORDER:
        row = onset_per_kind[kind]
        med_str = (
            f"{row['median_abs_delta_row']:.2f}"
            if row["median_abs_delta_row"] is not None
            else "N/A"
        )
        print(
            f"{kind:<10} {med_str:>12} {row['both_crossing']:>8d} "
            f"{row['missed']:>8d} {row['false_onset']:>8d} {row['neither']:>8d}"
        )
    print()

    print(f"Measurement D — U-Net Inference Cost ({ep00['file']}, {TIMING_STEPS} steps, batch 1):")
    runs_str = ", ".join(f"{r:.4f}s" for r in runs_s)
    print(f"  Runs (3x):   [{runs_str}]")
    print(f"  Median:      {median_s:.4f}s ({ms_per_step:.3f} ms/step)")
    print(f"\nWrote {OUT_JSON_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
