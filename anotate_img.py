#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg


# =========================
# USER KNOBS
# =========================
PIER_PNG = "img.png"

DAYS_TO_SHOW = 2
WINDOW_END = None  # None means end at last sample, or use "2026-01-18 12:00"

# Reference line
REF_LABEL = "MLLW"
REF_VALUE = -1.4

# Optional: use this to clip the wave band so it doesn't go above a "deck" elevation
CLIP_TO_DECK = True
DECK_ELEV = -1.25  # clipping elevation (can be different from where you draw the image)

# Pier placement and size
PIER_X0_FRAC = 0.50      # where pier starts (0..1) across the shown time window
PIER_WIDTH_FRAC = 0.33   # pier width as fraction of shown time window (0..1)
PIER_HEIGHT_M = 2.35     # pier height in y-axis units (meters)

# NEW: direct Y position control
PIER_Y_TOP = 0.2         # put the TOP of the image at y = 0.2
# If you prefer center instead of top, set PIER_Y_CENTER and flip the USE_CENTER flag
USE_CENTER = False
PIER_Y_CENTER = 0.2

# Data scaling (make signal less extreme)
TIDE_SCALE = 0.5
WAVE_SCALE = 0.15

# Image background handling
REMOVE_BG = True
BG_THRESHOLD = 225  # try 220..240


# =========================
# DATA (FAKE)
# =========================
def make_fake_data(
    start="2026-01-12 00:00",
    end="2026-01-18 12:00",
    freq="5min",
    seed=7,
    tide_scale=1.0,
    wave_scale=1.0,
):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start=start, end=end, freq=freq)
    hours = (t - t[0]).total_seconds().to_numpy() / 3600.0

    semi_diurnal = 0.30 * np.sin(2 * np.pi * hours / 12.42)
    diurnal = 0.10 * np.sin(2 * np.pi * hours / 24.0 + 1.2)
    setup = 0.12 * np.sin(2 * np.pi * hours / (24.0 * 3.0) - 0.8)

    tide = tide_scale * (semi_diurnal + diurnal + setup)
    eta_mean = -1.35 + tide + 0.02 * rng.standard_normal(len(t))

    wave_amp = 0.45 + 0.18 * np.sin(2 * np.pi * hours / 36.0 - 0.3)
    wave_amp += 0.08 * np.sin(2 * np.pi * hours / 8.0 + 0.9)
    wave_amp = np.clip(wave_amp, 0.25, 0.85)
    wave_amp = wave_scale * wave_amp

    eta_low = eta_mean - wave_amp + 0.02 * rng.standard_normal(len(t))
    eta_high = eta_mean + wave_amp + 0.02 * rng.standard_normal(len(t))

    return pd.DataFrame({"time": t, "eta_mean": eta_mean, "eta_low": eta_low, "eta_high": eta_high})


def subset_last_days(df, days=2, window_end=None):
    df = df.sort_values("time").reset_index(drop=True)
    t1 = df["time"].iloc[-1] if window_end is None else pd.to_datetime(window_end)
    t0 = t1 - pd.Timedelta(days=days)
    out = df[(df["time"] >= t0) & (df["time"] <= t1)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


# =========================
# IMAGE HELPERS
# =========================
def _to_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (img * 255).round().astype(np.uint8)
    return img.astype(np.uint8)


def remove_nearwhite_background_to_alpha(img, threshold=225):
    img8 = _to_uint8(img)

    if img8.ndim == 2:
        rgb = np.stack([img8, img8, img8], axis=-1)
        a = np.full(img8.shape, 255, dtype=np.uint8)
        rgba = np.dstack([rgb, a])
    elif img8.shape[2] == 3:
        a = np.full(img8.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([img8, a])
    else:
        rgba = img8.copy()

    rgb = rgba[:, :, :3].astype(np.int16)
    near_white = (
        (rgb[:, :, 0] >= threshold)
        & (rgb[:, :, 1] >= threshold)
        & (rgb[:, :, 2] >= threshold)
    )
    rgba[:, :, 3][near_white] = 0
    return rgba


# =========================
# PLOT
# =========================
def plot_with_overlay(df, pier_png_path):
    df = df.sort_values("time").reset_index(drop=True)

    t = df["time"]
    eta_mean = df["eta_mean"].to_numpy()
    eta_low = df["eta_low"].to_numpy()
    eta_high = df["eta_high"].to_numpy()

    if CLIP_TO_DECK:
        eta_high_plot = np.minimum(eta_high, DECK_ELEV)
    else:
        eta_high_plot = eta_high

    pier_png_path = Path(pier_png_path)
    if not pier_png_path.exists():
        raise FileNotFoundError(f"Could not find PNG: {pier_png_path}")

    pier_img = mpimg.imread(str(pier_png_path))
    if REMOVE_BG:
        pier_img = remove_nearwhite_background_to_alpha(pier_img, threshold=BG_THRESHOLD)

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 350,
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(12.5, 5.2))

    # ---- x placement ----
    t0, t1 = t.iloc[0], t.iloc[-1]
    span = t1 - t0
    x0 = t0 + span * PIER_X0_FRAC
    x1 = t0 + span * min(PIER_X0_FRAC + PIER_WIDTH_FRAC, 0.999)

    # ---- y placement (THIS is what you asked for) ----
    if USE_CENTER:
        pier_y0 = PIER_Y_CENTER - 0.5 * PIER_HEIGHT_M
        pier_y1 = PIER_Y_CENTER + 0.5 * PIER_HEIGHT_M
    else:
        pier_y1 = PIER_Y_TOP
        pier_y0 = pier_y1 - PIER_HEIGHT_M

    # ---- y limits: ensure image is visible even if data is far below ----
    y_min = float(np.nanmin(eta_low))
    y_max = float(np.nanmax(eta_high))
    y_low = min(y_min - 0.6, pier_y0 - 0.2)
    y_high = max(y_max + 0.6, pier_y1 + 0.2)
    ax.set_ylim(y_low, y_high)

    # ---- draw pier behind data ----
    ax.imshow(
        pier_img,
        extent=[mdates.date2num(x0), mdates.date2num(x1), pier_y0, pier_y1],
        aspect="auto",
        zorder=1,
        alpha=1.0,
    )

    # ---- data on top ----
    ax.fill_between(t, eta_low, eta_high_plot, alpha=0.30, linewidth=0, label="Wave Action", zorder=5)
    ax.plot(t, eta_mean, linewidth=2.4, label="Water Level", zorder=6)
    ax.axhline(REF_VALUE, linestyle=(0, (6, 4)), linewidth=2.0, label=REF_LABEL, zorder=4)

    if CLIP_TO_DECK:
        ax.axhline(DECK_ELEV, linewidth=1.2, alpha=0.35, zorder=3)

    #ax.set_title("Water-level Relative to Fishing Pier", pad=12)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Water Level (m)")

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.5)

    ax.legend(loc="upper right", frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig("pier_water_level_2days.png", bbox_inches="tight")
    fig.savefig("pier_water_level_2days.pdf", bbox_inches="tight")
    plt.show()

    print("Saved: pier_water_level_2days.png")
    print("Saved: pier_water_level_2days.pdf")


if __name__ == "__main__":
    df = make_fake_data(tide_scale=TIDE_SCALE, wave_scale=WAVE_SCALE)
    df2 = subset_last_days(df, days=DAYS_TO_SHOW, window_end=WINDOW_END)
    plot_with_overlay(df2, PIER_PNG)
