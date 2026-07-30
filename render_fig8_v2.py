#!/usr/bin/env python3
"""Generate new Figure 8: 2x3 gallery of 6 selected moving solitons.

#142 morse/relu_like (tadpole)
#69  sinc/step_function (ring)
#136 elliptical/sigmoid_pair (crescent)
#140 rbf_mixture/laplace_peak (pac-man)
#151 polynomial/sigmoid_pair (original kernel + new growth)
bessel/gaussian_bell (robustness champion, 9/9)

Inferno colormap, black bg, 80x80 crop at t=5000.
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localized_search_v4_locomotion import AnisotropicLenia

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RES = 256
STEPS = 5000
CROP = 80

CREATURES = [
    {
        "label": "morse / relu_like",
        "sublabel": "tadpole, d=75, c=0.72",
        "genome": {
            "kernel_type": "morse", "growth_type": "relu_like", "geometry": "flat_plane",
            "resolution": 256, "kernel_radius": 20.278144951053108, "kernel_peaks": 0.9056801391770588,
            "growth_mu": 0.12471621858819384, "growth_sigma": 0.037866588257700104,
            "dt": 0.2207736733736189, "aniso_strength": 0.4648416110974048,
            "aniso_angle": 3.9184863096721143,
        },
    },
    {
        "label": "sinc / step_function",
        "sublabel": "ring, d=47, c=0.95",
        "genome": {
            "kernel_type": "sinc", "growth_type": "step_function", "geometry": "flat_plane",
            "resolution": 256, "kernel_radius": 17.329005919231342, "kernel_peaks": 1.928142899902756,
            "growth_mu": 0.25179096477365187, "growth_sigma": 0.052387161505028146,
            "dt": 0.15877548528946211, "aniso_strength": 0.5424616732757163,
            "aniso_angle": 6.283185307179586,
        },
    },
    {
        "label": "elliptical / sigmoid_pair",
        "sublabel": "crescent, d=55, c=0.96",
        "genome": {
            "kernel_type": "elliptical", "growth_type": "sigmoid_pair", "geometry": "flat_plane",
            "resolution": 256, "kernel_radius": 18.04439809217401, "kernel_peaks": 0.5466939870525485,
            "growth_mu": 0.3307144552076915, "growth_sigma": 0.07593782529552982,
            "dt": 0.28152222305643393, "aniso_strength": 0.4982660797914147,
            "aniso_angle": 1.0044911425827787,
        },
    },
    {
        "label": "rbf_mixture / laplace_peak",
        "sublabel": "pac-man, d=67, c=0.96",
        "genome": {
            "kernel_type": "rbf_mixture", "growth_type": "laplace_peak", "geometry": "flat_plane",
            "resolution": 256, "kernel_radius": 21.95705631082839, "kernel_peaks": 0.8745980280195933,
            "growth_mu": 0.3192335046943549, "growth_sigma": 0.09418289398760224,
            "dt": 0.2851525624363028, "aniso_strength": 0.2632268108845566,
            "aniso_angle": 4.006134506909816,
        },
    },
    {
        "label": "polynomial / sigmoid_pair",
        "sublabel": "ring, d=38, c=0.95\n(original kernel)",
        "genome": {
            "kernel_type": "polynomial", "growth_type": "sigmoid_pair", "geometry": "flat_plane",
            "resolution": 256, "kernel_radius": 23.052089770196005, "kernel_peaks": 1.0092416105234698,
            "growth_mu": 0.20397099274055341, "growth_sigma": 0.029717258845237036,
            "dt": 0.02, "aniso_strength": 0.17248495510848028,
            "aniso_angle": 4.9562436320405165,
        },
    },
    {
        "label": "bessel / gaussian_bell",
        "sublabel": "ring, 9/9 robust\n(robustness champion)",
        "genome": {
            "kernel_type": "bessel", "growth_type": "gaussian_bell", "geometry": "flat_plane",
            "resolution": 256, "kernel_radius": 19.781727907414716, "kernel_peaks": 2.3706531180187023,
            "growth_mu": 0.20070798643942994, "growth_sigma": 0.02896182105922773,
            "dt": 0.02857361828857563, "aniso_strength": 0.21452527791620304,
            "aniso_angle": 1.7911590897613312,
        },
    },
]


def init_state(sim):
    YY, XX = torch.meshgrid(
        torch.arange(RES, device=sim.device, dtype=torch.float32),
        torch.arange(RES, device=sim.device, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = RES * 0.35, RES * 0.5
    w = RES * 0.06
    r2 = ((XX - cx) / w) ** 2 + ((YY - cy) / (w / 2.0)) ** 2
    sim.state = (torch.exp(-r2) * 0.8).to(sim.device)


def find_centroid(state):
    Y, X = np.mgrid[0:RES, 0:RES]
    total = state.sum()
    if total < 1e-6:
        return RES // 2, RES // 2
    return int(round((X * state).sum() / total)), int(round((Y * state).sum() / total))


def crop_periodic(state, cx, cy, size=CROP):
    half = size // 2
    padded = np.tile(state, (3, 3))
    px, py = cx + RES, cy + RES
    return padded[py - half:py + half, px - half:px + half]


def simulate(genome):
    torch.manual_seed(42)
    sim = AnisotropicLenia(genome, device=DEVICE)
    init_state(sim)
    for _ in range(STEPS):
        sim.step()
    return sim.state.cpu().numpy()


def main():
    print("Simulating 6 creatures...")
    crops = []
    for i, c in enumerate(CREATURES):
        print(f"  [{i+1}/6] {c['label']}...")
        state = simulate(c["genome"])
        cx, cy = find_centroid(state)
        crop = crop_periodic(state, cx, cy)
        crops.append(crop)

    print("Rendering Figure 8...")
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.2), facecolor="black")
    fig.patch.set_facecolor("black")
    cmap = plt.cm.inferno

    for idx, (crop, creature) in enumerate(zip(crops, CREATURES)):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        ax.set_facecolor("black")
        ax.imshow(crop, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")

        # Title: kernel/growth
        ax.set_title(creature["label"], color="white", fontsize=11,
                      fontweight="bold", pad=6)

        # Bottom annotation
        ax.text(0.5, -0.06, creature["sublabel"], transform=ax.transAxes,
                ha="center", va="top", color="#ffcc00", fontsize=8.5,
                fontweight="bold", linespacing=1.3)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
            spine.set_linewidth(0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(hspace=0.35, wspace=0.15)

    # Save for paper
    out_path = "paper/figures/fig8_creature_gallery.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

    # Also save PDF version
    out_pdf = "paper/figures/fig8_creature_gallery.pdf"
    fig2, axes2 = plt.subplots(2, 3, figsize=(10, 7.2), facecolor="black")
    fig2.patch.set_facecolor("black")
    for idx, (crop, creature) in enumerate(zip(crops, CREATURES)):
        row, col = idx // 3, idx % 3
        ax = axes2[row, col]
        ax.set_facecolor("black")
        ax.imshow(crop, cmap=cmap, vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(creature["label"], color="white", fontsize=11,
                      fontweight="bold", pad=6)
        ax.text(0.5, -0.06, creature["sublabel"], transform=ax.transAxes,
                ha="center", va="top", color="#ffcc00", fontsize=8.5,
                fontweight="bold", linespacing=1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
            spine.set_linewidth(0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(hspace=0.35, wspace=0.15)
    fig2.savefig(out_pdf, dpi=300, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"Saved: {out_pdf} ({os.path.getsize(out_pdf)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
