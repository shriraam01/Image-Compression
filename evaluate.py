"""Sweep K and compare against scikit-learn: size vs. quality."""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from compress import kmeans, load_image, pack_labels, psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="assets/sample_image.jpg")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--out-dir", default="output")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    original = np.array(load_image(args.input, args.size))
    h, w, _ = original.shape
    pixels = original.reshape(-1, 3).astype(float)

    rows = []
    print(f"{'K':>4} {'bits':>5} {'size KB':>9} {'PSNR (ours)':>12} {'PSNR (sklearn)':>15} {'time ours/sk (s)':>18}")
    for k in args.ks:
        t0 = time.perf_counter()
        palette, labels, _ = kmeans(pixels, k)
        t_ours = time.perf_counter() - t0
        packed, bits = pack_labels(labels, k)
        size_kb = (palette.nbytes + packed.nbytes) / 1024
        _, ours = psnr(original, palette[labels].reshape(h, w, 3))

        t0 = time.perf_counter()
        sk = KMeans(n_clusters=k, n_init=1, random_state=42).fit(pixels)
        t_sk = time.perf_counter() - t0
        sk_img = np.clip(np.rint(sk.cluster_centers_), 0, 255).astype(np.uint8)[sk.labels_]
        _, theirs = psnr(original, sk_img.reshape(h, w, 3))

        rows.append((k, size_kb, ours, theirs))
        print(f"{k:>4} {bits:>5} {size_kb:>9.2f} {ours:>12.2f} {theirs:>15.2f} {t_ours:>9.2f}/{t_sk:<8.2f}")

    ks, sizes, ours, theirs = zip(*rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, ours, "o-", label="ours")
    ax[0].plot(ks, theirs, "s--", label="sklearn")
    ax[0].set(xlabel="K", ylabel="PSNR (dB)", title="Quality vs K")
    ax[0].legend()
    ax[1].plot(sizes, ours, "o-")
    for k, s, q in zip(ks, sizes, ours):
        ax[1].annotate(f"K={k}", (s, q), textcoords="offset points", xytext=(5, -10))
    ax[1].set(xlabel="Packed size (KB)", ylabel="PSNR (dB)", title="Size vs quality")
    plt.tight_layout()
    fig.savefig(Path(args.out_dir) / "k_sweep.png", dpi=150)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
