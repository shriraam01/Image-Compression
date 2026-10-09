"""K-Means colour quantisation for image compression.

Each pixel is replaced by the index of its nearest palette colour. Only the
palette (K x 3 bytes) and the bit-packed indices are stored.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path, size):
    """Load an image as RGB and resize it to size x size."""
    img = Image.open(path).convert("RGB").resize((size, size))
    return img


def assign(pixels, centroids):
    """Index of the nearest centroid for every pixel (vectorised)."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
    d = (
        (pixels ** 2).sum(axis=1)[:, None]
        - 2 * pixels @ centroids.T
        + (centroids ** 2).sum(axis=1)[None, :]
    )
    return d.argmin(axis=1)


def kmeans_pp_init(pixels, k, rng):
    """k-means++ initialisation: spread the starting centroids out."""
    n = len(pixels)
    centroids = np.empty((k, 3))
    centroids[0] = pixels[rng.integers(n)]
    d2 = ((pixels - centroids[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = d2.sum()
        idx = rng.choice(n, p=d2 / total) if total > 0 else rng.integers(n)
        centroids[i] = pixels[idx]
        d2 = np.minimum(d2, ((pixels - centroids[i]) ** 2).sum(axis=1))
    return centroids


def kmeans(pixels, k, max_iters=50, tol=1e-3, seed=42):
    """K-Means from scratch. Stops early when the centroids stop moving."""
    rng = np.random.default_rng(seed)
    centroids = kmeans_pp_init(pixels, k, rng)

    for it in range(1, max_iters + 1):
        labels = assign(pixels, centroids)
        new = centroids.copy()
        for j in range(k):
            members = labels == j
            if members.any():
                new[j] = pixels[members].mean(axis=0)
            else:  # empty cluster: re-seed on a random pixel
                new[j] = pixels[rng.integers(len(pixels))]
        shift = np.linalg.norm(new - centroids)
        centroids = new
        if shift < tol:
            break

    # Round to real 8-bit colours, then label against what we actually store
    palette = np.clip(np.rint(centroids), 0, 255).astype(np.uint8)
    labels = assign(pixels, palette.astype(float)).astype(np.uint8)
    return palette, labels, it


def pack_labels(labels, k):
    """Pack labels into ceil(log2 k) bits each (4 bits for K=16)."""
    bits = max(1, int(np.ceil(np.log2(k))))
    b = np.unpackbits(labels[:, None], axis=1)[:, 8 - bits:]
    return np.packbits(b.ravel()), bits


def unpack_labels(packed, n, bits):
    """Inverse of pack_labels."""
    b = np.unpackbits(packed)[: n * bits].reshape(n, bits)
    full = np.zeros((n, 8), dtype=np.uint8)
    full[:, 8 - bits:] = b
    return np.packbits(full, axis=1).ravel()


def psnr(original, reconstructed):
    """MSE and PSNR (dB) between two uint8 images."""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    value = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    return mse, value


def main():
    p = argparse.ArgumentParser(description="K-Means image compression")
    p.add_argument("--input", default="assets/sample_image.jpg")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--max-iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--no-show", action="store_true", help="save plot, don't open a window")
    args = p.parse_args()

    if not 2 <= args.k <= 256:
        p.error("--k must be between 2 and 256")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = load_image(args.input, args.size)
    original = np.array(img)
    h, w, _ = original.shape
    pixels = original.reshape(-1, 3).astype(float)

    palette, labels, iters = kmeans(pixels, args.k, args.max_iters, seed=args.seed)
    packed, bits = pack_labels(labels, args.k)

    compressed = palette[labels].reshape(h, w, 3)
    mse, score = psnr(original, compressed)

    np.savez(out / "compressed.npz", palette=palette, packed=packed,
             bits=bits, shape=np.array([h, w]))
    Image.fromarray(compressed).save(out / "compressed.png")

    kb = lambda f: os.path.getsize(f) / 1024
    original_jpg = kb(args.input)
    raw = h * w * 3 / 1024
    npz = kb(out / "compressed.npz")
    png = kb(out / "compressed.png")

    print(f"Converged in {iters} iterations (K={args.k}, {bits} bits/pixel)")
    print(f"MSE: {mse:.2f}   PSNR: {score:.2f} dB")
    print(f"Raw RGB ({w}x{h}):          {raw:8.2f} KB")
    print(f"Original file (JPEG):       {original_jpg:8.2f} KB")
    print(f"Compressed PNG:             {png:8.2f} KB")
    print(f"Packed .npz (palette+idx):  {npz:8.2f} KB  ({raw / npz:.1f}x smaller than raw)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original)
    ax[0].set_title(f"Original\n{w} x {h}, raw {raw:.1f} KB")
    ax[1].imshow(compressed)
    ax[1].set_title(f"K-Means (K={args.k})\nPSNR {score:.2f} dB, packed {npz:.1f} KB")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
