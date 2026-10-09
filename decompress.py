"""Rebuild an image from the packed palette + indices written by compress.py."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from compress import unpack_labels


def main():
    p = argparse.ArgumentParser(description="Decompress a K-Means compressed image")
    p.add_argument("--input", default="output/compressed.npz")
    p.add_argument("--output", default="output/decompressed.png")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    data = np.load(args.input)
    palette = data["palette"]
    bits = int(data["bits"])
    h, w = (int(v) for v in data["shape"])

    labels = unpack_labels(data["packed"], h * w, bits)
    pixels = palette[labels].reshape(h, w, 3)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(f"Decompressed {w}x{h} image saved to {args.output}")

    # Sanity check: should match compress.py's PNG exactly
    reference = Path(args.input).with_name("compressed.png")
    if reference.exists():
        same = np.array_equal(pixels, np.array(Image.open(reference)))
        print("Matches compressed.png exactly:", same)

    if not args.no_show:
        plt.imshow(pixels)
        plt.axis("off")
        plt.title("Decompressed")
        plt.show()


if __name__ == "__main__":
    main()
