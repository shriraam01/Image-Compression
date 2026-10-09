# K-Means Image Compression

Colour-quantisation image compression using K-Means implemented from scratch with NumPy.
Every pixel is replaced by its nearest colour from a K-colour palette; only the palette
and the bit-packed pixel indices are stored.

## How it works
1. Resize the image and treat each pixel as a point in RGB space.
2. Run K-Means (k-means++ init, vectorised distance step, stops when centroids converge).
3. Store the K x 3 palette plus one `ceil(log2 K)`-bit index per pixel (4 bits for K=16).
4. Decompress by looking up each index in the palette.

## Usage
```bash
pip install -r requirements.txt
python compress.py --input assets/sample_image.jpg --k 16
python decompress.py
python evaluate.py        # K sweep + comparison with scikit-learn
```
Add `--no-show` to save plots without opening a window.

## Results
Add your own numbers here after running on your image: a table of K, packed size (KB)
and PSNR (dB), plus `output/comparison.png` and `output/k_sweep.png`.

## Notes
- Raw 256x256 RGB is 192 KB. At K=16 the packed file is about 32 KB, a 6x reduction.
- Banding in smooth regions (like the sky) is the expected cost of a small palette.
- Compared against a JPEG/PNG the picture differs: those formats use their own entropy coding,
  so size comparisons are made against raw RGB.

## Possible extensions
Dithering, entropy coding of the indices, other colour spaces (Lab), larger images with mini-batch K-Means.
