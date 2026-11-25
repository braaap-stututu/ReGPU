"""Compare two folders of resolve dumps (regpuctl capture's out/res): same file names, mean absolute difference per
image (0-255) and the mean brightness of each side, sorted by guest address (resolve order is roughly address order
within a frame, not exactly).
usage: resdiff.py <a_dir> <b_dir> [min_diff]"""
import os
import sys

from PIL import Image
import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
min_diff = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
common = sorted(set(os.listdir(a_dir)) & set(os.listdir(b_dir)))
only_a = sorted(set(os.listdir(a_dir)) - set(os.listdir(b_dir)))
only_b = sorted(set(os.listdir(b_dir)) - set(os.listdir(a_dir)))
rows = []
for name in common:
    try:
        a = np.asarray(Image.open(os.path.join(a_dir, name)).convert('RGBA'), dtype=np.float32)
        b = np.asarray(Image.open(os.path.join(b_dir, name)).convert('RGBA'), dtype=np.float32)
    except Exception as e:
        print('skip', name, e)
        continue
    if a.shape != b.shape:
        rows.append((name, -1, 0, 0, f'shape {a.shape} vs {b.shape}'))
        continue
    d = np.abs(a[..., :3] - b[..., :3]).mean()
    rows.append((name, d, a[..., :3].mean(), b[..., :3].mean(), ''))
for name, d, ma, mb, note in rows:
    if d < 0 or d >= min_diff:
        print(f'{name:34s} diff {d:6.2f}  mean {ma:6.1f} vs {mb:6.1f} {note}')
print(f'{len(common)} common, {len(only_a)} only in a, {len(only_b)} only in b')
if only_a:
    print('only a:', ' '.join(only_a[:40]))
if only_b:
    print('only b:', ' '.join(only_b[:40]))
