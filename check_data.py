"""Scan the Livrable 1 images and list files TensorFlow cannot read."""
import os
from collections import Counter
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

DATA = Path(r"C:\my\CESI\A5\Data Science\Data")
CLASSES = ["Painting", "Photo", "Schematics", "Sketch", "Text"]

bad, exts, shapes = [], Counter(), Counter()
for c in CLASSES:
    for p in (DATA / f"Dataset Livrable 1 - {c}" / c).iterdir():
        exts[p.suffix.lower()] += 1
        try:
            img = tf.io.decode_image(tf.io.read_file(str(p)), expand_animations=False)
            shapes[img.shape[-1]] += 1
        except Exception:
            bad.append(p)

print("Extensions:", dict(exts))
print("Colour channels (1=grey, 3=RGB, 4=RGBA):", dict(shapes))
print("Unreadable files:", len(bad))
Path("bad_files.txt").write_text("\n".join(map(str, bad)), encoding="utf-8")
