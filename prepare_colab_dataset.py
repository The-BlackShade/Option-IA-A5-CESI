"""Shrink the Livrable 1 images and zip them for upload to Google Drive.

Originals are ~7.5 GB. Training resizes to 128 px, so 256 px JPEG copies are
enough and upload in minutes. Produces colab_upload/livrable1_small.zip with
one folder per class.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import zipfile

from PIL import Image

DATA_DIR = Path(r"C:\my\CESI\A5\Data Science\Data")
SOURCES = ["Photo", "Painting", "Schematics", "Sketch", "Text"]
OUT = Path("colab_upload")
STAGE = OUT / "livrable1_small"
MAX_SIDE = 256
QUALITY = 85

def shrink(args):
    src_path, dst_path = args
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            im.save(dst_path, "JPEG", quality=QUALITY, optimize=True)
        return True
    except Exception as exc:                      # corrupted files are skipped
        print("skipped", src_path.name, exc)
        return False

jobs = []
for src in SOURCES:
    (STAGE / src).mkdir(parents=True, exist_ok=True)
    for p in (DATA_DIR / f"Dataset Livrable 1 - {src}" / src).iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stat().st_size > 0:
            jobs.append((p, STAGE / src / (p.stem + ".jpg")))

print(f"{len(jobs)} images to convert")
with ThreadPoolExecutor(max_workers=12) as pool:
    ok = sum(pool.map(shrink, jobs))
print(f"{ok} converted")

zip_path = OUT / "livrable1_small.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:   # JPEGs: no gain from deflate
    for p in STAGE.rglob("*.jpg"):
        z.write(p, p.relative_to(STAGE))
shutil.rmtree(STAGE)
print(f"{zip_path} -> {zip_path.stat().st_size / 1024**2:.0f} MB")
