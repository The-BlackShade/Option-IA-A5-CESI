"""Train the Livrable 1 classifier of Jupyter_notebook_p1_complete_1.ipynb on CPU.

Faithful to the team notebook: the same PhotoClassifier, images padded to a
square (resize_with_pad), pixels in [0, 1], BCEWithLogitsLoss with pos_weight,
Adam + ReduceLROnPlateau, early stopping on the test loss, and the same two
steps - photo vs painting first, then every category starting from those
weights. Two concessions to a CPU: 128 px instead of 256 (the architecture
ends with AdaptiveAvgPool, so the saved state_dict is identical and
interchangeable with the 256 px weights), and the images are decoded once into
RAM instead of re-read every epoch.

Writes models/photo_classifier_best.pth (+ a JSON history and a curves PNG).
SMOKE=1 trains on a few hundred images for one epoch per step, to check the code.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

SMOKE = os.environ.get("SMOKE") == "1"
SIZE = 128
BATCH_SIZE = 32
MAX_EPOCHS = 1 if SMOKE else 8          # the notebook allows 50; on CPU each epoch is 20-45 min
PATIENCE = 1 if SMOKE else 2
LR_STEP1, LR_STEP2 = 0.001, 0.0001          # as in the notebook
SEED = 42
OUT = Path("models"); OUT.mkdir(exist_ok=True)
LOG = open(OUT / "train_team_classifier.log", "a", encoding="utf-8")

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(os.cpu_count() or 4)
device = torch.device("cpu")

# ---------------------------------------------------------------- data ------
ROOT = Path(r"C:\my\CESI\A5\Data Science\Data")
SOURCES = ["Photo", "Painting", "Schematics", "Sketch", "Text"]
STEP1, STEP2 = ["Photo", "Painting"], ["Schematics", "Sketch", "Text"]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}

def find_data_dir():
    for folder in [ROOT, *sorted(p for p in ROOT.iterdir() if p.is_dir())]:
        if (folder / "Dataset Livrable 1 - Photo" / "Photo").is_dir():
            return folder
    raise FileNotFoundError(f"no Livrable 1 data under {ROOT}")

DATA_DIR = find_data_dir()
log("data:", DATA_DIR, "| size", SIZE, "| smoke" if SMOKE else "| full run")

def list_images(source):
    folder = DATA_DIR / f"Dataset Livrable 1 - {source}" / source
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
    return files[:120] if SMOKE else files

def load_padded(path):
    # Livrable 1 pre-processing: RGB, pad to a square keeping proportions (black borders)
    try:
        with Image.open(path) as im:
            im = ImageOps.pad(im.convert("RGB"), (SIZE, SIZE), color=(0, 0, 0))
            return np.asarray(im, dtype=np.uint8)
    except Exception as exc:                     # the one corrupt painting
        log("skipped", path.name, exc)
        return None

def build_split(sources):
    paths, labels = [], []
    for s in sources:
        files = list_images(s)
        paths += files; labels += [1 if s == "Photo" else 0] * len(files)
    tr_p, te_p, tr_y, te_y = train_test_split(paths, labels, test_size=0.2,
                                              random_state=SEED, stratify=labels)
    return tr_p, te_p, tr_y, te_y

def decode(paths, labels):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        arrays = list(pool.map(load_padded, paths))
    keep = [i for i, a in enumerate(arrays) if a is not None]
    x = torch.from_numpy(np.stack([arrays[i] for i in keep])).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
    y = torch.tensor([labels[i] for i in keep], dtype=torch.float32)
    log(f"  decoded {len(keep)} images in {time.time() - t0:.0f}s ({x.numel() / 1e9:.2f} GB)")
    return x, y

# ---------------------------------------------------------------- model -----
class PhotoClassifier(nn.Module):              # verbatim from the notebook
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.classifier(self.features(x))

def batches(x, y, shuffle):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), BATCH_SIZE):
        j = idx[i:i + BATCH_SIZE]
        yield x[j].float().div_(255.0), y[j].unsqueeze(1)

def run_step(name, model, x_tr, y_tr, x_te, y_te, lr):
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    log(f"[{name}] train {len(x_tr)} (photos {n_pos}, others {n_neg}, pos_weight {n_neg / n_pos:.3f}) | test {len(x_te)} | lr {lr}")

    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_loss, best_state, best_metrics, bad = float("inf"), None, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time(); model.train()
        tot_loss, correct, n = 0.0, 0, 0
        for xb, yb in batches(x_tr, y_tr, True):
            optimizer.zero_grad(set_to_none=True)
            out = model(xb); loss = criterion(out, yb)
            loss.backward(); optimizer.step()
            tot_loss += loss.item() * len(xb); correct += ((out >= 0) == yb.bool()).sum().item(); n += len(xb)
        tr_loss, tr_acc = tot_loss / n, correct / n

        model.eval(); tot_loss, n = 0.0, 0; y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in batches(x_te, y_te, False):
                out = model(xb); tot_loss += criterion(out, yb).item() * len(xb); n += len(xb)
                y_true += yb.ravel().tolist(); y_pred += (torch.sigmoid(out) >= 0.5).float().ravel().tolist()
        te_loss = tot_loss / n
        metrics = {"accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0),
                   "recall": recall_score(y_true, y_pred, zero_division=0), "f1": f1_score(y_true, y_pred, zero_division=0)}
        scheduler.step(te_loss)
        for k, v in zip(hist, (tr_loss, te_loss, tr_acc, metrics["accuracy"])):
            hist[k].append(v)
        log(f"[{name}] epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} "
            f"acc {metrics['accuracy']:.4f} P {metrics['precision']:.4f} R {metrics['recall']:.4f} F1 {metrics['f1']:.4f} | {time.time() - t0:.0f}s")
        if te_loss < best_loss:
            best_loss, bad = te_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = dict(metrics, epoch=epoch, test_loss=te_loss)
            torch.save(best_state, OUT / "photo_classifier_best.pth")
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"[{name}] early stopping after {epoch} epochs (best test loss {best_loss:.4f})"); break
    model.load_state_dict(best_state)
    return hist, best_metrics

# ---------------------------------------------------------------- run -------
t_all = time.time()
log("step 1: photo vs painting")
tr_p, te_p, tr_y, te_y = build_split(STEP1)
x_tr, y_tr = decode(tr_p, tr_y); x_te, y_te = decode(te_p, te_y)
model = PhotoClassifier().to(device)
hist1, best1 = run_step("step1", model, x_tr, y_tr, x_te, y_te, LR_STEP1)
torch.save(model.state_dict(), OUT / "photo_classifier_step1.pth")
del x_tr, x_te

log("step 2: every category, starting from the step 1 weights")
tr_p2, te_p2, tr_y2, te_y2 = build_split(STEP2)
x_tr, y_tr = decode(tr_p + tr_p2, tr_y + tr_y2); x_te, y_te = decode(te_p + te_p2, te_y + te_y2)
hist2, best2 = run_step("step2", model, x_tr, y_tr, x_te, y_te, LR_STEP2)
torch.save(model.state_dict(), OUT / "photo_classifier.pth")

json.dump({"size": SIZE, "step1": {"history": hist1, "best": best1}, "step2": {"history": hist2, "best": best2},
           "minutes": round((time.time() - t_all) / 60, 1)}, open(OUT / "team_classifier_history.json", "w"), indent=1)
log("done in", round((time.time() - t_all) / 60, 1), "minutes | best step 2:", best2)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for h, lab in ((hist1, "step 1"), (hist2, "step 2")):
    e = range(1, len(h["train_loss"]) + 1)
    ax[0].plot(e, h["train_loss"], marker="o", label=f"{lab} train"); ax[0].plot(e, h["test_loss"], marker="o", label=f"{lab} test")
    ax[1].plot(e, h["train_acc"], marker="o", label=f"{lab} train"); ax[1].plot(e, h["test_acc"], marker="o", label=f"{lab} test")
ax[0].set_title("Loss"); ax[1].set_title("Accuracy")
for a in ax: a.set_xlabel("epoch"); a.legend(); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "team_classifier_curves.png", dpi=110)
