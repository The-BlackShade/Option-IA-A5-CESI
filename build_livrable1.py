"""Generates livrable1_photo_classifier.ipynb (edit the cells here, then rerun).

Architecture from the workshop "Reseaux de neurones convolutifs". Adds, with a
justification in the notebook: early stopping, a two-stage curriculum (the hard
pair first, then every source), and a decision threshold tuned on validation.
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Livrable 1: Binary classification, photo vs. other
**Projet Leyenda — TouNum**

TouNum digitises large volumes of documents. Before captioning (Livrable 3), the pipeline must keep **only the photographs** and set aside paintings, schematics, sketches and scanned text.

This notebook trains a **convolutional neural network (CNN)** that answers one question per image: *is it a photo?*

## Method
The architecture is the one built in the workshop *Réseaux de neurones convolutifs*. Three additions are made, each justified by a measurement rather than by habit:

| Addition | Why |
|---|---|
| **Early stopping** | A first study (section 9) showed the best validation loss at epoch 3 of 10; the remaining epochs only over-fitted. We now keep the weights of the best epoch. |
| **Two-stage training** | Nearly every error is photo versus painting. The model is first trained on that pair alone, then on all five sources starting from those weights. |
| **Tuned threshold** | The 0.5 cut-off is arbitrary. For TouNum, a missed photo is worse than a painting wrongly kept, so the threshold is chosen on validation. |

| Section | Content |
|---|---|
| 1 | Data: loading, exploration, class imbalance |
| 2 | Train / validation / test split |
| 3 | Input pipeline (`tf.data`) |
| 4 | Architecture, loss and optimiser |
| 5 | Reference model: all five sources at once |
| 6 | Stage A: the hard pair, photo versus painting |
| 7 | Stage B: continue on all five sources |
| 8 | Comparison, and the decision threshold |
| 9 | Regularisation study and bias/variance analysis |
| 10 | Final evaluation on the test set |
| 11 | Ways to improve further |
""")

code(r"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path(r"C:\my\CESI\A5\Data Science\Data")
SOURCES = ["Photo", "Painting", "Schematics", "Sketch", "Text"]
HARD_PAIR = ["Photo", "Painting"]

IMG_SIZE = 160      # 128 lost too much texture, 256 is out of reach on CPU
BATCH_SIZE = 32
MAX_EPOCHS = 20     # early stopping decides the real number
PATIENCE = 3
SEED = 42

tf.keras.utils.set_random_seed(SEED)
print("TensorFlow", tf.__version__)
""")

md(r"""
## 1. Data
The images are not labelled one by one: **the folder is the label**. We build a table (pandas DataFrame) with one row per image: its path, its source folder, and the binary target `is_photo` (1 = photo, 0 = anything else).

> *Note on method:* the workshop used `image_dataset_from_directory`, which expects one sub-folder per class under a single root. Here the five datasets live in five separate archives, and we need a **binary** target while keeping each image's original category for the error analysis. We therefore build the table with pandas and feed it to `tf.data`.
""")

code(r"""
IMAGE_EXT = {".jpg", ".jpeg", ".png"}   # skips files such as desktop.ini

rows = []
for src in SOURCES:
    folder = DATA_DIR / f"Dataset Livrable 1 - {src}" / src
    for p in folder.iterdir():
        if p.suffix.lower() in IMAGE_EXT and p.stat().st_size > 0:
            rows.append({"path": str(p), "source": src, "is_photo": int(src == "Photo")})

df = pd.DataFrame(rows)
print(len(df), "images")
df.groupby("source").size().to_frame("images")
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
df["source"].value_counts().plot.bar(ax=ax[0], title="Images per source")
df["is_photo"].map({1: "photo", 0: "other"}).value_counts().plot.bar(ax=ax[1], title="Binary target")
plt.tight_layout()
""")

md(r"""
The classes are **imbalanced**: about 1 photo for 3 other images. A model answering "other" every time would already score ~76% accuracy, so accuracy alone cannot be trusted. Hence:
- we report **precision and recall on the photo class**, not only accuracy;
- we pass **class weights** to `fit`, so an error on a photo costs about three times more. The flower dataset of the workshop was balanced and needed none of this.
""")

code(r"""
fig, axes = plt.subplots(len(SOURCES), 6, figsize=(14, 12))
for r, src in enumerate(SOURCES):
    for c, path in enumerate(df[df.source == src].sample(6, random_state=SEED).path):
        axes[r, c].imshow(plt.imread(path), cmap="gray")
        axes[r, c].axis("off")
    axes[r, 0].set_title(src, loc="left")
plt.tight_layout()
""")

md(r"""
## 2. Train / validation / test split
- **Train (70%)**: the network learns from these images.
- **Validation (15%)**: used after each epoch for early stopping, for comparing models, and for choosing the decision threshold.
- **Test (15%)**: untouched until section 10, used once.

The split is **stratified by source**, so every set keeps the same mix of categories.

> *Note on method:* a two-way split (train / "test") is common, but when the same set stops the training, picks the model and reports the score, that score is optimistic. The third set is what makes our final figure honest.
""")

code(r"""
train_df, rest = train_test_split(df, test_size=0.30, stratify=df.source, random_state=SEED)
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest.source, random_state=SEED)

# Subsets for stage A: the pair that causes nearly every error
train_hard = train_df[train_df.source.isin(HARD_PAIR)]
val_hard = val_df[val_df.source.isin(HARD_PAIR)]

print("train", len(train_df), "| val", len(val_df), "| test", len(test_df))
print("hard pair - train", len(train_hard), "| val", len(val_hard))
pd.DataFrame({name: d.source.value_counts() for name, d in
              [("train", train_df), ("val", val_df), ("test", test_df)]})
""")

md(r"""
## 3. Input pipeline with `tf.data`
41,000 images do not fit in memory at full size, so `tf.data` reads them in parallel and:
1. decodes JPG/PNG forcing **3 channels** (some images are greyscale, some have transparency),
2. resizes them to 160×160,
3. caches them after the first epoch, as in the workshop,
4. shuffles, batches and prefetches.

Corrupted files are skipped (`ignore_errors`). The source and the path travel with each image so section 10 can analyse errors per category.
""")

code(r"""
SRC_ID = {s: i for i, s in enumerate(SOURCES)}

def load(path, label, src):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.uint8), label, src, path

def make_ds(d, training):
    ds = tf.data.Dataset.from_tensor_slices(
        (d.path.values, d.is_photo.values.astype("float32"), d.source.map(SRC_ID).values))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).ignore_errors().cache()
    if training:
        ds = ds.shuffle(5000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

xy = lambda img, label, src, path: (img, label)     # what Keras trains on

train_ds = make_ds(train_df, True).map(xy)
val_ds = make_ds(val_df, False).map(xy)
train_hard_ds = make_ds(train_hard, True).map(xy)
val_hard_ds = make_ds(val_hard, False).map(xy)
test_ds = make_ds(test_df, False)                   # keeps source and path
val_eval_ds = make_ds(val_df, False)                # same, for threshold tuning
""")

md(r"""
## 4. Architecture, loss and optimiser

The architecture of the workshop, with three convolutional blocks:

```
Input 160×160×3
 → Data augmentation (flip, small rotation, small zoom)   — training only
 → Rescaling (pixels 0–255 → 0–1)
 → Conv2D 16 filters 3×3, ReLU, padding "same"  → MaxPooling 2×2   160 → 80
 → Conv2D 32 filters 3×3, ReLU, padding "same"  → MaxPooling 2×2    80 → 40
 → Conv2D 64 filters 3×3, ReLU, padding "same"  → MaxPooling 2×2    40 → 20
 → Dropout 0.2
 → Flatten                                        (20×20×64 = 25 600)
 → Dense 128, ReLU
 → Dense 1, sigmoid        → probability that the image is a photo
```

**Output layer — the difference from the workshop.** It classified 5 flower species with `Dense(5)` and `SparseCategoricalCrossentropy(from_logits=True)`. Our task is **binary**, so one neuron with a **sigmoid** suffices, and the matching loss is **`binary_crossentropy`**.

**Optimiser:** `Adam`, learning rate 0.001, as in the workshop.

**Regularisation:** dropout 0.2 before `Flatten` and the augmentation layer, both from section 6 of the workshop. Section 9 reports what each contributed.

**Augmentation caution:** the transformations must not change the label. A horizontal flip is safe for a photo, a painting or a sketch; a *vertical* flip would be wrong, since upside-down scans are not what TouNum receives. The rotation is kept small for the same reason.
""")

code(r"""
augmentation = Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),      # ~18 degrees, as a fraction of a full turn
    layers.RandomZoom(0.1),
], name="augmentation")

# The workshop architecture, with the regularisation of its section 6.
def build_model(dropout=0.2, augment=True, name="cnn"):
    model_layers = [layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))]
    if augment:
        model_layers.append(augmentation)
    model_layers += [
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
    ]
    if dropout:
        model_layers.append(layers.Dropout(dropout))
    model_layers += [
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
    model = Sequential(model_layers, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )
    return model

build_model(name="preview").summary()
""")

md(r"""
### Early stopping
Training stops when the validation loss has not improved for 3 epochs, and the **weights of the best epoch are restored**. Without it the model keeps training past its best point and we would report over-fitted weights.
""")

code(r"""
def early_stop():
    return keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                         restore_best_weights=True, verbose=1)

def weights_for(d):
    # An error on a photo costs about 3x more than an error on another image
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=d.is_photo)
    return {0: w[0], 1: w[1]}

def plot_curves(history, title):
    h = pd.DataFrame(history.history)
    h.index += 1
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    h[["loss", "val_loss"]].plot(ax=ax[0], marker="o", title=f"{title} — loss")
    h[["accuracy", "val_accuracy"]].plot(ax=ax[1], marker="o", title=f"{title} — accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    plt.tight_layout()
    return h

print("class weights, all sources:", weights_for(train_df))
print("class weights, hard pair  :", weights_for(train_hard))
""")

md(r"""
## 5. Reference model: all five sources at once
The straightforward approach, and the control against which the two-stage training is measured. The first epoch is slower because every image is read and decoded; afterwards they come from the cache.
""")

code(r"""
model_ref = build_model(name="reference")
history_ref = model_ref.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
                            class_weight=weights_for(train_df),
                            callbacks=[early_stop()], verbose=2)
plot_curves(history_ref, "Reference").round(4)
""")

md(r"""
## 6. Stage A — the hard pair
Photos and paintings share composition, colour and subject; schematics and text do not look like photographs at all. Training first on the pair alone forces the network to spend its capacity on the distinction that decides the score, instead of on easy wins it would get anyway.

The sub-problem is also **balanced** (9,993 photos against 10,000 paintings), so the model learns the difference itself rather than the base rate.
""")

code(r"""
model_two = build_model(name="two_stage")
history_a = model_two.fit(train_hard_ds, validation_data=val_hard_ds, epochs=MAX_EPOCHS,
                          class_weight=weights_for(train_hard),
                          callbacks=[early_stop()], verbose=2)
plot_curves(history_a, "Stage A — photo vs painting").round(4)
""")

md(r"""
## 7. Stage B — continue on all five sources
The same model, with the weights learned in stage A, now trained on everything. Nothing is reset: stage A is the starting point, which is why this is a form of transfer learning inside our own data.
""")

code(r"""
history_b = model_two.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
                          class_weight=weights_for(train_df),
                          callbacks=[early_stop()], verbose=2)
plot_curves(history_b, "Stage B — all sources").round(4)
""")

md(r"""
## 8. Comparison, and the decision threshold
""")

code(r"""
runs = {"reference (1 stage)": history_ref, "stage A (hard pair)": history_a, "stage B (all sources)": history_b}
summary = pd.DataFrame({
    name: {"epochs run": len(h.history["loss"]),
           "train accuracy": h.history["accuracy"][-1],
           "val accuracy": h.history["val_accuracy"][-1],
           "gap": h.history["accuracy"][-1] - h.history["val_accuracy"][-1],
           "best val loss": min(h.history["val_loss"]),
           "best epoch": int(np.argmin(h.history["val_loss"]) + 1)}
    for name, h in runs.items()}).T
summary.round(4)
""")

md(r"""
### Choosing the threshold on validation
The model outputs a probability; turning it into a decision needs a cut-off. The default 0.5 maximises nothing in particular. We sweep the threshold on the **validation** set and keep the value with the best F1 on the photo class — the test set stays untouched.
""")

code(r"""
def probabilities(model, ds):
    probs, labels, srcs, paths = [], [], [], []
    for img, label, src, path in ds:
        probs.append(model.predict(img, verbose=0).ravel())
        labels.append(label.numpy()); srcs.append(src.numpy()); paths.append(path.numpy())
    return (np.concatenate(probs), np.concatenate(labels).astype(int),
            np.array(SOURCES)[np.concatenate(srcs)], [p.decode() for p in np.concatenate(paths)])

best_model = model_two          # change here if the reference model wins in the table above
val_prob, val_true, _, _ = probabilities(best_model, val_eval_ds)

grid = np.arange(0.05, 0.96, 0.05)
scores = [precision_recall_fscore_support(val_true, (val_prob >= t).astype(int),
                                          average="binary", zero_division=0)[:3] for t in grid]
sweep = pd.DataFrame(scores, columns=["precision", "recall", "f1"], index=grid.round(2))
THRESHOLD = float(sweep.f1.idxmax())
print("best threshold on validation:", THRESHOLD)

sweep.plot(marker="o", figsize=(9, 4), title="Validation metrics vs decision threshold")
plt.axvline(THRESHOLD, color="grey", linestyle="--"); plt.xlabel("threshold"); plt.grid(alpha=0.3)
sweep.round(4)
""")

md(r"""
## 9. Regularisation study and bias/variance analysis

A first experiment, run at 128×128 for 10 fixed epochs without early stopping, compared the same architecture with and without the regularisation of the workshop:

| Stage | Train accuracy | Val accuracy | Gap | Best val loss | Best epoch |
|---|---|---|---|---|---|
| Baseline, no regularisation | 0.9825 | 0.9148 | 0.068 | 0.204 | 3 |
| + Dropout 0.2 | 0.9632 | 0.9143 | 0.049 | 0.262 | 8 |
| + Dropout and augmentation | 0.9077 | 0.8704 | 0.037 | 0.303 | 7 |

Three facts come out of it, and they drive the choices made above:
1. **Over-fitting is real but mild.** The baseline gap is 6.8 points, far from the flower workshop's 40 points: 41,000 images against 2,936 change the picture entirely.
2. **Regularisation reduced the gap but not the error.** Validation accuracy stayed flat with dropout and *fell* with augmentation. Reducing variance only helps when variance is what limits you; here capacity was already the binding constraint.
3. **The best epoch was the 3rd of 10.** Everything after it was over-fitting, which is why early stopping now ends the training.

*Write your own reading of the curves of sections 5 to 7 here: where does each model stop, how big is the gap, and does the two-stage model start better than the reference?*
""")

md(r"""
## 10. Final evaluation on the test set
These images were never seen: not in training, not for early stopping, not for the threshold.
""")

code(r"""
y_prob, y_true, y_src, y_path = probabilities(best_model, test_ds)
y_pred = (y_prob >= THRESHOLD).astype(int)

print(f"threshold {THRESHOLD:.2f}")
print(classification_report(y_true, y_pred, target_names=["other", "photo"], digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["other", "photo"]).plot(cmap="Blues")
plt.title("Confusion matrix — test set");
""")

code(r"""
# What the default cut-off would have given, for comparison
print(classification_report(y_true, (y_prob >= 0.5).astype(int),
                            target_names=["other", "photo"], digits=4))
""")

md(r"""
### Which sources are confused with photos?
For each original category, the share of its images the model calls "photo". Ideally 100% for Photo, 0% elsewhere. Paintings are expected to be the hardest, some being very realistic.
""")

code(r"""
per_source = (pd.DataFrame({"source": y_src, "predicted_photo": y_pred, "correct": y_pred == y_true})
              .groupby("source").agg(images=("correct", "size"),
                                      predicted_as_photo=("predicted_photo", "mean"),
                                      accuracy=("correct", "mean")))
per_source.style.format({"predicted_as_photo": "{:.1%}", "accuracy": "{:.1%}"})
""")

code(r"""
# A few mistakes, to see what fools the model
wrong = np.where(y_pred != y_true)[0][:12]
fig, axes = plt.subplots(2, 6, figsize=(15, 6))
for ax, i in zip(axes.ravel(), wrong):
    ax.imshow(plt.imread(y_path[i]), cmap="gray")
    ax.set_title(f"{y_src[i]}\np(photo)={y_prob[i]:.2f}", fontsize=9)
    ax.axis("off")
plt.suptitle("Misclassified test images"); plt.tight_layout()
""")

md(r"""
## 11. Ways to improve further
| Technique | Effect | Status |
|---|---|---|
| **Dropout and augmentation** | Reduce variance | Applied; measured in section 9 |
| **Class weights** | Compensate the 1:3 imbalance | Applied |
| **Early stopping** | Keep the best epoch | Applied, patience 3 |
| **Two-stage training** | Spend capacity on the hard pair | Applied, sections 6–7 |
| **Threshold tuning** | Trade precision against recall | Applied, section 8 |
| **Larger images** (256 px) | Keeps the texture that separates paint from sensor noise | Not done: ~6 h per model on CPU. The obvious next step on a GPU |
| **L2 regularisation** | Penalises large weights | To try |
| **Transfer learning** (MobileNetV2, EfficientNet) | Reuses features learned on millions of photographs | Phase 6 of the course; the largest expected gain |

**For TouNum's pipeline**, the threshold is a business decision: a missed photo is never captioned, while a painting wrongly kept only wastes computation downstream. That argues for favouring **recall** on the photo class, and section 8 shows the exact cost in precision.
""")

code(r"""
Path("models").mkdir(exist_ok=True)
best_model.save("models/photo_classifier.keras")
with open("models/threshold.txt", "w") as f:
    f.write(str(THRESHOLD))
print("Saved model and threshold", THRESHOLD)
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "livrable1_photo_classifier.ipynb")
print("Wrote livrable1_photo_classifier.ipynb with", len(cells), "cells")
