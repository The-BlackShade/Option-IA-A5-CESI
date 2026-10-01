"""Generates livrable1_photo_classifier.ipynb (edit the cells here, then rerun).

Follows the method of the CESI workshops: the same convolutional architecture as
WS "Reseaux de neurones convolutifs", introduced in three stages (baseline,
+dropout, +augmentation) so the bias/variance compromise is shown by evidence.
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
We follow the approach of the workshop *Réseaux de neurones convolutifs*, with the same architecture, applied in three stages so that the effect of each regularisation technique is measured rather than assumed:

| Stage | Model | Purpose |
|---|---|---|
| 1 | Baseline CNN | Establish the reference, and expose over-fitting |
| 2 | + Dropout | Measure the effect of one regularisation technique |
| 3 | + Data augmentation | Measure the effect of the second |

| Section | Content |
|---|---|
| 1 | Data: loading, exploration, class imbalance |
| 2 | Train / validation / test split |
| 3 | Input pipeline (`tf.data`) |
| 4 | Architecture, loss and optimiser |
| 5 | Stage 1 — baseline |
| 6 | Stage 2 — dropout |
| 7 | Stage 3 — data augmentation |
| 8 | Comparison and bias/variance analysis |
| 9 | Final evaluation on the test set |
| 10 | Ways to improve further |
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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path(r"C:\my\CESI\A5\Data Science\Data")
SOURCES = ["Photo", "Painting", "Schematics", "Sketch", "Text"]

IMG_SIZE = 128      # the workshop used 180; 128 keeps training feasible on CPU for 41,000 images
BATCH_SIZE = 64
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)
print("TensorFlow", tf.__version__)
""")

md(r"""
## 1. Data
The images are not labelled one by one: **the folder is the label**. We build a table (pandas DataFrame) with one row per image: its path, its source folder, and the binary target `is_photo` (1 = photo, 0 = anything else).

> *Note on method:* the workshop used `image_dataset_from_directory`, which expects one sub-folder per class under a single root. Here the five datasets live in five separate archives, and we need a **binary** target while keeping the original category of each image for the error analysis. We therefore build the table with pandas and feed it to `tf.data`, which gives the same result with the information we need.
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
The classes are **imbalanced**: about 1 photo for 3 other images. A model that always answered "other" would already score ~76% accuracy, so accuracy alone cannot be trusted. Two consequences for the rest of the notebook:
- we report **precision and recall on the photo class**, not only accuracy;
- we pass **class weights** to `fit`, so that an error on a photo costs about three times more than an error on another image. The flower dataset of the workshop was balanced and did not need this.
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
- **Validation (15%)**: watched after each epoch to detect over-fitting and to compare the three stages.
- **Test (15%)**: untouched until section 9, used once, for an honest final score.

The split is **stratified by source**, so each set keeps the same mix of photos, paintings, sketches, etc.

> *Note on method:* the workshop split in two (train / "test", the latter used as validation during training). Using the validation set both to steer decisions and to report the final score flatters the result, so we hold out a third, independent set.
""")

code(r"""
train_df, rest = train_test_split(df, test_size=0.30, stratify=df.source, random_state=SEED)
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest.source, random_state=SEED)
pd.DataFrame({name: d.source.value_counts() for name, d in
              [("train", train_df), ("val", val_df), ("test", test_df)]})
""")

md(r"""
## 3. Input pipeline with `tf.data`
41,000 images do not fit in memory at full size, so `tf.data` reads them in parallel and:
1. decodes JPG/PNG forcing **3 channels** (some images are greyscale, some have transparency),
2. resizes them to 128×128,
3. keeps them in memory after the first epoch (`cache`), as in the workshop,
4. shuffles, batches and prefetches (`prefetch`), again as in the workshop.

Corrupted files are skipped (`ignore_errors`). The source name and the path travel with each image so that section 9 can analyse the errors per category.
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

xy = lambda img, label, src, path: (img, label)      # what Keras trains on
train_ds = make_ds(train_df, True).map(xy)
val_ds = make_ds(val_df, False).map(xy)
test_ds = make_ds(test_df, False)                    # keeps source and path for the analysis
""")

md(r"""
## 4. Architecture, loss and optimiser

The architecture is the one built in the workshop, with three convolutional blocks:

```
Input 128×128×3
 → Rescaling (pixels 0–255 → 0–1)
 → Conv2D 16 filters 3×3, ReLU, padding "same"  → MaxPooling 2×2    128 → 64
 → Conv2D 32 filters 3×3, ReLU, padding "same"  → MaxPooling 2×2     64 → 32
 → Conv2D 64 filters 3×3, ReLU, padding "same"  → MaxPooling 2×2     32 → 16
 → Flatten                                       (16×16×64 = 16 384)
 → Dense 128, ReLU
 → Dense 1, sigmoid        → probability that the image is a photo
```

- **Convolution** detects local patterns: edges first, then textures (brush strokes, paper grain, sensor noise), then shapes. `padding="same"` keeps the size, so only pooling reduces it.
- **MaxPooling** halves height and width, keeping the strongest activations; later layers therefore see a wider area of the image.
- **Flatten + Dense 128** is the classification part, as in the workshop.

**Output layer — the one difference from the workshop.** The workshop classified 5 flower species and ended with `Dense(5)` plus `SparseCategoricalCrossentropy(from_logits=True)`. Our problem is **binary**, so a single neuron with a **sigmoid** is enough: it outputs one probability, and the matching loss is **`binary_crossentropy`**. Two classes do not need two neurons.

**Optimiser:** `Adam`, learning rate 0.001 (the Keras default), as in the workshop. Adam adapts the step size per weight, which makes it a safe default.

**Metrics:** accuracy, plus precision, recall and AUC on the photo class because of the imbalance.
""")

code(r"""
# The workshop architecture; dropout and augmentation are added in stages.
def build_model(dropout=0.0, augmentation=None, name="cnn"):
    model_layers = [layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))]
    if augmentation is not None:
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
        model_layers.append(layers.Dropout(dropout))   # before Flatten, as in the workshop
    model_layers += [
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),         # binary: one probability
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

model = build_model(name="baseline")
model.summary()
""")

md(r"""
Look at the parameter column: the three convolutional layers hold about 23,000 weights, while the `Dense(128)` after `Flatten` holds over 2 million. The convolutions analyse the image with ~1% of the parameters — the memory and computation argument for CNNs, visible on our own data.
""")

code(r"""
# A mistake on a photo costs ~3x more than a mistake on another image
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df.is_photo)
class_weight = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weight)
""")

md(r"""
## 5. Stage 1 — baseline
No regularisation. The first epoch is slower because every image is read and decoded; afterwards they come from the cache.
""")

code(r"""
history_base = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                         class_weight=class_weight, verbose=2)
""")

code(r"""
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

plot_curves(history_base, "Baseline").round(4)
""")

md(r"""
**Read the curves.** If the training loss keeps falling while the validation loss rises, the model is memorising the training images: that is **over-fitting** (high variance). The gap between the two accuracy curves measures it.

*Write your reading of these curves here: at which epoch does `val_loss` stop improving, and how large is the gap?*
""")

md(r"""
## 6. Stage 2 — dropout
Same architecture, with a `Dropout(0.2)` before the `Flatten`, exactly as in the workshop. During training, 20% of the values are randomly set to zero, so no neuron can become the dedicated detector of one training image. Dropout adds **no parameters**: it reduces the effective capacity, not the real one.
""")

code(r"""
model_dropout = build_model(dropout=0.2, name="with_dropout")
history_dropout = model_dropout.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                                    class_weight=class_weight, verbose=2)
plot_curves(history_dropout, "With dropout").round(4)
""")

md(r"""
## 7. Stage 3 — dropout + data augmentation
We add the augmentation layer of the workshop: horizontal flip, small rotation, small zoom. It is active **during training only**; at prediction time it does nothing.

A caution specific to this project: the transformations must not change the label. A horizontal flip is safe for a photograph **and** for a painting or a sketch. A **vertical** flip would be a bad idea here, because an upside-down scanned text is not what TouNum will receive, and the rotation is kept small for the same reason.
""")

code(r"""
augmentation = Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),      # ~18 degrees, as a fraction of a full turn
    layers.RandomZoom(0.1),
], name="augmentation")

model_full = build_model(dropout=0.2, augmentation=augmentation, name="dropout_augmentation")
history_full = model_full.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                              class_weight=class_weight, verbose=2)
plot_curves(history_full, "Dropout + augmentation").round(4)
""")

md(r"""
## 8. Comparison and bias/variance analysis
""")

code(r"""
runs = {"1. baseline": history_base, "2. + dropout": history_dropout, "3. + augmentation": history_full}
summary = pd.DataFrame({
    name: {"train accuracy": h.history["accuracy"][-1],
           "val accuracy": h.history["val_accuracy"][-1],
           "gap": h.history["accuracy"][-1] - h.history["val_accuracy"][-1],
           "train loss": h.history["loss"][-1],
           "val loss": h.history["val_loss"][-1],
           "best val loss": min(h.history["val_loss"]),
           "best epoch": int(np.argmin(h.history["val_loss"]) + 1)}
    for name, h in runs.items()}).T
summary.round(4)
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for name, h in runs.items():
    ax[0].plot(range(1, len(h.history["val_loss"]) + 1), h.history["val_loss"], marker="o", label=name)
    ax[1].plot(range(1, len(h.history["val_accuracy"]) + 1), h.history["val_accuracy"], marker="o", label=name)
ax[0].set_title("Validation loss"); ax[1].set_title("Validation accuracy")
for a in ax:
    a.set_xlabel("epoch"); a.legend(); a.grid(alpha=0.3)
plt.tight_layout()
""")

md(r"""
### The bias/variance compromise
- **High bias (under-fitting):** both curves poor and close together. The model is too simple, or has not trained long enough.
- **High variance (over-fitting):** training keeps improving while validation stalls or degrades. The model memorises.
- The **gap** column above measures the variance; the **val accuracy** column measures what the client actually gets.

*Write your analysis here, using the table and the comparison plot: which stage gives the best compromise, what did dropout change, what did augmentation change, and is any under-fitting visible (for example a validation score that stops improving while both curves stay mediocre)?*
""")

md(r"""
## 9. Final evaluation on the test set
The test images have never been seen: not during training, and not for choosing between the three models. Choose below the model you justified in section 8.
""")

code(r"""
best_model = model_full          # change if your analysis designates another stage
y_true, y_prob, y_src, y_path = [], [], [], []
for img, label, src, path in test_ds:
    y_prob.append(best_model.predict(img, verbose=0).ravel())
    y_true.append(label.numpy()); y_src.append(src.numpy()); y_path.append(path.numpy())
y_true = np.concatenate(y_true).astype(int)
y_prob = np.concatenate(y_prob)
y_src = np.array(SOURCES)[np.concatenate(y_src)]
y_path = [p.decode() for p in np.concatenate(y_path)]
y_pred = (y_prob >= 0.5).astype(int)

print(classification_report(y_true, y_pred, target_names=["other", "photo"], digits=4))
ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["other", "photo"]).plot(cmap="Blues")
plt.title("Confusion matrix — test set");
""")

md(r"""
### Which sources are confused with photos?
For each original category, the share of its images that the model calls "photo". Ideally 100% for Photo and 0% elsewhere. The course expects paintings to be the hardest, some being very realistic.
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
## 10. Ways to improve the bias/variance compromise
| Technique | Effect | Status here |
|---|---|---|
| **Data augmentation** | More varied training images; reduces variance | Applied, stage 3 |
| **Dropout** | Randomly disables neurons; reduces variance | Applied, stage 2 |
| **Class weights** | Compensates the 1:3 imbalance | Applied |
| **Early stopping** | Stops at the best validation epoch | Not applied: the "best epoch" column shows where it would have stopped |
| **L2 regularisation** (`kernel_regularizer`) | Penalises large weights | To try |
| **Larger images** (180 or 224 px) | Keeps fine texture: brush strokes vs. sensor noise | To try; costly on CPU |
| **Transfer learning** (MobileNetV2 / EfficientNet pre-trained on ImageNet) | Reuses features learned on millions of photographs | Week 3 of the course; likely the biggest gain on paintings |
| **Threshold tuning** | Trades precision against recall by moving the 0.5 cut-off | To try on the validation set |

**For TouNum's pipeline**, the choice of threshold is a business decision: a missed photo never gets captioned, whereas a painting wrongly kept only wastes compute downstream. That argues for favouring **recall** on the photo class.
""")

code(r"""
Path("models").mkdir(exist_ok=True)
best_model.save("models/photo_classifier.keras")
print("Saved to models/photo_classifier.keras")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "livrable1_photo_classifier.ipynb")
print("Wrote livrable1_photo_classifier.ipynb with", len(cells), "cells")
