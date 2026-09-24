"""Generates livrable1_photo_classifier.ipynb (edit the cells here, then rerun)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Livrable 1: Binary classification, photo vs. other
**Projet Leyenda — TouNum**

TouNum digitises large volumes of documents. Before captioning (Livrable 3), the pipeline has to keep **only the photos** and set aside paintings, schematics, sketches and scanned text.

This notebook trains a **convolutional neural network (CNN)** that answers one question for each image: *is it a photo?*

| Step | Section |
|---|---|
| 1 | Load and explore the data |
| 2 | Split into train / validation / test |
| 3 | Input pipeline (`tf.data`) |
| 4 | Network architecture, loss and optimiser |
| 5 | Training |
| 6 | Learning curves and bias/variance |
| 7 | Evaluation on the test set |
| 8 | Ways to improve |
""")

code(r"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = Path(r"C:\my\CESI\A5\Data Science\Data")
SOURCES = ["Photo", "Painting", "Schematics", "Sketch", "Text"]
IMG_SIZE = 128      # every image is resized to 128x128 pixels
BATCH = 64
SEED = 42

tf.keras.utils.set_random_seed(SEED)
print("TensorFlow", tf.__version__)
""")

md(r"""
## 1. Load and explore the data
The images are not labelled one by one: **the folder is the label**. We build a table (pandas DataFrame) with one row per image: its path, its source folder, and the binary target `is_photo` (1 = photo, 0 = anything else).
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
The classes are **imbalanced**: about 1 photo for 3 other images. A model that always answers "other" would already reach ~76% accuracy, so accuracy alone is not enough. We also look at **precision / recall on the photo class**, and we give photos more weight during training (*class weights*).
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
- **Validation (15%)**: checked after each epoch to detect over-fitting and to decide when to stop.
- **Test (15%)**: used only once, at the end, for an honest final score.

The split is **stratified by source**, so each set keeps the same mix of photos, paintings, sketches, etc.
""")

code(r"""
train_df, rest = train_test_split(df, test_size=0.30, stratify=df.source, random_state=SEED)
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest.source, random_state=SEED)
pd.DataFrame({name: d.source.value_counts() for name, d in
              [("train", train_df), ("val", val_df), ("test", test_df)]})
""")

md(r"""
## 3. Input pipeline with `tf.data`
40,000 images do not fit comfortably in memory at full size, so `tf.data` reads them in parallel and:
1. decodes JPG/PNG and forces **3 channels** (some images are greyscale, some have transparency),
2. resizes them to 128×128,
3. keeps them in memory after the first pass (`cache`), which makes later epochs much faster,
4. shuffles, batches and prefetches.

Corrupted files are skipped (`ignore_errors`). The source name travels with each image so we can analyse errors per source later.
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
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

xy = lambda img, label, src, path: (img, label)          # what Keras trains on
train_ds = make_ds(train_df, True).map(xy)
val_ds = make_ds(val_df, False).map(xy)
test_ds = make_ds(test_df, False)                   # keeps source and path for the analysis
""")

md(r"""
## 4. Network architecture
```
Input 128×128×3
 → Rescaling (pixels 0–255 → 0–1)
 → Data augmentation (random flip, small rotation, zoom)  — training only
 → [Conv 3×3, 32]  → BatchNorm → MaxPool 2×2      128 → 64
 → [Conv 3×3, 64]  → BatchNorm → MaxPool 2×2       64 → 32
 → [Conv 3×3, 128] → BatchNorm → MaxPool 2×2       32 → 16
 → [Conv 3×3, 256] → BatchNorm → MaxPool 2×2       16 → 8
 → GlobalAveragePooling  (8×8×256 → 256 numbers)
 → Dropout 0.4
 → Dense 1, sigmoid  → probability "this is a photo"
```
- **Convolutions** learn local patterns: edges first, then textures (brush strokes, paper grain, noise), then shapes.
- **MaxPooling** halves the resolution, so later layers see a wider area of the image.
- **BatchNormalization** stabilises and speeds up training.
- **GlobalAveragePooling** replaces a big `Flatten + Dense`, which saves parameters and reduces over-fitting.
- **Dropout** randomly switches off 40% of the neurons during training (regularisation).

**Loss:** `binary_crossentropy`, the standard loss for a yes/no output from a sigmoid.
**Optimiser:** `Adam` with learning rate 0.001. It adapts the step size for each weight.
""")

code(r"""
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
], name="augmentation")

def conv_block(filters):
    return [tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D()]

model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.Rescaling(1 / 255),
    augment,
    *conv_block(32), *conv_block(64), *conv_block(128), *conv_block(256),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid"),
], name="photo_classifier")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")],
)
model.summary()
""")

md(r"""
## 5. Training
- **Class weights**: a mistake on a photo costs about 3× more than a mistake on another image, which compensates for the imbalance.
- **EarlyStopping**: stops when the validation loss has not improved for 4 epochs and restores the best weights.
- **ReduceLROnPlateau**: divides the learning rate by 2 when progress stalls.
- **ModelCheckpoint**: saves the best model to disk.

The first epoch is slower because it reads and decodes every image. After that they come from the cache.
""")

code(r"""
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df.is_photo)
class_weight = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weight)

Path("models").mkdir(exist_ok=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    tf.keras.callbacks.ModelCheckpoint("models/photo_classifier.keras", monitor="val_loss", save_best_only=True),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=25,
                    class_weight=class_weight, callbacks=callbacks, verbose=2)
""")

md(r"""
## 6. Learning curves: loss and accuracy
The course asks for the evolution of the training error and the validation error, and of the accuracy on both sets.
""")

code(r"""
h = pd.DataFrame(history.history)
h.index += 1
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
h[["loss", "val_loss"]].plot(ax=ax[0], marker="o", title="Loss (binary cross-entropy)")
h[["accuracy", "val_accuracy"]].plot(ax=ax[1], marker="o", title="Accuracy")
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=0.3)
plt.tight_layout()
h.round(4)
""")

md(r"""
### How to read these curves (bias / variance)
- **Both losses high and close together** → *under-fitting* (high bias): the model is too simple or has not trained long enough.
- **Training loss keeps falling while validation loss rises** → *over-fitting* (high variance): the model memorises the training images.
- **Both low and close together** → a good compromise.

*Write your own analysis here, based on the curves above. At which epoch does the validation loss stop improving? How big is the gap between train and validation?*
""")

md(r"""
## 7. Evaluation on the test set
The test images have never been seen during training or used to choose when to stop.
""")

code(r"""
y_true, y_prob, y_src, y_path = [], [], [], []
for img, label, src, path in test_ds:
    y_prob.append(model.predict(img, verbose=0).ravel())
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
### Which sources get confused with photos?
For each source: the share of its images the model calls "photo". Ideally 100% for Photo and 0% for the others. The course expects paintings to be the hardest.
""")

code(r"""
per_source = (pd.DataFrame({"source": y_src, "predicted_photo": y_pred, "correct": y_pred == y_true})
              .groupby("source").agg(images=("correct", "size"),
                                      predicted_as_photo=("predicted_photo", "mean"),
                                      accuracy=("correct", "mean")))
per_source.style.format({"predicted_as_photo": "{:.1%}", "accuracy": "{:.1%}"})
""")

code(r"""
# A few mistakes, to understand what fools the model
wrong = np.where(y_pred != y_true)[0][:12]
fig, axes = plt.subplots(2, 6, figsize=(15, 6))
for ax, i in zip(axes.ravel(), wrong):
    ax.imshow(plt.imread(y_path[i]), cmap="gray")
    ax.set_title(f"{y_src[i]}\np(photo)={y_prob[i]:.2f}", fontsize=9)
    ax.axis("off")
plt.suptitle("Misclassified test images"); plt.tight_layout()
""")

md(r"""
## 8. Ways to improve the bias/variance compromise
| Technique | Effect | In this notebook |
|---|---|---|
| **Data augmentation** | More varied training images, less over-fitting | Yes (flip, rotation, zoom) |
| **Dropout** | Randomly disables neurons, less over-fitting | Yes (0.4) |
| **Early stopping** | Stops before the model starts memorising | Yes (patience 4) |
| **Batch normalisation** | Faster, more stable training | Yes |
| **L2 regularisation** (`kernel_regularizer`) | Penalises large weights | Not yet — try it |
| **Bigger images** (e.g. 224×224) | Keeps fine texture: brush strokes vs. camera noise | Try it (slower on CPU) |
| **Transfer learning** (week 3, e.g. MobileNetV2 / EfficientNet pre-trained on ImageNet) | Reuses features learned on millions of photos; usually the biggest gain on paintings | Next step |
| **Threshold tuning** | Move the 0.5 cut-off to trade precision against recall | Try it on the validation set |
""")

code(r"""
model.save("models/photo_classifier.keras")
print("Saved to models/photo_classifier.keras")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "livrable1_photo_classifier.ipynb")
print("Wrote livrable1_photo_classifier.ipynb with", len(cells), "cells")
