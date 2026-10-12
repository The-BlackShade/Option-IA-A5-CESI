"""Generates livrable2_denoising.ipynb (edit this builder, then rerun).

Continues Livrable 1: the saved photo classifier filters TouNum's noisy batch,
then a convolutional autoencoder (workshop "Autoencodeur et traitement
d'image", adapted to RGB) cleans what is kept. Section 8 adds the
autoencoder-as-classifier comparison moved here from phase 3.
"""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Livrable 2: Image denoising
**Projet Leyenda — TouNum**

TouNum scans at speed, and some scans come out noisy. Before captioning (Livrable 3), those images have to be cleaned.

This notebook continues **Livrable 1**: its saved classifier decides which images are photographs, and a **convolutional autoencoder** then cleans them.

## The problem to solve first
The delivered dataset contains **148 noisy images and no clean versions**. A denoiser is trained on pairs (noisy → clean), so there is nothing to learn from as it stands.

Our answer, in three steps:
1. **Measure** the noise actually present in TouNum's images.
2. **Recreate** it on the clean photographs of Livrable 1, which gives as many exact pairs as we want.
3. **Apply** the trained model to the real noisy images, and check the result both visually and by measurement.

| Section | Content |
|---|---|
| 1 | The chain: Livrable 1 filters the noisy batch |
| 2 | Measuring the real noise |
| 3 | Building training pairs |
| 4 | Architecture of the convolutional autoencoder |
| 5 | Training |
| 6 | Evaluation on known pairs (PSNR) |
| 7 | Application to TouNum's real images |
| 8 | The autoencoder as a feature extractor (comparison with Livrable 1) |
| 9 | Analysis and improvements |
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

DATA_DIR = Path(r"C:\my\CESI\A5\Data Science\Data")
NOISY_DIR = DATA_DIR / "Dataset Livrable 2" / "Dataset"
CLEAN_DIR = DATA_DIR / "Dataset Livrable 1 - Photo" / "Photo"

IMG_SIZE = 128        # working size of the denoiser
CLS_SIZE = 160        # input size of the Livrable 1 classifier
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 3
N_CLEAN = 6000        # clean photographs used to build the pairs
SEED = 42

tf.keras.utils.set_random_seed(SEED)
print("TensorFlow", tf.__version__)
print(len(list(NOISY_DIR.glob("*.jpg"))), "noisy images |", len(list(CLEAN_DIR.glob("*.jpg"))), "clean photos")
""")

md(r"""
## 1. The chain: Livrable 1 filters the batch
The deliverables form a pipeline. Before cleaning anything we run **the classifier trained in Livrable 1**, with the threshold chosen there on validation (0.65), over TouNum's 148 noisy images.

Two things come out of it: the images to clean, and a first measurement — *does noise disturb the classifier?* It was trained on clean images, so this is a genuine test of its robustness.
""")

code(r"""
classifier = keras.models.load_model("models/photo_classifier.keras")
THRESHOLD = float(Path("models/threshold.txt").read_text())
print("classifier input", classifier.input_shape, "| threshold", THRESHOLD)

noisy_paths = sorted(NOISY_DIR.glob("*.jpg"))

def read_image(path, size):
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    return tf.image.resize(img, (size, size)) / 255.0

batch_cls = tf.stack([read_image(p, CLS_SIZE) * 255.0 for p in noisy_paths])   # classifier rescales itself
p_photo = classifier.predict(batch_cls, verbose=0).ravel()

kept = p_photo >= THRESHOLD
print(f"kept as photographs: {kept.sum()} / {len(noisy_paths)}")
plt.figure(figsize=(9, 3))
plt.hist(p_photo, bins=20)
plt.axvline(THRESHOLD, color="grey", linestyle="--")
plt.title("Livrable 1 classifier on TouNum's noisy images"); plt.xlabel("p(photo)"); plt.grid(alpha=0.3);
""")

md(r"""
*Comment on the figure: how many images pass, and are the rejected ones genuinely not photographs, or photographs that the noise disturbed? The classifier never saw noisy images during training, so a drop here is expected and is itself an argument for denoising **before** the rest of the pipeline.*
""")

md(r"""
## 2. Measuring the real noise
We cannot copy noise we have not looked at. For each image we estimate the **standard deviation of the noise** by comparing it with a median-filtered version of itself: the filter removes the grain, and what is left is mostly noise.

The robust estimator is `sigma ≈ 1.4826 × median(|image − median_filter(image)|)` — the 1.4826 converts a median absolute deviation into a standard deviation for Gaussian noise.
""")

code(r"""
from scipy.ndimage import median_filter

# img: float array in [0, 1]; returns the estimated noise sigma, in the same scale
def estimate_sigma(img):
    smooth = median_filter(img, size=(3, 3, 1))
    return 1.4826 * np.median(np.abs(img - smooth))

sample = [read_image(p, IMG_SIZE).numpy() for p in noisy_paths[:40]]
sigmas = np.array([estimate_sigma(im) for im in sample])
print(f"estimated sigma on TouNum's images: mean {sigmas.mean():.3f}, "
      f"min {sigmas.min():.3f}, max {sigmas.max():.3f}  (pixels in [0, 1])")

clean_paths = sorted(CLEAN_DIR.glob("*.jpg"))
clean_sample = [read_image(p, IMG_SIZE).numpy() for p in clean_paths[:40]]
clean_sigmas = np.array([estimate_sigma(im) for im in clean_sample])
print(f"same estimator on clean photographs: mean {clean_sigmas.mean():.3f}")

plt.figure(figsize=(9, 3))
plt.hist(clean_sigmas, bins=20, alpha=0.6, label="clean photos")
plt.hist(sigmas, bins=20, alpha=0.6, label="TouNum noisy")
plt.legend(); plt.xlabel("estimated sigma"); plt.title("How noisy is the delivered batch?"); plt.grid(alpha=0.3);
""")

md(r"""
The gap between the two histograms is what we have to reproduce. Note the limit of the method: the estimator also reacts to fine **texture**, so a detailed clean photograph gets a non-zero value. We therefore use the difference between the two distributions, not the absolute number.
""")

code(r"""
# Noise levels used for training: a range around what was measured, so the model
# does not learn one single level by heart.
NOISE_MIN = float(max(0.02, clean_sigmas.mean()))
NOISE_MAX = float(max(0.10, sigmas.mean() * 1.5))
print(f"training noise sigma drawn in [{NOISE_MIN:.3f}, {NOISE_MAX:.3f}]")
""")

md(r"""
## 3. Building the training pairs
Each pair is: **a clean photograph of Livrable 1** on one side, **the same photograph with noise we added** on the other. The clean image is the target, so supervision is free and exact.

Two choices worth defending:
- **A random noise level per image**, drawn in the interval above, instead of one fixed level. A model trained on a single level generalises badly to any other.
- **Noise drawn afresh at every epoch**, so the model never sees the same noisy image twice. This acts as data augmentation and is why this architecture over-fits so little.
""")

code(r"""
paths_used = [str(p) for p in clean_paths[:N_CLEAN]]
n_val = int(0.15 * len(paths_used))
train_paths, val_paths = paths_used[n_val:], paths_used[:n_val]
print(len(train_paths), "training photos |", len(val_paths), "validation photos")

def load_clean(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return img

def add_noise(clean):
    sigma = tf.random.uniform([], NOISE_MIN, NOISE_MAX)
    noisy = clean + tf.random.normal(tf.shape(clean), stddev=sigma)
    return tf.clip_by_value(noisy, 0.0, 1.0), clean      # (input, target)

def make_pairs(paths, training):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_clean, num_parallel_calls=tf.data.AUTOTUNE).ignore_errors().cache()
    if training:
        ds = ds.shuffle(1000, seed=SEED)
    return ds.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_pairs(train_paths, True)
val_ds = make_pairs(val_paths, False)
""")

code(r"""
noisy_b, clean_b = next(iter(val_ds))
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for i in range(6):
    axes[0, i].imshow(noisy_b[i]); axes[0, i].axis("off")
    axes[1, i].imshow(clean_b[i]); axes[1, i].axis("off")
axes[0, 0].set_title("noisy (input)", loc="left"); axes[1, 0].set_title("clean (target)", loc="left")
plt.tight_layout()
""")

md(r"""
## 4. Architecture of the convolutional autoencoder

The architecture of the workshop *Autoencodeur et traitement d'image*, adapted to colour images:

```
Input 128×128×3
 encoder : Conv2D 32 → MaxPooling 2×2     128 → 64
           Conv2D 32 → MaxPooling 2×2      64 → 32        bottleneck 32×32×32
 decoder : Conv2D 32 → UpSampling 2×2      32 → 64
           Conv2D 32 → UpSampling 2×2      64 → 128
           Conv2D 3, sigmoid                            → 128×128×3
```

- **The bottleneck is what denoises.** Noise is random, so it does not compress: there is no room to keep it. Reconstructing a plausible image is cheaper than reproducing the grain.
- **UpSampling mirrors MaxPooling**, so the output has exactly the shape of the input — unlike the classifier, which collapses everything into one number.
- **sigmoid on the last layer**, with **3 channels**: the output must be pixels in [0, 1], like the normalised input. This is the one real difference from the MNIST workshop, which used a single channel.

**Loss: `mse`** rather than the workshop's `binary_crossentropy`. On MNIST, pixels are nearly binary, which makes cross-entropy natural; our photographs are continuous, denoising is a regression on pixel values, and the mean squared error is also what **PSNR** is computed from — so the training objective and the reported metric agree.
""")

code(r"""
def build_denoiser(name="denoiser"):
    model = Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(padding="same"),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(padding="same", name="bottleneck"),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.UpSampling2D(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.UpSampling2D(),
        layers.Conv2D(3, 3, activation="sigmoid", padding="same"),
    ], name=name)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model

denoiser = build_denoiser()
denoiser.summary()
""")

md(r"""
## 5. Training
Early stopping on the validation loss, restoring the best epoch — the same correction as in Livrable 1.
""")

code(r"""
history = denoiser.fit(
    train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                             restore_best_weights=True, verbose=1)],
    verbose=2)

h = pd.DataFrame(history.history); h.index += 1
h[["loss", "val_loss"]].plot(marker="o", figsize=(9, 4), title="Denoiser — mean squared error")
plt.xlabel("epoch"); plt.grid(alpha=0.3)
h.round(5)
""")

md(r"""
## 6. Evaluation on known pairs
Here we **do** have the ground truth, because we created the noise. The standard measure is the **PSNR** (peak signal-to-noise ratio), in decibels:

`PSNR = 10 · log10(1 / MSE)` for pixels in [0, 1]

Higher is better; +3 dB means the squared error was halved. The honest comparison is **noisy vs clean** against **denoised vs clean**: the difference is what the model actually bought.
""")

code(r"""
def psnr(a, b):
    return float(tf.reduce_mean(tf.image.psnr(a, b, max_val=1.0)))

rows = []
for noisy_b, clean_b in val_ds.take(10):
    out = denoiser.predict(noisy_b, verbose=0)
    rows.append({"noisy vs clean": psnr(noisy_b, clean_b),
                 "denoised vs clean": psnr(tf.convert_to_tensor(out), clean_b)})
res = pd.DataFrame(rows).mean()
res["gain (dB)"] = res["denoised vs clean"] - res["noisy vs clean"]
res.round(2).to_frame("PSNR (dB)")
""")

code(r"""
noisy_b, clean_b = next(iter(val_ds))
out = denoiser.predict(noisy_b, verbose=0)
fig, axes = plt.subplots(3, 6, figsize=(15, 7.5))
for i in range(6):
    axes[0, i].imshow(noisy_b[i]); axes[1, i].imshow(out[i]); axes[2, i].imshow(clean_b[i])
    for r in range(3):
        axes[r, i].axis("off")
for r, lab in enumerate(["noisy", "denoised", "original"]):
    axes[r, 0].set_title(lab, loc="left")
plt.tight_layout()
""")

md(r"""
## 7. Application to TouNum's real images
The real test. Here there is **no ground truth**, so PSNR is impossible and the evaluation has to be indirect:

1. **Visual**: is the grain gone, and is the detail still there?
2. **The sigma estimator of section 2**, applied before and after.
3. **The Livrable 1 classifier**: if denoising really restores the appearance of a photograph, its confidence `p(photo)` should rise. This uses one deliverable to measure the other.
""")

code(r"""
real = tf.stack([read_image(p, IMG_SIZE) for p in noisy_paths])
real_out = denoiser.predict(real, verbose=0)

before = np.array([estimate_sigma(im) for im in real.numpy()[:40]])
after = np.array([estimate_sigma(im) for im in real_out[:40]])
print(f"estimated sigma  before {before.mean():.3f}   after {after.mean():.3f}")

up = tf.image.resize(tf.convert_to_tensor(real_out), (CLS_SIZE, CLS_SIZE)) * 255.0
p_after = classifier.predict(up, verbose=0).ravel()
print(f"p(photo) mean    before {p_photo.mean():.3f}   after {p_after.mean():.3f}")
print(f"kept as photos   before {(p_photo >= THRESHOLD).sum()}   after {(p_after >= THRESHOLD).sum()}  of {len(noisy_paths)}")
""")

code(r"""
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for i in range(6):
    axes[0, i].imshow(real[i]); axes[0, i].axis("off")
    axes[1, i].imshow(real_out[i]); axes[1, i].axis("off")
axes[0, 0].set_title("TouNum, noisy", loc="left"); axes[1, 0].set_title("denoised", loc="left")
plt.tight_layout()
""")

md(r"""
*Comment here on the three measurements. Watch in particular for the gap between the synthetic case and the real one: our noise is Gaussian and uniform, while a real scan may carry compression artefacts, blur or coloured noise. A model trained on one kind of noise has no reason to excel on another — say so rather than hide it.*
""")

md(r"""
## 8. The autoencoder as a feature extractor
The workshop *Autoencodeur et classification binaire* uses the encoder differently: not to rebuild an image, but to **describe** it compactly, and then classify from that description.

We reuse the encoder just trained — its bottleneck is 32×32×32 — pool it into 32 numbers per image, and train a simple classifier on those. The comparison with the CNN of Livrable 1 is the point: **how much is lost by replacing a supervised network with 32 unsupervised features?**
""")

code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

encoder = keras.Model(denoiser.input,
                      layers.GlobalAveragePooling2D()(denoiser.get_layer("bottleneck").output),
                      name="encoder")
print("encoder output:", encoder.output_shape)

SOURCES = ["Photo", "Painting", "Schematics", "Sketch", "Text"]
rows = []
for src in SOURCES:
    folder = DATA_DIR / f"Dataset Livrable 1 - {src}" / src
    for p in sorted(folder.glob("*.jpg"))[:1200]:
        rows.append({"path": str(p), "source": src, "is_photo": int(src == "Photo")})
sub = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(len(sub), "images encoded")

ds = (tf.data.Dataset.from_tensor_slices(sub.path.values)
      .map(load_clean, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE))
Z = encoder.predict(ds, verbose=0)
y = sub.is_photo.values
print("features:", Z.shape)

cut = int(0.8 * len(Z))
clf = LogisticRegression(max_iter=2000).fit(Z[:cut], y[:cut])
print(classification_report(y[cut:], clf.predict(Z[cut:]), target_names=["other", "photo"], digits=4))
""")

code(r"""
# What do the 32 features see? PCA of the latent space, coloured by source.
from sklearn.decomposition import PCA

P = PCA(n_components=2).fit_transform(Z)
plt.figure(figsize=(7, 6))
for src in SOURCES:
    m = sub.source.values == src
    plt.scatter(P[m, 0], P[m, 1], s=4, alpha=0.5, label=src)
plt.legend(markerscale=3); plt.title("Autoencoder latent space (PCA)"); plt.grid(alpha=0.3)
""")

md(r"""
*Compare the F1 on the photo class with that of Livrable 1, and say why the difference exists. Two reasons to weigh: these 32 features were learned to **reconstruct**, not to **separate** — nothing told them that "photo versus painting" matters; and pooling the bottleneck throws away the spatial layout. A supervised CNN optimises exactly the quantity being measured, so it should win; the question is by how much, and at what cost in labels.*
""")

md(r"""
## 9. Analysis and improvements

| Lever | Expected effect | Status |
|---|---|---|
| **Noise drawn per image and per epoch** | The model handles a range of levels, not one | Applied |
| **Early stopping** | Keeps the best epoch | Applied |
| **Larger images** (256 px) | Fine detail survives the reconstruction | Not done: CPU |
| **Skip connections (U-Net)** | Carry detail from encoder to decoder; the usual cure for the blur of a plain autoencoder | The first thing to try |
| **Wider bottleneck** | Less blur, but less denoising — the trade-off to measure | To try |
| **Perceptual loss** | MSE favours blurry images, since blurring is the safest way to reduce squared error | Advanced |
| **Real noise instead of Gaussian** | Scans also carry JPEG artefacts and blur | Would need clean/noisy pairs from TouNum |

**For TouNum's pipeline**, the honest conclusion to present: the model removes the kind of noise we modelled, measured by PSNR on known pairs, and the visual result on the real batch is *(fill in)*. To go further we would ask the client for a sample of images **scanned twice**, once in fast mode and once in quality mode — that would give real pairs and remove every assumption we had to make here.
""")

code(r"""
Path("models").mkdir(exist_ok=True)
denoiser.save("models/denoiser_rgb.keras")
print("saved models/denoiser_rgb.keras")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "livrable2_denoising.ipynb")
print("Wrote livrable2_denoising.ipynb with", len(cells), "cells")
