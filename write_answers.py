"""Write the theory answers into the workshop notebooks.

Each marker (<em>PLEASE COMPLETE</em>, <em>TO COMPLETE</em>, ...) is replaced in
place by a **Solution:** block, so the workshop's own text, images and questions
are kept. A cell with two markers gets two answers, in order.
"""
from pathlib import Path
import re
import shutil
import nbformat

CESI = Path(r"C:\my\CESI\A5\Data Science")
PROJECT = Path(r"C:\my\Claude\cesi-data-science")
CNN_NAME = "WS_Reseaux_de_neurones_convolutifs_EN.ipynb"
AE_NAME = "WS_Autoencodeur_et_classification_EN_relu.ipynb"

MARKER = re.compile(r"<em>\s*(?:PLEASE COMPLETE|TO COMPLETE|TO BE COMPLETED)\s*</em>", re.I)

CNN = {
18: ['''**Solution: why a CNN rather than a classical dense network for images?**

1. **Parameter count.** A dense layer connects every input to every neuron. For a 180x180x3 image that is 97,200 inputs, so a single `Dense(128)` on raw pixels would need about 12.4 million weights. One filter of the first conv layer holds 3x3x3 + 1 = 28 weights, and the whole layer 448.

2. **Weight sharing and locality.** The same filter slides over the whole image, so a pattern is recognised wherever it appears (translation invariance); a dense network would have to learn "petal top-left" and "petal centre" as two unrelated things. Each conv neuron also sees only a small neighbourhood, which matches how images work: nearby pixels are related, distant ones usually are not.

3. **Hierarchy.** Stacking conv + pooling blocks lets each successive layer cover a wider area of the original image while keeping few parameters: edges, then textures, then shapes, then objects.'''],

24: ['''**Solution: the benefit in memory and computation time.**

From `model.summary()`:

| Part | Parameters |
|---|---|
| Conv2D 16 | 448 |
| Conv2D 32 | 4,640 |
| Conv2D 64 | 18,496 |
| **Three conv layers** | **23,584** |
| Dense 128 (after Flatten) | 3,965,056 |
| Dense 5 | 645 |
| Total | 3,989,285 |

The convolutions do all the image analysis with **0.6%** of the weights, while the single dense layer holds **99%**. The pooling layers are what make that possible: they take 180x180 down to 22x22, so `Flatten` yields 30,976 values instead of 97,200. Without pooling the same dense layer would need about 265 million weights — over a gigabyte in float32, and hopeless to train on a CPU.

Fewer parameters means less memory, fewer multiplications per image, and less capacity to memorise the training set.'''],

31: ['''**Solution: why is augmentation useful when the new images come from the training set?**

Strictly speaking it adds no new information *about flowers* — every pixel comes from an image we already had. What it adds is information about **what does not matter**, which we supply from our own knowledge of photographs.

Rotate a rose by 15 degrees and it is still a rose. Showing both versions with the same label teaches the network that orientation is irrelevant to the class; the same goes for a horizontal flip and a small zoom. The network can no longer memorise exact pixel positions, since they change at every epoch, so it must find features that survive the transformations — and those are exactly the features that generalise.

Two limits: the transformation **must preserve the label** (flipping a flower is safe, flipping scanned text or the digit 2 is not — worth remembering for Livrable 1), and augmentation cannot invent a species that is missing from the dataset. It reduces variance; it does not repair a biased sample.''',
'''**Solution: how does dropout help, and what are its advantages?**

At each training step dropout sets a random fraction of the values to zero, so a different, thinner network is trained every time. No neuron can become the dedicated detector of one training image, because it may be switched off at any moment, and none can rely on a specific partner being present — the network is forced to spread each representation across several neurons (it prevents *co-adaptation*).

In terms of size, dropout adds **zero parameters**, yet it lowers the *effective* capacity during training. It approximates training an ensemble of many smaller networks that share their weights and are averaged at prediction time, and ensembling is a classic way to reduce variance.

Advantages: almost free in computation, one hyper-parameter, it disables itself automatically at prediction time, and it combines with any other regularisation. Its limit: too high a rate starves the network of signal and causes under-fitting.'''],
}

AE = {
43: ['''**Solution: the effect of dimensionality reduction on the three classifiers.**

Read the heatmap column by column (raw = 784 features, pca = 10, tsne = 2); the effect differs per algorithm:

- **Naive Bayes** gains the most. On raw pixels it assumes the 784 features are independent given the class, which is badly false for an image — neighbouring pixels are strongly correlated. PCA components are **uncorrelated by construction**, much closer to the model's assumption, so the same algorithm does far better on 10 features than on 784.

- **SVM** also benefits. With 3,067 points in 784 dimensions, distances between points become nearly uniform (the *curse of dimensionality*) and a distance-based kernel loses its discriminating power. In 10 or 2 dimensions the neighbourhood structure is meaningful again.

- **Random Forest** changes least. It splits on one feature at a time and selects informative ones by itself, so the always-black border pixels are simply never chosen. It was already doing an implicit feature selection.

General intuition: reduction removes noise and redundancy, concentrates the signal into few descriptors, speeds up training and limits over-fitting — at the cost of whatever the discarded dimensions carried.

**A caveat about the t-SNE column.** t-SNE has no `transform()`: the 2D projection was computed on *all* the data before the train/test split, so the test points helped build their own representation. That is **data leakage**, and it flatters the t-SNE scores. It also explains why t-SNE could not serve in TouNum's pipeline: a newly scanned image cannot be projected without recomputing the whole embedding. PCA learns a reusable matrix and transforms new data forever after.'''],

44: ['''**Solution: impact on other tasks.**

- **Regression.** Same benefit as classification: fewer, uncorrelated descriptors give a more stable fit and less over-fitting, which matters when the number of features approaches the number of observations. The cost is interpretability — a coefficient on "component 3" cannot be explained to a client, unlike one on a measured variable.

- **Clustering.** Usually a clear gain, since k-means and friends rely on distances, which degenerate in high dimension; reduction also makes the result visualisable, which is how clustering is normally judged. The risk is that the projection invents structure: t-SNE produces well-separated blobs even from data that has none, and its cluster sizes and distances carry no meaning.

- **Anomaly detection.** Double-edged, and the most interesting case. An autoencoder trained on normal data reconstructs anomalies badly, so the **reconstruction error becomes the anomaly score** — the reduction *is* the detector. But if the anomaly lives in the low-variance directions PCA discards, reduction erases the very signal being hunted, and a rare defect is low-variance by definition.

- **Visualisation.** Projection to 2 or 3 dimensions is the only way to look at the data directly, which is why t-SNE survives despite its limits.

- **Compression and transfer.** The encoder output is a compact code: cheaper to store, transmit and feed to a later model. That is exactly Livrable 3 — a pre-trained CNN turns each photo into a compact feature vector, and the RNN writes the caption from that vector instead of the raw pixels.'''],
}


def patch(path: Path, answers: dict, label: str):
    nb = nbformat.read(path, as_version=4)
    for idx, texts in answers.items():
        cell = nb.cells[idx]
        assert cell.cell_type == "markdown", f"{path.name}: cell {idx} is not markdown"
        found = MARKER.findall(cell.source)
        assert len(found) == len(texts), f"{path.name} cell {idx}: {len(found)} markers, {len(texts)} answers"
        for text in texts:                      # replace one marker at a time, in order
            cell.source = MARKER.sub(lambda _m: text, cell.source, count=1)
    backup = path.with_suffix(".ipynb.answers.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    nbformat.write(nb, path)
    shutil.copy2(path, PROJECT / path.name)
    left = sum(len(MARKER.findall(c.source)) for c in nb.cells if c.cell_type == "markdown")
    print(f"{label}: {sum(len(v) for v in answers.values())} answers written, {left} markers left")


if __name__ == "__main__":
    patch(CESI / CNN_NAME, CNN, "CNN")
    patch(CESI / AE_NAME, AE, "Autoencoder")
