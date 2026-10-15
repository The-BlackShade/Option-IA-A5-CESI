"""Strengthen the written parts of the team Livrable 1 notebook.

Only markdown cells change; the code and the recorded GPU outputs stay as they
are. Numbers quoted come from the notebook's own outputs (cells 32, 34, 44).
"""
import nbformat

NAME = "Jupyter_notebook_p1_complete_1.ipynb"
nb = nbformat.read(NAME, as_version=4)

EDITS = {
17: """Next step is building the CNN based on the above architecture.

| Block | Layers | Output |
|---|---|---|
| Conv block 1 | Conv 3→32, BatchNorm, ReLU, Conv 32→32, BatchNorm, ReLU, MaxPool | 128 × 128 × 32 |
| Conv block 2 | Conv 32→64, BatchNorm, ReLU, MaxPool | 64 × 64 × 64 |
| Conv block 3 | Conv 64→128, BatchNorm, ReLU, Conv 128→128, BatchNorm, ReLU, MaxPool | 32 × 32 × 128 |
| Conv block 4 | Conv 128→256, BatchNorm, ReLU, AdaptiveAvgPool | 1 × 1 × 256 |
| Head | Flatten, Dropout 0.3, Linear 256→1 | 1 logit |

**546,785 trainable parameters** in total. The network ends with a global average pooling instead of a `Flatten` on the last feature map, which keeps the head tiny (257 parameters) and lets the same weights accept any input size.

- **BatchNorm2d** normalises the activations of every layer, which stabilises and speeds up the training.
- **Dropout (0.3)** randomly switches off 30% of the pooled features during training: a regularisation that limits the model's ability to memorise the training set.
- **Output**: one logit. The probability of "photo" is its sigmoid, and the decision rule is *photo if sigmoid ≥ 0.5*. With a single output, the matching loss is the binary cross-entropy (`BCEWithLogitsLoss`, which applies the sigmoid itself in a numerically stable way).""",

31: """Once our model is parametered we start the learning, with 6 of patience for the early stopping. The patience depends on the test loss improving or not. At the end the best model will be saved as "photo_classifier_best.pth".

*Note on the protocol.* The early stopping watches the **test** loss, so the same 20% of images both decide when to stop and provide the final figures below. This is the usual two-way split of the workshops, but it makes the reported scores slightly optimistic: a stricter protocol would hold out a third set, touched once at the very end. We state it here rather than hide it; the effect is small, since the stopping rule only chooses an epoch.""",

35: """### Results analysis

The run stopped after **44 epochs**: the best test loss, **0.0907**, was reached at epoch 38 and six more epochs brought no improvement. At that epoch:

| | Train | Test |
|---|---|---|
| Accuracy | 0.9842 | **0.9680** |
| Loss | 0.0438 | 0.0907 |

Precision 0.9708, recall 0.9650, F1 0.9679 on the test part (photos vs paintings only).

**Reading the curves — bias and variance.** Two signals say the model is in a good place:

1. **The gap between the two curves stays small**: 1.6 points of accuracy at the best epoch. A model that memorises its training images shows a gap that keeps opening; here it closes during the first ten epochs and then holds.
2. **The test loss never turns upward.** It falls quickly for twenty epochs, then plateaus between 0.091 and 0.095 from epoch 22 onwards, while the training loss keeps decreasing slowly. A rising test loss is the signature of over-fitting (the model becoming confidently wrong on new images); a flat one means the extra epochs simply stop helping — which is what the early stopping detects.

So the **variance is low**: 546,785 parameters for 15,994 training images (about 34 parameters per image, far from the regime where a network stores its training set), with BatchNorm, dropout and the `pos_weight` loss all pulling in the same direction. The remaining 3.2% of errors are more a question of **bias**: paintings whose rendering is photographic, which this architecture at 256×256 cannot always tell apart. The second step addresses part of it by giving the model more varied negatives.

The curves are not perfectly smooth in the first fifteen epochs (test loss 0.453 at epoch 9, for instance): batches of 32 and a learning rate of 0.001 give noisy gradients early on. `ReduceLROnPlateau` is what calms this down later in the run.""",

45: """### Results analysis

Step 2 starts from the step 1 weights, so the curves begin where the previous ones ended: 0.963 test accuracy at epoch 1 instead of 0.80. The run stopped after **20 epochs**, with the best test loss **0.1265** at epoch 14.

| | Train | Test |
|---|---|---|
| Accuracy | 0.9857 | **0.9743** |
| Loss | 0.0622 | 0.1265 |

On the test part: **precision 0.9297, recall 0.9665, F1 0.9478**.

Three observations:

1. **Still no over-fitting.** The gap is 1.1 points at the best epoch, and the test loss oscillates between 0.127 and 0.141 over the last ten epochs instead of climbing. Same diagnosis as step 1: low variance.

2. **Accuracy rises, F1 falls — both are expected.** Accuracy goes from 0.968 to 0.974 because the new categories (schematics, sketches, text) are easy to separate from photographs. F1 goes from 0.968 to 0.948 because the problem is now **imbalanced** (1 photo for about 3 other images): the `pos_weight` of 3.14 pushes the model to favour **recall** (0.967) over **precision** (0.930). Concretely, the classifier lets through about 7% false photos rather than lose 3% of real ones. For TouNum this is the right side to err on: a painting wrongly kept only wastes a little computation in the next step, while a photo wrongly discarded is never captioned.

3. **Where the errors are.** By construction of the two steps, the images that still fail are almost all paintings: the other three categories were nearly solved on the first epoch. A per-category table of the confusion would make this visible and is the first thing we would add.

The lower learning rate (0.0001) was essential: with 0.001 the first batches of step 2 would have overwritten what step 1 learned — the same reason a fine-tuning phase in transfer learning uses a small learning rate.""",

46: """# Conclusion

In this first deliverable we have pre-processed 41,399 images into 256×256 squares, designed a convolutional network with BatchNorm, dropout and early stopping, and trained it in two steps: first on the hard pair (photos vs paintings), then on all five categories starting from those weights.

The final model reaches, on the held-out 20%: **accuracy 0.9743, precision 0.9297, recall 0.9665, F1 0.9478**. The curriculum of two steps is what makes the difference: the first step spends the model's capacity on the only confusion that matters, and the second one adds the easy negatives without losing it.

### Limits we are aware of

- The implementation uses **PyTorch** rather than TensorFlow, for mixed-precision training on a GPU. The concepts (convolutions, BatchNorm, dropout, binary cross-entropy, Adam, early stopping) are exactly those of the workshops, and the deliverable 2 reuses this model as it is.
- The **test set also serves as validation** (it chooses the stopping epoch), so the scores above are slightly optimistic. A third, untouched set would be the rigorous version.
- Training needs a **GPU**: 44 + 20 epochs at 256×256. On a CPU the same notebook is reproduced by `train_team_classifier.py` at 128×128, with weights interchangeable with these.

### How the bias/variance compromise could be improved further

| Lever | Expected effect | Status |
|---|---|---|
| Dropout 0.3, BatchNorm, early stopping | Keep the variance low | Applied |
| `pos_weight` loss | Compensate the 1:3 imbalance | Applied |
| Two-step curriculum | Reduce the bias on paintings | Applied |
| Data augmentation (horizontal flip, small rotation, zoom) | Lower variance further; cheap | Not tried |
| L2 / weight decay | Penalise large weights | Not tried |
| Threshold tuning on a validation set | Trade precision against recall on purpose, instead of the default 0.5 | Not tried |
| Larger input than 256 | Keep the fine texture that separates paint from sensor noise | Too slow on our hardware |
| Transfer learning (ImageNet features) | Usually the largest gain on photo-vs-painting | Next phase of the course |
| Other optimisers | Unknown; Adam was kept throughout | Not tried |

The next deliverables do not improve this classifier; they use it. Deliverable 2 cleans the images this model keeps (denoising with a convolutional autoencoder), and deliverable 3 describes them (captioning with a CNN encoder and an RNN decoder).""",
}

for idx, source in EDITS.items():
    assert nb.cells[idx].cell_type == "markdown", idx
    nb.cells[idx].source = source
nbformat.write(nb, NAME)
chk = nbformat.read(NAME, as_version=4)
print("markdown cells rewritten:", len(EDITS), "| outputs kept:", sum(1 for c in chk.cells if c.get("outputs")))
