"""Writes the remaining theory answers of WS_NLP_RNN_Etudiant, in place of the
markers so the workshop's own text and figures are kept."""
from pathlib import Path
import re
import shutil
import nbformat

CESI = Path(r"C:\my\CESI\A5\Data Science")
PROJECT = Path(r"C:\my\Claude\cesi-data-science")
NAME = "WS_NLP_RNN_Etudiant_EN_relu.ipynb"
MARKER = re.compile(r"<em>\s*(?:PLEASE COMPLETE|TO COMPLETE|TO BE COMPLETED)\s*</em>", re.I)

ANSWERS = {
13: ['''**Solution:** a tweet is written for humans, not for a model. Everything that varies without changing the meaning multiplies the vocabulary and leaves each word with fewer examples to learn from. We therefore remove, in this order:

| Element | Why |
|---|---|
| URLs (`http...`) | A unique string in almost every tweet; carries no meaning for the classifier |
| HTML tags (`&amp;`, `<br>`) | Encoding artefacts of the scraping |
| Case | "Fire", "FIRE" and "fire" must be one word, not three |
| Punctuation and special characters | `fire!` and `fire` should be the same token |
| Emojis and non-ASCII (`\\x89Û`) | Mojibake from a bad encoding; noise |
| Stopwords (*the, is, at*) | Very frequent, carry almost no discriminating information |
| Words of 1–2 characters | Mostly leftovers of the previous steps |
| Inflections (lemmatisation) | *burning*, *burned*, *burns* become one word |

**One caution worth stating in the report:** NLTK's English stopword list contains **"not"** and other negations. After cleaning, *"this is not a fire"* becomes *"fire"* — the opposite meaning. For a disaster classifier that is a real source of false positives, and keeping negations would be a defensible improvement.'''],

40: ['''**Solution:**
- **`dtrain` (80%)** is what the network learns from: its weights are updated on these tweets only.
- **`dtest` (20%)** is never learned from. It is passed as `validation_data`, so after each epoch we see the score on texts the model has not been fitted on. That is what reveals over-fitting and what we use to compare the GRU with the LSTM.

`stratify=D['target']` keeps the same proportion of disaster tweets in both parts, so the two scores are comparable.

Strictly speaking this set plays **two** roles here — measuring progress *and* choosing between the two models — so the final figure is slightly optimistic. A rigorous protocol splits three ways: train, validation and a test set touched once. (The Kaggle `test.csv` cannot play that role: it has no labels.)'''],

42: ['''**Solution:** the tokenizer builds a **dictionary** and then uses it as a translation table.

1. `fit_on_texts` reads all the sentences, counts every distinct word, and assigns each an integer **ranked by frequency**: the most common word gets index 1, the next 2, and so on. Index **0 is reserved for padding**, which is why the embedding needs `len(word_index) + 1` rows.
2. `texts_to_sequences` then rewrites each sentence as the list of its words' indices: `"forest fire near town"` → `[412, 36, 88, 205]`.

The numbers are **labels, not quantities**: word 412 is not "larger" than word 36, and nothing says 412 and 413 are related. Giving these integers meaning is exactly the job of the embedding layer that follows.

A word absent from the dictionary produces nothing at all (it is silently dropped) unless an `oov_token` is set — which is why this workshop fits the tokenizer on train, validation and test text together.'''],

44: ['''**Solution:** sentences have different lengths, but a tensor must be rectangular. `pad_sequences` adds zeros so that every row of the batch has the same length.

`padding='post'` puts those zeros **after** the sentence: `[412, 36, 88, 0, 0]`. With `padding='pre'` they would go before: `[0, 0, 412, 36, 88]`.

Zero is the index reserved by the tokenizer, so padding cannot be confused with a real word — the embedding simply maps it to row 0.

The choice matters for a recurrent layer, which reads left to right and whose final state is used for the prediction. With `post`, the last steps read padding and the useful signal can fade; with `pre`, the sentence ends right at the last step. This is handled properly by **masking** (`mask_zero=True` on the embedding), which tells the recurrent layer to ignore padded positions altogether — a worthwhile improvement to mention.'''],

49: ['''**Solution:** because the tokenizer's integers carry no meaning, and the obvious alternative does not scale.

**One-hot encoding** would give each word a vector as long as the vocabulary (~20,000 here) with a single 1. Two problems: the size, and the fact that every pair of words is equally distant — *fire* would be no closer to *flames* than to *birthday*.

An **embedding** maps each word to a short dense vector (100 numbers here) learned so that **words used in similar contexts land near each other**. That is what the cosine similarity between `good` and `nice` demonstrates in the next cells. The model can then generalise: having learned something about *fire*, it already knows a little about *flames* and *blaze*.

**GloVe** gives those vectors ready-made, trained on 6 billion words. With `trainable=False` our model inherits general English and only has to learn the disaster-specific decision from 6,000 tweets — a first, concrete case of **transfer learning**, which is the subject of the next phase.'''],

66: ['''**Solution:**

**1. Embedding layer.** A lookup table of shape (vocabulary + 1) × 100. It receives integers and returns one 100-dimension vector per word, so a sentence of *n* words becomes an *n* × 100 matrix. Here it is initialised with the GloVe matrix and frozen (`trainable=False`): no gradient flows into it, which saves about 2 million parameters and avoids distorting good vectors with very little data.

**2. Recurrent layer (GRU or LSTM).** It reads the word vectors **one at a time**, keeping a hidden state that summarises everything read so far. At each step it combines the new word with the previous state, reusing the same weights throughout, which is why any sentence length works. Returning only its final state, it hands the dense layer a single 128-number summary of the whole tweet — and unlike a bag of words, that summary depends on the **order**.

**3. Dense layer.** One neuron with a sigmoid, turning those 128 numbers into a single probability between 0 and 1: "is this tweet about a real disaster?". A threshold (0.5 by default) converts it into a decision.'''],

72: ['''**Solution:** the GRU has an **update gate** (z) and a **reset gate** (r).

```
z = σ(W_z · [h_prev, x])          update gate
r = σ(W_r · [h_prev, x])          reset gate
ĥ = tanh(W · [r ⊙ h_prev, x])     candidate state
h = (1 − z) ⊙ h_prev + z ⊙ ĥ      new state
```

- The **update gate** decides how much of the old state to keep versus how much of the new candidate to write. It plays the role of the LSTM's forget *and* input gates at once: whatever share is written in, the same share is forgotten. With z near 0 the state passes through unchanged, which is how information — and the gradient — survives many steps.
- The **reset gate** decides how much of the previous state is allowed to take part in computing the candidate. Near 0, it lets the cell ignore the history and start afresh, which is useful when a new idea begins.

Both are sigmoid, so their outputs lie in 0–1 and act **per dimension**: every one of the 128 state values has its own valve, recomputed for every word.'''],

80: ['''**Solution:** the LSTM has **three** gates plus a separate **cell state**, the horizontal line in the figure.

| Gate | Formula | Function |
|---|---|---|
| **Forget** | `f = σ(W_f · [h, x])` | How much of the existing cell state to erase |
| **Input** | `i = σ(W_i · [h, x])` | How much of the new candidate to write |
| **Output** | `o = σ(W_o · [h, x])` | How much of the cell state becomes the visible output |

With the candidate `g = tanh(W_g · [h, x])`, the two updates are:

```
c = f ⊙ c_prev + i ⊙ g        new cell state
h = o ⊙ tanh(c)               new hidden state
```

The key point is that the cell state is **added to**, not multiplied through, so with the forget gate near 1 information crosses many steps intact — this is the cure for the vanishing gradient discussed above.

Compared with the GRU: the LSTM can forget and write **independently** (two separate gates where the GRU has one), and it separates its memory (c) from what it exposes (h). It is more expressive and about 33% more expensive: 117,248 parameters against 88,320 for 128 units on a 100-dimension input.'''],

83: ['''**Solution:** *(replace the figures with those of your own run — the two `summary()` outputs and the two curves give them)*

Three things are worth comparing, not just accuracy:

| | GRU (model 1) | Bidirectional LSTM (model 2) |
|---|---|---|
| Parameters | 88,320 | 234,496 (two LSTMs of 117,248) |
| Validation accuracy | … | … |
| Epoch of best validation loss | … | … |
| Time per epoch | … | … |

What to expect, and to confirm against your curves:
- The **bidirectional LSTM usually scores slightly higher**, because each word is read with its left *and* right context — in a tweet, the word that disambiguates "fire" often comes after it.
- The gain is **small** relative to the cost: nearly 3× the parameters and roughly 2× the time per epoch. For an equal budget the GRU is often the better deal, which is why it is a common default.
- **Both over-fit quickly here**: a learning rate of 0.01 (ten times the usual) with 50 epochs on ~6,000 short texts. Expect the validation loss to reach its minimum within the first few epochs and then rise while training accuracy keeps climbing. Model 2's higher `SpatialDropout1D` (0.4 against 0.2) delays it a little.

The honest conclusion is usually: the two architectures are within about a point of each other, and the real limits are the dataset size and the absence of early stopping — not the choice between GRU and LSTM.'''],

90: ['''**Solution:** *(quote your own two numbers here)*

Read them against the baselines rather than in absolute terms:
- **Always answering "not a disaster"** scores about 57% on this dataset, so anything near that is worthless.
- Published results on this Kaggle competition sit around **80–83%** with this kind of model; the leaders, using transformers such as BERT, reach about 85%.

So a validation accuracy in the high seventies or low eighties means the model works, and that the remaining errors are mostly genuine ambiguity rather than a broken pipeline.

Two caveats to state plainly:
1. The **loss** is the more informative number, and after 50 epochs at a learning rate of 0.01 it has usually started climbing again — the reported weights are the last ones, not the best ones. **Early stopping with `restore_best_weights=True` would raise the score for free.**
2. The same 20% split was used for early feedback *and* for this final figure, so it is slightly optimistic.

A single accuracy also hides the asymmetry: look at the false positives and negatives below before concluding anything about whether this model could be deployed.'''],

91: ['''**Solution:** a **false positive** is a tweet the model flags as a real disaster when it is not: predicted 1, truth 0.

Typical causes in this dataset:
- **Metaphor and hyperbole** — "this exam was a disaster", "my ex is a natural catastrophe". The vocabulary is identical to a real emergency; only the intent differs.
- **News, films, songs** quoting disaster words.
- **Over-cleaned negations** — "not a fire" becomes "fire" once stopwords are removed, as noted at the start.

Operationally this is the **cheaper** error: an alerting system raises an alarm that a human dismisses. Costly only at scale, where it becomes alert fatigue.'''],

93: ['''**Solution:** a **false negative** is a real disaster the model misses: predicted 0, truth 1.

Typical causes:
- The tweet describes the event **without the expected vocabulary** — "the whole street is under water", "we can't breathe in here".
- **Place names and proper nouns** absent from GloVe, which the embedding therefore maps to a zero vector.
- Very **short** tweets, left with two or three words after cleaning.

This is the **expensive** error: an unreported emergency. For TouNum's equivalent in Livrable 1, a photo wrongly discarded is never captioned.

That asymmetry is an argument for **not** leaving the threshold at 0.5: lowering it raises recall at the cost of precision, and the right trade-off is a business decision, not a statistical one.'''],
}


path = CESI / NAME
nb = nbformat.read(path, as_version=4)
for idx, texts in ANSWERS.items():
    cell = nb.cells[idx]
    assert cell.cell_type == "markdown", f"cell {idx} is not markdown"
    found = MARKER.findall(cell.source)
    assert len(found) == len(texts), f"cell {idx}: {len(found)} markers, {len(texts)} answers"
    for text in texts:
        cell.source = MARKER.sub(lambda _m: text, cell.source, count=1)

nbformat.write(nb, path)
shutil.copy2(path, PROJECT / NAME)

left = [i for i, c in enumerate(nb.cells) if MARKER.search(c.source)]
print(f"{NAME}: {sum(len(v) for v in ANSWERS.values())} more answers written, markers left: {left}")
