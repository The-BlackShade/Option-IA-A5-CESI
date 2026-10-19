# Projet Leyenda, revision of every workshop

Last-week revision, in easy words. One block per day. Every number below comes from your own runs, so you can say it at the defence as *our* result.

---

## 0. The basics every workshop uses

### A neural network is a chain of tiny calculations
Each neuron takes its inputs, multiplies each one by a weight, adds a bias, and passes the sum through an activation function. A layer is many neurons working on the same inputs. Training changes only the weights and biases. You choose the architecture; the data chooses the weights.

### Activation functions
- Without them, ten stacked layers would still be one straight-line formula, so the network could only learn straight-line rules.
- **ReLU** keeps positive numbers and turns negatives into 0. It is cheap, and its slope is 1, so the training signal does not shrink when it passes back through many layers. Used inside every model of the course.
- **Sigmoid** squashes any number into 0 to 1. It is the last layer of a yes/no model: photo or not, disaster tweet or not.
- **Softmax** turns several numbers into probabilities that add up to 1. Used for the five flowers and for the 5,000 words of the captioning vocabulary.

### Loss: the one number that training tries to lower
- **Mean squared error** is the average of the squared differences between output and target. Every autoencoder uses it, because the target is an image.
- **Cross-entropy** is for classes. It punishes confident mistakes very hard: saying "99% daisy" for a tulip costs far more than saying "60% daisy".
- That is why the validation loss can go up while the accuracy stays flat. The decisions do not change, but the model becomes more sure of its wrong ones. The loss is the earlier warning.
- *Binary* cross-entropy is the two-class version, *categorical* the many-class version, and *sparse* only means the labels are integers instead of one-hot vectors.

### Gradient descent, backpropagation, learning rate
- **Backpropagation** computes, for every weight, the slope of the loss: if this weight goes up a little, does the loss go up or down?
- **Gradient descent** then moves every weight a small step downhill. The step size is the **learning rate**.
- Too high: the weights jump over the valley and can land in a dead corner. Your GRU at 0.01 fell from 0.82 to 0.657 accuracy and stayed there, answering one class only.
- Too low: training crawls and stops early in a bad place.
- Rule used all through the course: start at 0.001 with Adam; divide by 10 when fine-tuning, or when a second training stage must not erase the first.

### Adam
- Plain gradient descent uses one step size for all weights. Adam keeps, for each weight, its average recent direction, which is momentum, and the average recent size of its gradient, and gives each weight its own step size.
- It is the safe default; every workshop used it.
- Side effect: Adam rescales each gradient by its own size, so multiplying the whole loss by a constant changes nothing. That is why removing the "× 784" in the autoencoder loss did not change the result.

### Epoch, batch, step

| Word | Meaning |
|---|---|
| Batch | The images used for one weight update, 32 in most of the course |
| Step | One update |
| Epoch | One full pass over the training set |

Steps per epoch = training size ÷ batch size. 2,936 flower images at 32 per batch give 92 steps; 60,000 MNIST digits at 128 give 469.

- Smaller batches: noisier updates, more of them per epoch, a small regularising effect. Larger batches: smoother and faster per epoch, but each epoch learns less.
- Changing the batch size from one epoch to the next is possible but pointless. Keep one value, 32 or 64, and tune the learning rate instead.

### Train, validation, test
- **Train**: the model learns from it.
- **Validation**: used during training to make choices: when to stop, which architecture, which threshold.
- **Test**: touched once, at the very end, to report the score.
- If the same set chooses the stopping epoch *and* gives the final score, the score is a little optimistic. The team Livrable 1 does this, and the notebook says so in its text.
- How to justify the sizes to the jury: the test set only needs to be big enough for a precise score. The margin of error of an accuracy p measured on n images is

```
± 1.96 × sqrt( p × (1 − p) / n )
```

| Test images | Margin at 90% accuracy |
|---|---|
| 148 | ± 4.8 points |
| 1,000 | ± 1.9 points |
| 8,280, which is 20% of Livrable 1 | ± 0.6 points |

Jury line: *"We keep enough test images for a precise score and give everything else to training, because more training data is the strongest cure for over-fitting. With 41,000 images, 20% is already far more than needed. With 148 images we would not split at all; we would use cross-validation."*

### Under-fitting, over-fitting, bias, variance
- **Under-fitting, high bias**: the model is too simple or trained too little. Train and validation scores are both poor and close together.
- **Over-fitting, high variance**: the model memorises. Train keeps improving while validation stalls or gets worse, so the gap widens.
- The two-second test on a curve: how big is the gap, and is the validation loss turning upward?

| Model | Train accuracy | Validation / test accuracy | Verdict |
|---|---|---|---|
| Flower CNN baseline | 99.2% | 63.8% | over-fits |
| Team photo classifier, step 2 | 98.6% | 97.4% | healthy |

- Cures for variance: more data, augmentation, dropout, early stopping, weight decay, a smaller model, transfer learning.
- Cures for bias: a bigger or deeper model, more epochs, better inputs, less regularisation.

### Early stopping
Watch the validation loss every epoch. If it has not improved for N epochs, the *patience*, stop and go back to the best epoch. Team notebook: patience 6, best epoch 38, stopped at 44.

### Accuracy, precision, recall, F1
- **Accuracy**: correct answers over all images. Misleading when classes are unbalanced: with 1 photo for 3 others, always answering "not a photo" scores 75%.
- **Precision**: of the images I called photo, how many really were.
- **Recall**: of the real photos, how many I caught.
- **F1**: the harmonic mean of the two; one number that drops if either is bad.

| Team classifier, final | Value |
|---|---|
| Accuracy | 0.9743 |
| Precision | 0.9297 |
| Recall | 0.9665 |
| F1 | 0.9478 |

Recall is higher than precision on purpose: the loss weight pushes the model to keep doubtful images. For TouNum that is the right side to err on. A lost photo is never captioned; a painting let through only wastes a little computing in the next step.

### Class imbalance
`pos_weight` = others ÷ photos = 3.14. It multiplies the loss of every photo, so 10,000 photos weigh as much as 31,000 others. Keras equivalent: `class_weight`. Other option: oversample the small class.

---

## 1. Workshop CNN: the flowers

- **Data**: 3,670 photos of 5 kinds, read from folders, 80/20 split, resized to 180×180, pixels divided by 255. Rescaling matters because weights start tiny; inputs of 0 to 255 would blow up the first layer.
- **Convolution**: a 3×3 filter slides over the image. At each position it multiplies and sums, and writes one number into a *feature map*. The map lights up where the filter's pattern is present. 16 filters give 16 maps. Filters are learned: the first layer finds edges and colour changes, deeper layers combine them into petals and whole flowers.
- **Why convolution beats dense for images**: the same 3×3 filter is reused at every position, so the layer has very few weights, 448 for the first one, and finds its pattern wherever it is. A dense layer on 180×180×3 would need 97,200 weights per neuron and would learn "petal at pixel 40,60" instead of "petal".
- **Padding** "same" keeps the size, "valid" shrinks by 2 per layer. **Stride** is how far the filter jumps.
- **Max pooling** 2×2 keeps the largest value in each square. It halves width and height, keeps the strongest responses, and makes the model tolerant to small shifts. No weights.
- **Flatten and Dense**: the last maps are unrolled into one long vector of about 31,000 numbers, fed to 128 neurons, then 5 outputs. This dense layer alone holds almost 4 million weights, nearly all of the model. That is why the team notebook replaces Flatten with global average pooling and has a head of 257 weights.
- **Output**: 5 logits with sparse categorical cross-entropy "from logits", which applies the softmax inside.
- **Result**: train 99.2%, validation 63.8% after 10 epochs. 2,936 images cannot constrain 4 million weights, so the network memorises them.
- **Fix 1, augmentation**: random flip, rotation and zoom applied to each batch on the fly, so the model never sees exactly the same image twice. Active only during training.
- **Fix 2, dropout 0.2**: during training each neuron is switched off at random one time in five, so no neuron can carry a memorised image alone. All neurons are on at test time.
- With both, the gap closes and the validation curve keeps rising. The exact numbers are the run you still have to finish for the answers of cells 30, 33 and 37.

Jury line: *"Our baseline memorised: 99% on training images, 64% on new ones. Augmentation and dropout attack the variance; after them the two curves stay close."*

---

## 2. Workshop autoencoder and classification: MNIST

- An **autoencoder** is two networks glued together. The **encoder** squeezes the 784 pixels into a few numbers, the **bottleneck**. The **decoder** rebuilds the 784 pixels from them. The loss compares the rebuilt image with the input, so no labels are needed: the image is its own answer.
- The bottleneck is the "compact description" you asked about: 32 numbers instead of 784, for example. To rebuild the digit, those numbers must keep what matters, the shape, and drop what does not, the exact pixel noise.
- **Latent space**: the space of these bottleneck vectors. Similar images land close together, which is what makes the next three tools work.
- **PCA** finds the straight directions along which the data spreads most, and projects onto the first two to draw a picture. Fast and exact, and the *explained variance ratio* says how much information the two axes keep. But it only sees straight-line structure.
- **t-SNE** draws a non-linear picture that keeps neighbours together: points close in 32 dimensions stay close in 2. The ten digits form clearer islands than with PCA. Costs: slow, random, and distances between islands mean nothing. It is for looking only, never for classifying.
- **Encoder as feature extractor**: train a classifier on the bottleneck numbers instead of the 784 pixels. On digits it works, because the shape is exactly what the bottleneck keeps. On photos, in Livrable 2, it gave F1 0.35 against 0.90 for the supervised classifier: a reconstruction bottleneck keeps colours and layout, not "is this a photograph". Lesson: features learned for one job are not automatically good for another.
- **VAE**: the encoder outputs a mean and a spread for each latent number, and we draw a random point from that little cloud:

```python
z = z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon   # epsilon drawn from N(0, 1)
```

- This is the **reparameterisation trick**: the randomness lives in epsilon, so gradients still reach the mean and the spread.
- A second loss term, the **KL divergence**, pushes every cloud toward the standard normal. Without it the encoder would shrink the clouds to points and the VAE would be a plain autoencoder. With it, the latent space has no empty holes, so you can pick any point and decode a believable new digit. A plain autoencoder reproduces; a VAE generates.
- The two terms fight: reconstruction wants small precise clouds, KL wants wide centred ones. The balance is the quality-versus-variety dial.
- Keras 3 detail we fixed: the KL loss is added inside the Sampling layer with `self.add_loss`, because `model.add_loss` with symbolic tensors no longer works.

---

## 3. Workshop denoising: MNIST

- You make the training pairs yourself: clean image plus Gaussian noise, clipped to 0..1, as input; the clean image as target. So "what did it look like before the noise" is literally the target the network learns to produce.
- **Convolutional, not dense**: a pixel is noise or stroke depending on its neighbours, and a convolution looks exactly at neighbourhoods. Far fewer weights too.
- **Architecture**: two conv + pool steps take 28×28 down to 7×7; two upsampling steps bring it back to 28×28; a final conv with one filter and a sigmoid keeps pixels in 0..1. Loss: mean squared error.
- **PSNR**, peak signal-to-noise ratio, in decibels: 10 × log10 of 1 over the MSE, for pixels in 0..1. Higher is cleaner; +3 dB means the error energy halved. The denoiser must beat the PSNR of the noisy input, otherwise it is not denoising.
- Why the validation loss sat below the training loss: the training loss is averaged over the epoch while the weights are still improving; the validation loss is measured at the end, with the best weights so far. No dropout here, so that is the whole reason. Not a bug.
- **TensorBoard** is just a callback that writes the losses to `tb_logs` so you can watch the curves live.

Livrable 2 is this workshop at full scale, in RGB, on the 148 noisy TouNum images:

| Step | Result |
|---|---|
| Noise measured on the real images | sigma 0.005, very mild |
| Model A, plain autoencoder | −2.93 dB, worse than doing nothing |
| Model B, residual with skip connections | +4.18 dB, from 25.57 to 29.75 |
| Probability of "photo" after cleaning | 0.669 to 0.731 |
| Images kept by the classifier | 102 to 115 of 148 |

- **Why A failed**: a bottleneck of 32 maps at quarter resolution cannot store the fine detail of a photo, so its output is blurrier than the noise it removes. With mild noise, blur costs more than the noise.
- **Why B works**: skip connections carry the full-resolution maps from encoder to decoder, so detail survives; and residual learning predicts the *noise* and subtracts it, an easy job because noise is "almost nothing" and the network starts from "change nothing". It is fully convolutional, so the same weights run at any image size.

Jury line: *"The first model lost 3 dB because reconstruction blur costs more than mild noise. Predicting the noise and keeping skip connections gained 4 dB, and the classifier downstream kept 13 more images."*

---

## Livrable 1 concepts that are not in the workshops

- **BatchNorm**: after a layer, normalises each channel over the batch to mean 0 and spread 1, then learns a scale and a shift. Keeps the numbers in a comfortable range: faster, more stable, slightly regularising. At test time it uses running averages.
- **Global average pooling**: averages each final map to one number, 256 numbers whatever the image size. It replaces Flatten, keeps the head at 257 weights, and makes the 128 px CPU weights and the 256 px GPU weights interchangeable.
- **Two-step curriculum**: train on the hard pair, photo vs painting, at 0.001; then all five categories at 0.0001 starting from those weights, so step 2 adds the easy negatives without erasing step 1.
- **ReduceLROnPlateau**: halves the learning rate when the validation loss stops improving.

---

## 4. Workshop NLP and RNN: disaster tweets

- **Task**: 7,613 tweets, "is it about a real disaster?". Yes/no, so sigmoid and binary cross-entropy again.
- **Cleaning**: lowercase; remove URLs, HTML, emojis, punctuation, numbers; remove stopwords such as "the" or "at"; lemmatise so "running" and "ran" become "run". Purpose: a smaller vocabulary where each word is seen more often. Bug we found: the workshop removed stopwords *before* lowercasing, so "The" survived.
- **Tokenizer**: builds a dictionary from word to integer over the training texts; a tweet becomes a list of integers. Only the most frequent words get an id.
- **Padding**: the network needs one fixed length, so each list is cut or filled with zeros to the chosen maximum. Masking tells the layers to skip the zeros.
- **Embedding**: each word id becomes a dense vector of 100 numbers, learned with the model, so words used in the same contexts get similar vectors: "fire" and "blaze" end up neighbours. A one-hot vector cannot express similarity; an embedding can.
- **GloVe**: embeddings pre-trained on 6 billion words. Loading them gives the model word meaning before it reads a single tweet, which helps a lot with only 7,000 examples. We loaded them through gensim because the direct download failed.
- **Simple RNN**: reads one word at a time and keeps a *hidden state*, a memory vector updated at each word. The last state summarises the tweet and goes to the sigmoid. Weakness: the training signal that travels back through 30 words is multiplied 30 times by small numbers and vanishes, so the network forgets the start of the sentence. That is the **vanishing gradient**.
- **LSTM**: adds a *cell state*, a conveyor belt that runs through time with only small additive changes, and three gates, each a sigmoid between 0 and 1. **Forget gate**: what to erase from the belt. **Input gate**: what to write on it. **Output gate**: what part of the belt to show as the hidden state. Because the belt changes by addition, gradients travel far without vanishing.
- **GRU**: a lighter LSTM with two gates. **Reset**: how much of the past to use when proposing a new memory. **Update**: how much old versus new to keep. Fewer weights, faster, usually the same results. It is the cell used in the captioning decoder.
- **Bidirectional**: one RNN reads left to right, another right to left, and their states are joined. Each word is understood with its past and its future. Good for classification, impossible for generation because the future is not written yet.
- **Your result**: the GRU at learning rate 0.01 collapsed from 0.82 to 0.657 and answered one class only. Fix: 0.001, dropout, early stopping. Around 0.80 is a normal score on this dataset.

---

## 5. Workshop transfer learning

- **Idea**: a network trained on ImageNet, 1.4 million photos in 1,000 classes, already knows edges, textures, shapes and objects. Only its last layer is about those 1,000 classes. Cut it off and put your own head on top.
- **InceptionV3**: 48 layers, 299×299 inputs, pixels scaled to −1..1 by its own preprocess function. With pooling "avg" it returns 2,048 numbers per image, a summary of what is in it.
- **Stage 1, feature extraction**: freeze the base, train only a small dense head on the 2,048 numbers. Fast, works with little data, hard to over-fit.
- **Stage 2, fine-tuning**: unfreeze the top blocks and train with a very small learning rate, 1e-5, so the pretrained weights move a little toward your images without being destroyed. Keep BatchNorm frozen, otherwise its running statistics get corrupted by your small batches.
- Works best when your images look like ImageNet: cats, dogs, photos. For X-rays or drawings, more layers must be fine-tuned.
- **Your workshop**: cats vs dogs from cifar10, because the original download breaks on Windows; from scratch versus frozen base versus fine-tuned, compared on the same curves. The second half extracted InceptionV3 features for COCO images: that is the bridge to captioning, and the 2,000 `.npy` files in the data folder are exactly those.

Jury line: *"Transfer learning is the biggest untried lever for photo versus painting; we list it as the next step of Livrable 1."*

---

## 6. Workshop captioning

- **Task**: image in, sentence out. The CNN encoder sees, the RNN decoder speaks, attention links the two.
- **Data**: COCO val2014, 40,504 images with 5 human captions each; we used 2,000. Each caption is wrapped as `<start> ... <end>`. The tokenizer keeps the 5,000 most frequent words, the rest become `<unk>`, `<pad>` is 0, and all captions are padded to the longest one.
- **Features**: InceptionV3 without its top gives 8×8×2,048 per image, reshaped to 64 vectors of 2,048: 64 regions of the picture. Computed once and saved as `.npy`, so training never re-runs the CNN.
- **Encoder inside the model**: one Dense 256 with ReLU applied to each of the 64 region vectors, to match the decoder's size.
- **Attention**: at each word, the decoder's hidden state asks "which of the 64 regions matter now?".

```python
score = V(tanh(W1(features) + W2(hidden)))   # one score per region
attention_weights = softmax(score, axis=1)   # 64 weights that sum to 1
context_vector = sum(attention_weights * features)
```

- The attention plots are these 64 weights laid back on the image: bright where the model looked while writing "horse".
- **Decoder**: embedding of the previous word, 256 numbers, joined with the context vector, into a GRU of 512 units, then Dense 512, then Dense 5,000: one logit per word. The GRU state carries the sentence so far.
- **Teacher forcing**: during training, the input at each step is the *true* previous word of the caption, not the model's own guess. Stable and fast. Downside, called exposure bias: at test time the model feeds itself, so one wrong word can derail the rest.
- **Loss**: sparse categorical cross-entropy per word, from logits, multiplied by a mask so padding costs nothing, averaged over the real words.
- **Training loop by hand** with GradientTape: reset the GRU state, loop over caption positions, add up the loss, compute the gradients of encoder and decoder together, apply them with Adam. Checkpoints save the weights each epoch.
- **Inference**: start with `<start>`, predict, take the most likely word, feed it back, stop at `<end>` or the maximum length. That is greedy decoding. Beam search keeps the best few partial sentences and gives better captions. BLEU scores a caption by word-group overlap with the human ones.
- **Livrable 3** is this pipeline on the photos that Livrable 1 kept and Livrable 2 cleaned. Not started; the features are extracted.

---

## Self-check: cover the answers

1. **Why divide pixels by 255?** Weights start tiny; inputs in 0..1 keep the first activations small and the training stable.
2. **Why did the flower baseline over-fit, and what are two cures?** About 4 million weights for 2,936 images; augmentation and dropout.
3. **What is the bottleneck for?** It forces the encoder to keep only what is needed to rebuild the image: the shape, not the noise.
4. **PCA or t-SNE as a classifier's input?** PCA. t-SNE is for pictures only; its distances mean nothing.
5. **Why does a VAE need the KL term?** Without it the clouds shrink to points and nothing can be generated between them.
6. **Why −3 dB for Model A and +4 dB for Model B?** A blurs more than it denoises; B keeps detail through skips and only predicts the noise.
7. **What do the three LSTM gates do?** Forget erases from the cell state, input writes to it, output chooses what to show.
8. **Why a tiny learning rate when fine-tuning?** Big steps would overwrite the pretrained weights in the first batches.
9. **Teacher forcing and its downside?** Feed the true previous word during training; at test time the model feeds itself and can derail.
10. **Why prefer recall over precision for TouNum?** A missed photo is never captioned; a wrongly kept painting only wastes compute.
11. **How do you justify 80/20?** The test set needs a precise score, ± 0.6 points at 8,280 images; everything else goes to training.
12. **What does attention give the decoder?** A weighted sum of the 64 region features, with weights chosen by the current hidden state.
