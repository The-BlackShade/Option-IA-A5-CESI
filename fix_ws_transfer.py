"""Complete WS_Transfer_Learning: InceptionV3 embeddings on cats vs dogs (part 1)
and the COCO pre-processing that feeds the captioning workshop (part 2).
"""
from pathlib import Path
import re
import shutil
import nbformat

CESI = Path(r"C:\my\CESI\A5\Data Science")
PROJECT = Path(r"C:\my\Claude\cesi-data-science")
NAME = "WS_Transfer_Learning_EN_relu.ipynb"
MARKER = re.compile(r"<em>\s*(?:PLEASE COMPLETE|TO COMPLETE|TO BE COMPLETED)\s*</em>", re.I)

CODE = {
8: '''# Load a tensorflow dataset

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    # Solution
    "cats_vs_dogs",
    #"cifar10",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)

# Nb of training samples
print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
# Nb of validation samples
print(
    # Solution
    "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
)
# Nb of test samples
print(
    # Solution
    "Number of test samples: %d" % tf.data.experimental.cardinality(test_ds)
)''',

12: '''# Resizing images

size = (299, 299)

# Solution: map applies the lambda to every (image, label) pair; only the image changes
train_ds = train_ds.map(
    lambda x, y: (tf.image.resize(x, size), y)
)
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))''',

13: '''# Preprocessing the images to give them as input to InceptionV3

# Solution: each pre-trained network expects its own input scaling. InceptionV3
# was trained on pixels mapped to [-1, 1], and preprocess_input does exactly that.
# Feeding it 0-255 pixels would silently produce poor embeddings.
preprocess = tf.keras.applications.inception_v3.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
validation_ds = validation_ds.map(lambda x, y: (preprocess(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess(x), y))''',

14: '''batch_size = 32

train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
# Solution: same treatment, but these two are never shuffled
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)''',

17: '''# Definition of embedding model using InceptionV3
inputs = keras.Input(shape=(299, 299, 3))
x = image_model(inputs)
# Solution: a model that runs from the input image to the 2048-number embedding.
# No classification head: the output IS the representation.
model = keras.Model(inputs, x)
model.summary()''',

19: '''# Calculation of test dataset image embeddings

# Solution: one 2048-dimension vector per test image
output = model.predict(test_ds)

print(output.shape)''',

22: '''# We apply the tSNE algorithm to embeddings to reduce their size
# and display them (embeddings size 2048 -> 2)
from sklearn.manifold import TSNE
# Solution: fit_transform, because t-SNE has no transform() of its own
tsne = TSNE(n_components=2, init="pca", random_state=42).fit_transform(output)
print(tsne.shape)''',

23: '''# We define the color associated with each class (only 2 classes are used in the cats vs dogs case)

colors = {0:'red',
          1:'blue'}

# Plotting the image embeddings calculated by InceptionV3
# Each point corresponds to an image and its coordinates are the image's embedding tsne
plt.figure(figsize=(10, 10))
plt.scatter(*tsne.T, s=1.5, c=[colors[l] for l in labels])''',

32: '''# Reading annotation file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Grouping all annotations with the same identifier.
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    # marking the beginning and end of each annotation
    # Solution: the RNN needs to know where a sentence starts and where to stop
    caption = '<start> ' + val['caption'] + ' <end>'
    # Image ID is part of the image path
    image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
    # Adding caption associated with image_path
    # Solution: defaultdict(list) creates the list on first use
    image_path_to_caption[image_path].append(caption)

print(len(image_path_to_caption), "images,", sum(len(v) for v in image_path_to_caption.values()), "captions")''',

35: '''# List of all annotations
train_captions = []
# List of all duplicated image file names (in number of annotations per image)
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    # Adding caption_list to train_captions
    # Solution
    train_captions.extend(caption_list)
    # Adding duplicate image_path len(caption_list) times
    # Solution: one training pair per caption, so the same image appears 5 times
    img_name_vector.extend([image_path] * len(caption_list))''',

38: '''# Downloading the pre-trained InceptionV3 model with cassification from ImageNet
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
# Creation of a variable that will be the input to the new image pre-processing model
# Solution
new_input = image_model.input
# retrieving the last hidden layer containing the image in compact representation
# Solution: with include_top=False the last layer is the 8x8x2048 feature map,
# not the ImageNet classes
hidden_layer = image_model.layers[-1].output

# Model that calculates a dense representation of images with InceptionV3
# Solution
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
image_features_extract_model.summary()''',

39: '''def load_image(image_path):
    """
    The load_image function has as input the path of an image and as output a pair
    containing the processed image and its path.
    The load_image function performs the following processing:
        1. Loads the file corresponding to the path image_path
        2. Decodes the image into RGB.
        3. Resize image to size (299, 299).
        4. Normalize image pixels between -1 and 1.
    """
    # Solution
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path''',

41: '''from tqdm import tqdm

# Pre-processing images
# Taking image names
encode_train = sorted(set(img_name_vector))

# Creating an instance of "tf.data.Dataset" based on image names
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# Data split into batches after pre-processing by load_image
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

# Browsing dataset batch by batch for InceptionV3 pre-processing
for img, path in tqdm(image_dataset):
    # InceptionV3 pre-processes the current batch (size (16,8,8,2048))
    # Solution
    batch_features = image_features_extract_model(img)
    # Resizing batch size (16,8,8,2048) to (16,64,2048)
    batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
    # Browsing current batch and store path and batch with np.save()
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        # (image path associated with its new representation , image representation)
        # Solution: np.save appends .npy, so the feature file sits next to the image
        np.save(
            path_of_feature, bf.numpy()
        )''',

43: '''# Finding the maximum size
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Selecting the 5000 most frequent words in the vocabulary
top_k = 5000
#The Tokenizer class enables text pre-processing for neural networks
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')
# Builds a vocabulary based on the train_captions list
# Solution
tokenizer.fit_on_texts(train_captions)

# Creating token to fill annotations to equalize length
# Solution: index 0 is the padding slot, in both directions of the dictionary
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Creation of vectors (list of integer tokens) from annotations (list of words)
# Solution
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Filling each vector up to maximum annotation length
# Solution: post padding, so the sentence starts at position 0
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the maximum length used to store attention weights
# It will later be used for display during evaluation
max_length = calc_max_length(train_seqs)
print("vocabulary:", min(top_k, len(tokenizer.word_index)), "| longest caption:", max_length)''',
}

ANSWERS = {
4: ['''**Solution:** freezing keeps the pre-trained weights exactly as they are, so that only the layers we added are learned.

Three reasons:
- **The features are already good.** The early layers of a network trained on ImageNet detect edges, textures and shapes — knowledge that is valid for almost any photograph, and that our small dataset could never produce on its own.
- **It protects them.** The new layers start with random weights, so the first gradients are large and chaotic. Let them through and they would destroy, in a few batches, features learned over millions of images.
- **It costs far less.** Only a handful of parameters are trainable, so training is faster, needs less memory, and over-fits much less on a small dataset.''',

'''**Solution:** fine-tuning unfreezes the pre-trained layers and continues training them, so that the generic features are **specialised** for the new task. ImageNet never saw TouNum's scanned documents; after fine-tuning, the filters adjust to the textures that matter here. It typically adds a few points of accuracy.

The **very low learning rate** (often 10 to 100 times smaller) exists because the pre-trained weights are already close to a good solution. A normal rate would take large steps and wipe out what was learned — the phenomenon is called *catastrophic forgetting*. Small steps adjust without destroying.

Two conditions: fine-tune **only after** the new head is trained (otherwise its random gradients damage the base), and prefer it when you have enough data, since all the parameters become trainable again and the over-fitting risk returns.'''],

16: ['''**Solution:** `include_top=False` removes the **classification head** — the final dense layers that output the 1,000 ImageNet classes.

Those classes (*golden retriever*, *espresso maker*…) are useless here, and the head is where the task-specific knowledge sits. What we keep is the convolutional base, which produces a generic representation. Without it, the model would return 1,000 probabilities instead of the embedding we want.''',

'''**Solution:** `weights='imagenet'` loads the weights **learned on ImageNet** (1.2 million labelled photographs) instead of random values.

This is the whole point of transfer learning: those weights carry the visual knowledge we are borrowing. With `weights=None` the architecture would be identical but untrained, and the "embeddings" would be a meaningless projection — the comment at the top of the cell says exactly this, and the t-SNE plot below is the proof: with ImageNet weights the cats and dogs separate into two clouds, with random weights they do not.''',

'''**Solution:** `pooling='avg'` adds a **global average pooling** layer at the output of the convolutional base.

Without it, the output is a feature map of shape 8×8×2048 — a grid. Global average pooling takes the mean of each of the 2048 channels over the 8×8 positions, producing a single vector of **2048 numbers** per image: one value per detected feature, regardless of where it appeared.

That is what makes the output usable as an embedding, and why `output.shape` is `(2326, 2048)` and not `(2326, 8, 8, 2048)`. Note that part 2 deliberately does *not* use pooling: the captioning model needs the 64 spatial positions so that it can attend to different regions while generating words.'''],

31: ['''**Solution:** a **dictionary mapping each image path to the list of its captions**, which is what `collections.defaultdict(list)` builds:

```python
{
  ".../COCO_train2014_000000318556.jpg": ["<start> a man riding a wave <end>",
                                          "<start> a surfer on a blue wave <end>", ...],
  ...
}
```

Why this shape:
- The COCO annotation file is organised **by caption** (five entries per image, each with an `image_id`). We need the opposite view, **by image**, to select whole images rather than isolated captions.
- Selecting a subset then means taking *n* keys, which guarantees that the five captions of a chosen image stay together — a train/test split made on captions would put four of an image's sentences in training and one in test, leaking the image.
- It also removes the duplication: the path is stored once, not five times.

The following cell flattens it back into two parallel lists (`train_captions` and `img_name_vector`), because training pairs are (image, one caption) — hence the same image path repeated five times.'''],
}


path = CESI / NAME
src = PROJECT / NAME
if not path.exists():
    shutil.copy2(src, path)

nb = nbformat.read(path, as_version=4)
for idx, source in CODE.items():
    assert nb.cells[idx].cell_type == "code", f"cell {idx} is not code"
    nb.cells[idx].source = source
    nb.cells[idx].outputs, nb.cells[idx].execution_count = [], None
for idx, texts in ANSWERS.items():
    cell = nb.cells[idx]
    assert cell.cell_type == "markdown", f"cell {idx} is not markdown"
    found = MARKER.findall(cell.source)
    assert len(found) == len(texts), f"cell {idx}: {len(found)} markers, {len(texts)} answers"
    for text in texts:
        cell.source = MARKER.sub(lambda _m: text, cell.source, count=1)

backup = path.with_suffix(".ipynb.bak")
if not backup.exists():
    shutil.copy2(src, backup)
nbformat.write(nb, path)
shutil.copy2(path, src)

left_code = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "PLEASE COMPLETE" in c.source]
left_md = [i for i, c in enumerate(nb.cells) if MARKER.search(c.source)]
print(f"{NAME}: {len(CODE)} code cells, {sum(len(v) for v in ANSWERS.values())} answers")
print("code blanks left:", left_code, "| markers left:", left_md)
