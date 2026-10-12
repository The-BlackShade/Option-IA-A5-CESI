"""Complete WS_Captioning: CNN encoder, Bahdanau attention, GRU decoder on COCO."""
from pathlib import Path
import ast
import shutil
import nbformat

CESI = Path(r"C:\my\CESI\A5\Data Science")
PROJECT = Path(r"C:\my\Claude\cesi-data-science")
NAME = "WS_Captioning_EN_relu.ipynb"

CODE = {
5: '''# Annotation file path
annotation_folder = "/annotations/"
annotation_file = os.path.abspath('.')+"/annotations/captions_train2014.json"

# Path to folder containing images to be annotated
image_folder = '/train2014/'
PATH = os.path.abspath('.') + image_folder

# Read annotation file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Group all annotations with the same identifier.
image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    # mark the beginning and end of each annotation
    # Solution: <start> gives the decoder its first input, <end> is how it
    # learns to stop. Without them generation would never terminate.
    caption = '<start> ' + val['caption'] + ' <end>'
    # The image ID is part of the image path
    image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
    # Add caption associated with image_path
    # Solution
    image_path_to_caption[image_path].append(caption)

# Take first images only
image_paths = list(image_path_to_caption.keys())
train_image_paths = image_paths[:2000]

# List all annotations
train_captions = []
# List of all duplicated image filenames (in number of annotations per image)
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    # Add caption_list to train_captions
    # Solution
    train_captions.extend(caption_list)
    # Add duplicated image_path len(caption_list) times
    # Solution: one training pair per caption, so each image appears ~5 times
    img_name_vector.extend([image_path] * len(caption_list))

print(len(train_captions), "captions for", len(train_image_paths), "images")''',

6: '''# Download the pre-trained InceptionV3 model with cassification from ImageNet
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
# Creation of a variable that will be the input to the new image pre-processing model
# Solution
new_input = image_model.input
# retrieve the last hidden layer containing the image in compact representation
# Solution: with include_top=False this is the 8x8x2048 feature map, not the
# ImageNet classes. Keeping the grid (not pooling it) is what attention needs.
hidden_layer = image_model.layers[-1].output

# Model that calculates a dense representation of images with InceptionV3
# Solution
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Definition of load_image function
def load_image(image_path):
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
    return img, image_path

# Image pre-processing
# Take image names
encode_train = sorted(set(img_name_vector))

# Create an instance of "tf.data.Dataset" based on image names
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
# Data split into batches after pre-processing by load_image
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

# Browse dataset batch by batch for InceptionV3 pre-processing
for img, path in tqdm(image_dataset):
    # InceptionV3 pre-processes current batch (size (16,8,8,2048))
    batch_features = image_features_extract_model(img)
    # Resize batch size (16,8,8,2048) to (16,64,2048)
    batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
    # Browse current batch and store path and batch with np.save()
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        # (image path associated with its new representation , image representation)
        np.save(path_of_feature, bf.numpy())''',

8: '''# Find the maximum size
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Select the 5000 most frequent words in the vocabulary
top_k = 5000
#The Tokenizer class enables text pre-processing for neural networks
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\\\\]^_`{|}~ ')
# Builds a vocabulary based on the train_captions list
# Solution
tokenizer.fit_on_texts(train_captions)

# Create token to fill annotations to equalize length
# Solution: index 0 is the padding slot, in both directions of the dictionary.
# (The original line "tokenizer.index_word[0] = '<pad>'] = 0" was a typo.)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Creation of vectors (list of integer tokens) from annotations (list of words)
# Solution
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Fill each vector up to maximum annotation length
# Solution: post padding, so every caption starts at position 0
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the maximum length used to store attention weights
# This will be used later for display during evaluation
max_length = calc_max_length(train_seqs)
print("vocabulary:", min(top_k, len(tokenizer.word_index)), "| longest caption:", max_length)''',

10: '''img_to_cap_vector = collections.defaultdict(list)
# Creation of a dictionary associating image paths with (.npy file) annotations
# Images are duplicated because there are several annotations per image
print(len(img_name_vector), len(cap_vector))
for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)

"""
Creation of training and validation datasets using
random 80-20 splitting
"""
# Take the keys (names of processed image files), *these will not be duplicated*.
img_keys = list(img_to_cap_vector.keys())
# Solution: shuffle the IMAGES, not the captions. Splitting on captions would
# put four of an image's five sentences in train and one in validation, which
# leaks the image across the split.
random.shuffle(img_keys)
# Split indexes into training and test
slice_index = int(len(img_keys)*0.8)
# Solution
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

"""
Training and test sets are in the form of
lists containing mappings:(pre-processed image ---> annotation token(word) )
"""

# Loop to build training set
img_name_train = []
cap_train = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    # Duplicate images by number of annotations per image
    img_name_train.extend([imgt] * capt_len)
    cap_train.extend(img_to_cap_vector[imgt])

# Loop to build test set
img_name_val = []
cap_val = []
for imgv in img_name_val_keys:
    # Solution: the same construction as the training loop
    capv_len = len(img_to_cap_vector[imgv])
    # Duplication of images in the number of annotations per image
    img_name_val.extend([imgv] * capv_len)
    cap_val.extend(img_to_cap_vector[imgv])

len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)''',

15: '''class CNN_Encoder(tf.keras.Model):
    # Since the images are already preprocessed by InceptionV3, they are represented in compact form.
    # The CNN encoder will simply transmit these characteristics to a dense layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # form after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        # Solution: project each of the 64 regions from 2048 to embedding_dim,
        # so image features and word embeddings live in comparable spaces.
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x''',

17: '''class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # Solution: two projections (one for the image regions, one for the
        # decoder state) and a final layer that scores each region with 1 number.
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden layer shape == (batch_size, hidden_size)
        # Solution: add an axis so the state broadcasts over the 64 regions
        # -> (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # Solution: "additive" attention - project both, add, squash with tanh
        attention_hidden_layer = tf.nn.tanh(self.W1(features) +
                                            self.W2(hidden_with_time_axis))

        # This gives you a non-normalized score for each image feature.
        # Solution: (batch_size, 64, 1)
        score = self.V(attention_hidden_layer)

        # Solution: softmax over the 64 regions (axis=1), so the weights of one
        # image sum to 1. axis=-1 would be wrong: it has size 1.
        attention_weights = tf.nn.softmax(score, axis=1)

        # Solution: weight each region, then sum them into one context vector
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights''',

19: '''class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        # Solution: one trainable vector per word of the vocabulary
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        #Dense layer with GRU output as input
        # Solution
        self.fc1 = tf.keras.layers.Dense(self.units)
        # Last dense layer
        # Solution: one score per word of the vocabulary (logits)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        # Solution
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # Attention is defined by a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # Pass current word to embedding layer
        # Solution: (batch_size, 1) -> (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # Concatenation
        # Solution: the word the decoder just read, plus what it is looking at
        # -> (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Passage of the concatenated vector to the gru
        # Solution
        output, state = self.gru(x)

        # Dense layer
        # Solution
        y = self.fc1(output)

        # Fix: the original line reshaped with x.shape[2] (embedding_dim +
        # hidden_size), but y now has units columns. y.shape[2] is correct.
        y = tf.reshape(y, (-1, y.shape[2]))

        # Dense layer
        # Solution: logits over the vocabulary
        y = self.fc2(y)

        return y, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))''',

21: '''# Encoder creation
# Solution
encoder = CNN_Encoder(embedding_dim)
# Decoder creation
# Solution
decoder = RNN_Decoder(embedding_dim, units, vocab_size)''',

22: '''# ADAM optimizer
# Solution
optimizer = tf.keras.optimizers.Adam()
# The loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    # The mask removes the padding positions: the model must not be trained to
    # predict <pad>. reduction='none' is what makes this per-token masking work.
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)''',
}

path = CESI / NAME
src = PROJECT / NAME
if not path.exists():
    shutil.copy2(src, path)

nb = nbformat.read(path, as_version=4)
for idx, source in CODE.items():
    assert nb.cells[idx].cell_type == "code", f"cell {idx} is not code"
    ast.parse(source)
    nb.cells[idx].source = source
    nb.cells[idx].outputs, nb.cells[idx].execution_count = [], None

backup = path.with_suffix(".ipynb.bak")
if not backup.exists():
    shutil.copy2(src, backup)
nbformat.write(nb, path)
shutil.copy2(path, src)

left = [i for i, c in enumerate(nb.cells) if "PLEASE COMPLETE" in c.source]
print(f"{NAME}: {len(CODE)} cells written | blanks left: {left}")
for i, c in enumerate(nb.cells):
    if c.cell_type == "code":
        try:
            ast.parse(c.source)
        except SyntaxError as e:
            print("  SYNTAX cell", i, e)

