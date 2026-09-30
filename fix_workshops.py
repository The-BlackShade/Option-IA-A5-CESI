"""Fill the code blanks and fix Keras 3 / sklearn 1.9 incompatibilities in the
two CESI workshop notebooks. Written answers (<em>PLEASE COMPLETE</em>) are left
untouched on purpose. Re-runnable: it replaces whole cells by index.
"""
from pathlib import Path
import shutil
import nbformat

CESI = Path(r"C:\my\CESI\A5\Data Science")
PROJECT = Path(r"C:\my\Claude\cesi-data-science")

CNN = {
4: '''import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
# Keras now extracts into a sub-folder of the same name; without this the loader
# sees a single class instead of five.
if (data_dir / "flower_photos").is_dir():
    data_dir = data_dir / "flower_photos"
print(data_dir, [p.name for p in data_dir.iterdir() if p.is_dir()])''',

10: '''class_names = train_set.class_names
print(class_names)''',

12: '''import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for images, labels in train_set.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)          # 3x3 grid, positions start at 1
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")''',

14: '''print(type(train_set))
images, labels = next(iter(train_set))     # one batch out of the dataset
print(images.shape)
print(labels.shape)''',

16: '''AUTOTUNE = tf.data.AUTOTUNE                # tf.data.experimental.AUTOTUNE is deprecated

train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)''',

21: '''# Keras 3: layers live at layers.X, and the shape is declared once by an Input layer
model.add(layers.Input(shape=(image_h, image_w, 3)))
model.add(layers.Rescaling(1. / 255))''',

23: '''# Convolutional layer
model.add(
    layers.Conv2D(16, 3, padding="same", activation="relu")
)
# pooling layer
model.add(
    layers.MaxPooling2D()
)''',

25: '''# Convolutional block or filter size is (32, 3)
model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling2D())

# Convolutional block or filter size is (64, 3)
model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
model.add(layers.MaxPooling2D())

# Layer flattening
model.add(layers.Flatten())

# Fully connected layer (dense layer)
model.add(layers.Dense(128, activation="relu"))

# Fully connected layer returns classification result
# No activation: the layer outputs logits, hence from_logits=True in the loss.
model.add(layers.Dense(num_classes))''',

27: '''model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()''',

29: '''epochs=10
history = model.fit(train_set, validation_data=test_set, epochs=epochs)
history_base = history        # kept so sections 6 and 7 can be compared with this run

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()''',

32: '''# Le modele : le meme reseau, plus un Dropout avant le Flatten
model_with_dropout = Sequential([
    layers.Input(shape=(image_h, image_w, 3)),
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),          # regularisation: 20% of the values are zeroed while training
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes),
])
# Compilation du modele
model_with_dropout.compile(optimizer="adam",
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
# Resume du modele
model_with_dropout.summary()
# Entrainement du modele
history = model_with_dropout.fit(train_set, validation_data=test_set, epochs=epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()''',

34: '''# Keras 3: no more layers.experimental.preprocessing, and no input_shape on a layer
data_augmentation = keras.Sequential(
  [
    layers.Input(shape=(image_h, image_w, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(18 / 360),   # 18 degrees, expressed as a fraction of a full turn
    layers.RandomZoom(0.1),            # 10% zoom
  ]
)''',

36: '''# The model: augmentation first, then the same network with dropout
complete_model = Sequential([
    layers.Input(shape=(image_h, image_w, 3)),
    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes),
])
# Compiling the model
complete_model.compile(optimizer="adam",
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
# Model summary
complete_model.summary()
# Model training
history = complete_model.fit(train_set, validation_data=test_set, epochs=epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()''',
}

AE = {
6: '''# The CSV sits next to this notebook; the fallback keeps it working from elsewhere.
import os
CSV = 'mnist_test.csv'
if not os.path.exists(CSV):
    CSV = r'C:\\my\\CESI\\A5\\Data Science\\mnist_test.csv'

df = pd.read_csv(CSV, header=None)
df['pixels'] = df.index.map(lambda x: np.array(df.iloc[x][1:]))
dropcols = df.columns[(df.columns != 0) * (df.columns != 'pixels')]
df.drop(dropcols, axis=1, inplace=True)
df.columns = ['label','pixels']
print(df.shape)''',

13: '''## Figures considered ##
labels = [1,6,8]
colors = ['red', 'blue', 'green']

X = np.array([df['pixels'][i] for i in df.index if df['label'][i] in labels])
y = np.array([df['label'][i] for i in df.index if df['label'][i] in labels])

print('X shape: '+str(X.shape))
print('y shape: '+str(y.shape))''',

19: '''## PCA calculation
pca = PCA(n_components=10)
XPCA = pca.fit_transform(X)          # fit = find the axes, transform = project onto them
print('variance kept:', pca.explained_variance_ratio_.sum().round(3))''',

28: '''## Calculation of t-SNE 2D projection

## Parameters with a real influence on accuracy
perplexity = 30
learning_rate = 200
n_iter = 1000

# sklearn >= 1.5 renamed n_iter to max_iter
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, max_iter=n_iter)
XTSNE = tsne.fit_transform(X)        # t-SNE has no .transform(): new points cannot be projected later''',

36: '''def fit_my_model(model, features, test_size):

    ## Building train and test sets from all the  features
    if features == 'raw':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    elif features == 'tsne':
        X_train, X_test, y_train, y_test = train_test_split(XTSNE, y, test_size=test_size)
    else:
        # features holds an integer: keep that many PCA components
        X_train, X_test, y_train, y_test = train_test_split(XPCA[:, :features], y, test_size=test_size)

    print("Training samples: "+str(X_train.shape[0]))
    print("Testing samples: "+str(X_test.shape[0]))
    print("Number of features: "+str(X_train.shape[1]))

    ## Fit model
    if model == 'nb':
        clf = GaussianNB()
    elif model == 'svm':
        clf = SVC(gamma='auto')
    elif model == 'rf':
        clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, max_features=np.min([10, X_train.shape[1]]))

    clf.fit(X_train, y_train)

    ## Print scores
    learningScore = clf.score(X_train, y_train)
    generalizationScore = clf.score(X_test, y_test)
    print('Learning score: '+str(learningScore))
    print('Generalization score: '+str(generalizationScore))

    return generalizationScore''',

55: '''latent_space_dim = 15

# Encoder architecture (functional API: each layer is called on the previous tensor)
encoder_inputs = keras.Input(shape=(784,))
hidden1 = layers.Dense(200, activation="relu")(encoder_inputs)
latent_space = layers.Dense(latent_space_dim, activation="relu")(hidden1)

encoder = keras.Model(encoder_inputs, latent_space, name="encoder")
encoder.summary()''',

57: '''import os
# Graphviz is installed but not on PATH; plot_model needs the dot executable.
if os.path.isdir(r"C:\\Program Files\\Graphviz\\bin"):
    os.environ["PATH"] += r";C:\\Program Files\\Graphviz\\bin"

from keras.utils import plot_model
plot_model(encoder, to_file='encodersimple.png', show_shapes=True)''',

59: '''# Decoder architecture: the mirror of the encoder
decoder_inputs = keras.Input(shape=(latent_space_dim,))
hidden2 = layers.Dense(200, activation="relu")(decoder_inputs)
# sigmoid: the output must be pixel values in [0, 1], like the normalised input
decoder_outputs = layers.Dense(784, activation="sigmoid")(hidden2)

decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
decoder.summary()''',

63: '''# Combining the two Architecture (Enc,Dec): the decoder is called on the encoder output,
# so both models share the very same weights.
outputs = decoder(latent_space)
autoencoder = keras.Model(encoder_inputs, outputs, name="autoencoder")''',

67: '''# Keras 3 note -------------------------------------------------------------
# The original cell built the loss from symbolic tensors and attached it with
# model.add_loss(). Keras 3 raises NotImplementedError for that, and
# keras.backend.mean no longer works either. The loss is now declared in
# compile() (next cell): fit(X_train, X_train) compares output with input.
#
# Original intent, for the record:
#   reconstruction_loss = binary_crossentropy(input, output) * 784   # sum over pixels
#   autoencoder_loss    = mean(reconstruction_loss)                  # mean over the batch
# compile(loss="binary_crossentropy") is the same thing without the x784 scale.''',

69: '''# Compiling the model (Keras 3 style)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")''',

71: '''# Executing the model: the target IS the input -> self-supervised learning
history = autoencoder.fit(X_train, X_train,
          epochs=30,
          batch_size=128,
          shuffle=True,
          validation_data=(X_test, X_test))''',

73: '''# Keras 3 requires the weights file to end with .weights.h5
autoencoder.save_weights('./autoencoder.weights.h5')''',

84: '''# Coding the specific sampling layer as a Keras Layer object.
# Keras 3: keras.ops replaces the old backend functions (K.shape, K.exp, ...)
from keras import ops, random

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_logvar = inputs

        # Reparameterisation trick: sample z = mean + noise * std, so gradients
        # can still flow back through z_mean and z_logvar.
        eps = random.normal(shape=ops.shape(z_mean), mean=0.0, stddev=0.1)
        z = z_mean + eps * ops.exp(z_logvar)

        # Kullback-Leibler divergence: pulls each encoded distribution towards
        # N(0, 1), which is what keeps the latent space continuous.
        # Added here as a layer loss, because model.add_loss() with symbolic
        # tensors is not supported in Keras 3.
        kl = -0.5 * ops.sum(1 + z_logvar - ops.square(z_mean) - ops.exp(z_logvar), axis=-1)
        self.add_loss(ops.mean(kl))

        return z''',

94: '''# Loss function definition -------------------------------------------------
# The VAE loss has two terms:
#   reconstruction_loss : binary cross-entropy between input and output pixels
#                         -> declared in compile() below
#   kl_loss             : -0.5 * sum(1 + z_logvar - z_mean^2 - exp(z_logvar))
#                         -> computed inside the Sampling layer with self.add_loss()
# Keras adds the layer losses to the compiled loss automatically, so
# vae_loss = reconstruction_loss + kl_loss without any further code here.''',

95: '''# Compiling the model (Keras 3 style; the KL term comes from the Sampling layer)
vae.compile(optimizer="adam", loss="binary_crossentropy")''',
}


def patch(path: Path, edits: dict):
    nb = nbformat.read(path, as_version=4)
    for idx, source in edits.items():
        cell = nb.cells[idx]
        assert cell.cell_type == "code", f"{path.name}: cell {idx} is not code"
        cell.source = source
        cell.outputs, cell.execution_count = [], None
    backup = path.with_suffix(".ipynb.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    nbformat.write(nb, path)
    left = sum("PLEASE COMPLETE" in c.source for c in nb.cells if c.cell_type == "code")
    print(f"{path.name}: {len(edits)} cells rewritten, {left} code blanks left")


cnn_name = "WS_Reseaux_de_neurones_convolutifs_EN.ipynb"
ae_name = "WS_Autoencodeur_et_classification_EN_relu.ipynb"

patch(CESI / cnn_name, CNN)
patch(CESI / ae_name, AE)
for name in (cnn_name, ae_name):                 # keep the git copies identical
    shutil.copy2(CESI / name, PROJECT / name)
    print("copied to project:", name)
