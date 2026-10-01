"""Complete WS_Autoencodeur_et_traitement_d_image (denoising CAE).

Fills every code blank, writes the theory answers, and adapts the 2020 code to
Keras 3. Markers become "# Solution" / "**Solution:**".
"""
from pathlib import Path
import shutil
import nbformat

CESI = Path(r"C:\my\CESI\A5\Data Science")
PROJECT = Path(r"C:\my\Claude\cesi-data-science")
NAME = "WS_Autoencodeur_et_traitement_d_image_EN_relu.ipynb"

CODE = {
9: '''import tensorflow as tf
from tensorflow import keras
import numpy as np
# Solution
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# The labels are discarded (_): a denoiser never needs to know which digit it is.
print(x_train.shape, x_test.shape)''',

12: '''import matplotlib.pyplot as plt

def display_image(X, n):
    plt.figure(figsize=(20, 2))
    # Solution
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)          # one row, n columns, positions start at 1
        plt.imshow(X[i].reshape(28, 28))   # 28x28: IMG_SIZE is only defined further down
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()''',

18: '''noise_factor = 0.5
# Solution: add Gaussian noise (mean 0, standard deviation 1) scaled by noise_factor
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# clip keeps the pixels in [0, 1]: noise can push them outside, and the sigmoid
# output of the network could never reproduce values below 0 or above 1.
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)''',

20: '''# Solution
display_image(x_test, n=10)          # originals
display_image(x_test_noisy, n=10)    # same images, with Gaussian noise''',

21: '''# Main configurations of our models
IMG_SIZE          = 28                 # final dimension of an image in pixels (here 28x28)
NB_EPOCHS_DENOISE = 20                 # 100 in the original; 20 is enough and far faster on CPU
BATCH_SIZE        = 128                # batch size
SAV_MODEL_DENOISE = "denoiser.keras"   # Keras 3 saves models as .keras, not .h5''',

24: '''from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

# The encoding process
# Solution
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Encoding #

# Conv1 #
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
# kernel_size: Specifying the height and width of the 2D convolution window.
x = MaxPooling2D((2, 2), padding='same')(x)              # 28x28 -> 14x14

# Conv 2 #
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)        # 14x14 -> 7x7

# Note:
# padding is a hyper-arameter for either 'valid' or 'same'.
# "valid" means "no padding".
# "same" results in padding the input such that the output has the same length as the original input.
print(encoded.shape)   # (None, 7, 7, 32): the compressed representation''',

27: '''# Decoding #
# Solution: the mirror of the encoder. UpSampling2D doubles height and width,
# undoing what MaxPooling2D halved.

# DeConv1
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)                               # 7x7 -> 14x14

# DeConv2
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)                               # 14x14 -> 28x28

# Deconv3
# sigmoid: one channel of pixel values in [0, 1], matching the normalised input
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
print(decoded.shape)   # (None, 28, 28, 1): same shape as the input image''',

28: '''from tensorflow.keras.models import Model
# Declare the model
# Solution
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy')
autoencoder.summary()''',

32: '''%load_ext tensorboard''',

33: '''# Train the model
# Solution: noisy image IN, clean image OUT. That pairing is what teaches denoising.
history = autoencoder.fit(x_train_noisy, x_train,
                epochs=NB_EPOCHS_DENOISE,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./tb_logs', histogram_freq=0, write_graph=False)]
               )''',

35: '''# Visualization of learning (Train) and validation (Test) losses
# Solution
plt.plot(history.history['loss'],
         label='train')
plt.plot(history.history['val_loss'],
         label='test')
plt.legend()''',

38: '''# save the model
# Solution
autoencoder.save(SAV_MODEL_DENOISE)
print("saved to", SAV_MODEL_DENOISE)
# reload later with: keras.models.load_model(SAV_MODEL_DENOISE)''',

39: '''# Solution: run the trained denoiser on the noisy test images
decoded_imgs = autoencoder.predict(x_test_noisy)
print(decoded_imgs.shape)''',

40: '''# Noisy input, denoised output, and the original for comparison
display_image(x_test_noisy, n=10)
display_image(decoded_imgs, n=10)
display_image(x_test, n=10)''',
}

MARKDOWN = {
16: '''Noise in an image is usually quite visible to the naked eye and can be very annoying. With an auto-encoder, it is possible to remove noise grains from images in order to see the main objects clearly, or for pre-processing before applying a classification auto-encoder, for example.
To achieve this, we need to create a convolutional auto-encoder model consisting of :
* encoder composed of three convolutional layers
* decoder composed of the inverse of the encoder layers.

What are the input and output data of the auto-encoder?

**Solution:** the **input is the noisy image** and the **target output is the clean original image**. This is the one difference from the autoencoder of the previous workshop, where input and target were the same image (`fit(x_train, x_train)`). Here we call `fit(x_train_noisy, x_train)`.

The consequence is that the network cannot simply learn to copy its input: copying would reproduce the noise and be punished by the loss. To reduce the loss it has to learn what a digit looks like, and treat anything that does not fit that structure as noise to be discarded. The bottleneck (7x7x32) helps, because random noise is not compressible: there is no room to store it.

It is still **self-supervised** — no human labelled anything. We created the pairs ourselves by adding noise we generated, so the clean image is a free, exact target.''',

36: '''What do you think of the model's performance?

**Solution:** read the two curves together rather than the absolute loss value (binary cross-entropy on pixels has no intuitive scale).

- The training and validation losses fall together and stay close, which indicates **no over-fitting**. That is expected here: the network has only about 28,000 parameters for 60,000 training images, the exact opposite of the ratio in the CNN workshop, and the noise is drawn afresh for every image, which acts as a natural augmentation.
- Most of the progress happens in the first few epochs, then the curve flattens. Training longer gains very little, which is why 20 epochs are enough; the original 100 mostly costs time.
- The honest evaluation is **visual**: compare the noisy, denoised and original rows. The digits come back clearly readable, with the background returned to black, but the strokes are slightly blurred and thinner than the originals. Fine detail is lost because the bottleneck keeps only what is needed to reconstruct a plausible digit.
- A limitation to state: the model was trained on one kind of noise, Gaussian with `noise_factor = 0.5`. It has no reason to work as well on a different noise (blur, JPEG artefacts, scratches) or on images that are not handwritten digits. For **Livrable 2**, where the images are RGB photographs, the last layer must output 3 channels and the network must be trained on that kind of data.

A quantitative check to add if you want to be rigorous: compute the mean squared error between noisy and original, then between denoised and original, and show the second is much smaller.

# 1.4 Auto-encoder backup''',
}


path = CESI / NAME
src = PROJECT / NAME
if not path.exists():                       # the file arrived in the project folder
    shutil.copy2(src, path)

nb = nbformat.read(path, as_version=4)
for idx, source in CODE.items():
    assert nb.cells[idx].cell_type == "code", f"cell {idx} is not code"
    nb.cells[idx].source = source
    nb.cells[idx].outputs, nb.cells[idx].execution_count = [], None
for idx, source in MARKDOWN.items():
    assert nb.cells[idx].cell_type == "markdown", f"cell {idx} is not markdown"
    nb.cells[idx].source = source

backup = path.with_suffix(".ipynb.bak")
if not backup.exists():
    shutil.copy2(src, backup)
nbformat.write(nb, path)
shutil.copy2(path, src)

left = sum("PLEASE COMPLETE" in c.source or "TO BE COMPLETED" in c.source for c in nb.cells)
print(f"{NAME}: {len(CODE)} code + {len(MARKDOWN)} markdown cells written, {left} markers left")
