"""First TensorFlow model: recognise handwritten digits (MNIST).

Run from this folder:  .venv\\Scripts\\python.exe 01_first_model.py
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide the info messages at startup

import tensorflow as tf

# 1. Data: 70,000 images of digits 0-9, each 28x28 pixels (downloads once, ~11 MB)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # scale pixels from 0-255 to 0-1
print("Training images:", x_train.shape, " Test images:", x_test.shape)

# 2. Model: a stack of layers
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),                        # 28x28 grid -> 784 numbers
    tf.keras.layers.Dense(128, activation="relu"),    # hidden layer, learns patterns
    tf.keras.layers.Dropout(0.2),                     # reduces overfitting
    tf.keras.layers.Dense(10, activation="softmax"),  # one probability per digit
])

# 3. Compile: choose how it learns and what to measure
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# 4. Train
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 5. Evaluate on images the model has never seen
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.2%}")

# 6. Predict one image
probs = model.predict(x_test[:1], verbose=0)[0]
print("Predicted digit:", probs.argmax(), " Real digit:", y_test[0])
