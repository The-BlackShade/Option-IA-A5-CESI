#Paintings_02662
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split


"""
Nombre total d'images : 41398

Dimensions :
Largeur :
  min    : 1
  moyenne: 742.4586211894294
  médiane: 612.0
  max    : 10200

Hauteur :
  min    : 1
  moyenne: 725.4116382433934
  médiane: 640.0
  max    : 9894
"""


print("finished loading libraries")
dataset_dir = "Datasets"

images = []
labels = []


i = 0

for label_name in os.listdir(dataset_dir):
    
    label_dir = os.path.join(dataset_dir, label_name)

    if not os.path.isdir(label_dir):
        continue

    if label_name == "Photo":
        label = 1
    else:
        label = 0

    for filename in os.listdir(label_dir):
        filepath = os.path.join(label_dir, filename)

        if os.path.isfile(filepath):
            if i % 100 == 0:
                print(i, "image_num")
            i += 1
            processed_dir = label_dir + "_resized"
            processed_label_dir = os.path.join(
                processed_dir,
                label_name
            )

            os.makedirs(processed_label_dir, exist_ok=True)

            try:  

                with Image.open(filepath) as img:
                    img = img.convert("RGB")
                    
                    image = tf.convert_to_tensor(img)

                    image = tf.image.resize_with_pad(
                        image,
                        target_height=256,
                        target_width=256
                    )

                    image = tf.cast(image, tf.uint8)

                    # Sauvegarde
                    image = tf.io.encode_jpeg(image, quality=95)

                    output_path = os.path.join(
                        processed_label_dir,
                        filename
                    )

                    tf.io.write_file(output_path, image)

                images.append(filepath)
                labels.append(label)

            except Exception as e:
                print(f"Erreur avec {filepath} : {e}")

print("Nombre total d'images :", len(images))
print("Exemple :", images[0], labels[0])




