#Paintings_02662
from torchvision.transforms import v2
import os
from PIL import Image




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

widths = []
heights = []

IMG_SIZE = 128

i = 0
# Parcours des dossiers

for label_name in os.listdir(dataset_dir):
    
    label_dir = os.path.join(dataset_dir, label_name)

    if not os.path.isdir(label_dir):
        continue

    # On transforme le nom du dossier en label numérique
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
            
            os.makedirs(processed_dir, exist_ok=True)            
            
            try:  
                with Image.open(filepath) as img:
                    transform = v2.Compose([
                        v2.Grayscale(num_output_channels=1),
                    ])

                    # RGB Conversion
                    img = img.convert("RGB")

                    img.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

                    padded_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

                    x = (IMG_SIZE - img.width) // 2
                    y = (IMG_SIZE - img.height) // 2

                    padded_img.paste(img, (x, y))

                    # Applies the transformations to the image
                    image = transform(padded_img)

                    output_path = os.path.join(
                        processed_dir,
                        filename
                    )

                    image.save(output_path, format="JPEG", quality=95)

                images.append(output_path)
                labels.append(label)

                #widths.append(width)
                #heights.append(height)
            except Exception as e:
                print(f"Erreur avec {filepath} : {e}")

print("Nombre total d'images :", len(images))
print("Exemple :", images[0], labels[0])
"""
print("Largeur :")
print("  min    :", np.min(widths))
print("  moyenne:", np.mean(widths))
print("  médiane:", np.median(widths))
print("  max    :", np.max(widths))

print("\nHauteur :")
print("  min    :", np.min(heights))
print("  moyenne:", np.mean(heights))
print("  médiane:", np.median(heights))
print("  max    :", np.max(heights))"""



# train_images, test_images, train_labels, test_labels = train_test_split(
#     images,
#     labels,
#     test_size=0.2,
#     random_state=42,
#     stratify=labels
# )



