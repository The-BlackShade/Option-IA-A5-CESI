# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torchvision.transforms.v2 as v2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

import os

from sklearn.model_selection import train_test_split

import time

from PIL import Image

# Importing the datasets
dataset_dir = "Datasets"

images = []
labels = []

for label_name in os.listdir(dataset_dir):

    # Ignore resized directories
    if label_name.endswith("_resized"):
        continue

    label_dir = os.path.join(dataset_dir, label_name)

    if not os.path.isdir(label_dir):
        continue

    # Define the label from the ORIGINAL directory name
    if label_name == "Photo":
        label = 1
    else:
        label = 0

    # Load corresponding resized directory
    resized_dir = label_dir + "_resized"

    if not os.path.isdir(resized_dir):
        continue

    for image_name in os.listdir(resized_dir):

        image_path = os.path.join(
            resized_dir,
            image_name
        )

        images.append(image_path)
        labels.append(label)

print("Total:", len(images))
print("Class 0:", labels.count(0))
print("Class 1:", labels.count(1))

# Splitting the dataset into training and testing sets
train_images, temp_images, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images,
    temp_labels,
    test_size=1/3,
    random_state=42,
    stratify=temp_labels
)

print(f"Number of training images: {len(train_images)}")
print(f"Number of testing images: {len(test_images)}")

print(f"Number of training labels: {len(train_labels)}")
print(f"Number of testing labels: {len(test_labels)}")

# Creating the model

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            64 * 32 * 32,
            128
        )

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(
            self.relu(self.conv1(x))
        )

        x = self.pool(
            self.relu(self.conv2(x))
        )

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))

        x = self.dropout(x)


        # No sigmoid here
        x = self.fc2(x)

        return x

train_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(10),
    v2.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
])

# Creating the dataset class
class ImageDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("L")

        image = torch.from_numpy(
            np.array(image)
        ).unsqueeze(0).float() / 255.0

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        return image, label

# Creating the training and testing datasets
train_dataset = ImageDataset(train_images, train_labels)
val_dataset = ImageDataset(val_images, val_labels)
test_dataset = ImageDataset(test_images, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

images_batch, labels_batch = next(iter(train_loader))

print(images_batch.shape)
print(labels_batch.shape)

# training the model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BinaryClassifier().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

num_epochs = 30

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

patience = 5
min_delta = 0.001

best_val_loss = float("inf")
epochs_without_improvement = 0

best_model_path = "best_model.pth"

for epoch in range(num_epochs):

    start_time = time.time()

    print(f"--- Epoch {epoch+1}/{num_epochs} ---")

    # -------------------------
    # Training
    # -------------------------
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    nb_batches = len(train_loader)
    current_batch = 0

    for batch_idx, (images_batch, labels_batch) in enumerate(train_loader, 1):

        print(
            f"Batch {batch_idx}/{len(train_loader)}"
        )

        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images_batch).squeeze(1)

        loss = criterion(outputs, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images_batch.size(0)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).float()

        train_correct += (
            predictions == labels_batch
        ).sum().item()

        train_total += labels_batch.size(0)

    end_time = time.time()

    print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds.")

    train_loss /= train_total
    train_accuracy = train_correct / train_total

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images_batch, labels_batch in val_loader:

            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images_batch).squeeze(1)

            loss = criterion(outputs, labels_batch)

            val_loss += loss.item() * images_batch.size(0)

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= 0.5
            ).float()

            val_correct += (
                predictions == labels_batch
            ).sum().item()

            val_total += labels_batch.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # -------------------------
    # Early stopping
    # -------------------------

    if val_loss < best_val_loss - min_delta:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"Validation loss improved. "
            f"Saving model to {best_model_path}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No validation improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

        if epochs_without_improvement >= patience:

            print(
                f"Early stopping triggered after "
                f"{epoch + 1} epochs."
            )

            break

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))

plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

all_predictions = []
all_labels = []

print("Loading best model...")

model.load_state_dict(
    torch.load(best_model_path, weights_only=True)
)

model.eval()

model.eval()

test_loss = 0.0
test_correct = 0
test_total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images_batch, labels_batch in test_loader:

        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(images_batch).squeeze(1)

        loss = criterion(outputs, labels_batch)

        test_loss += loss.item() * images_batch.size(0)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).float()

        test_correct += (
            predictions == labels_batch
        ).sum().item()

        test_total += labels_batch.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels_batch.cpu().numpy()
        )

test_loss /= test_total
test_accuracy = test_correct / test_total

print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")

print(confusion_matrix(all_labels, all_predictions))

print(
    classification_report(
        all_labels,
        all_predictions
    )
)