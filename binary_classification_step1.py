import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
print("Device :", device)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.ToTensor()


# ============================================================
# DATASET
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# CNN
# ============================================================

class PhotoClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 3 x 128 x 128
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 32 x 64 x 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 64 x 32 x 32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # 128 x 16 x 16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



# ============================================================
# MAIN
# ============================================================

def main():

    dataset_dir = "Datasets_separated_step1_256"  # Should contain two folders: "Photo" and "Paintings"

    images = []
    labels = []

    # ========================================================
    # CHARGEMENT DES CHEMINS
    # ========================================================

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

                if i % 10000 == 0:
                    print(i, "images")

                images.append(filepath)
                labels.append(label)

                i += 1

    print("Nombre total d'images :", len(images))

    # ========================================================
    # TRAIN / TEST
    # ========================================================

    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print("Train :", len(train_images))
    print("Test  :", len(test_images))

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = ImageDataset(
        train_images,
        train_labels,
        transform=transform
    )

    test_dataset = ImageDataset(
        test_images,
        test_labels,
        transform=transform
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # Vérification
    images_batch, labels_batch = next(iter(train_loader))

    print("Batch :", images_batch.shape)
    print("Labels :", labels_batch.shape)

    # ========================================================
    # MODEL
    # ========================================================

    model = PhotoClassifier().to(device)

    print(model)

    # ========================================================
    # LOSS
    # ========================================================

    n_positive = sum(train_labels)
    n_negative = len(train_labels) - n_positive

    pos_weight = torch.tensor(
        [n_negative / n_positive],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    print("Photos :", n_positive)
    print("Non-photos :", n_negative)
    print("pos_weight :", pos_weight.item())

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1
    )

    # ========================================================
    # MIXED PRECISION
    # ========================================================

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ========================================================
    # HISTORIQUES
    # ========================================================

    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    # ========================================================
    # TRAINING
    # ========================================================

    # ============================================================
    # EARLY STOPPING
    # ============================================================

    PATIENCE = 6

    best_test_loss = float("inf")
    epochs_without_improvement = 0
    best_model_path = "photo_classifier_best.pth"

    for epoch in range(EPOCHS):

        start_epoch = time.perf_counter()

        print()
        print("=" * 60)
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print("=" * 60)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = torch.zeros(1, device=device)
        train_correct = torch.zeros(1, device=device)
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.float().unsqueeze(1).to(
                device,
                non_blocking=True
            )

            # Très légèrement plus rapide
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            # Backpropagation
            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            # Statistiques GPU
            train_loss += loss.detach() * images.size(0)

            predictions = outputs >= 0

            train_correct += (
                predictions == labels.bool()
            ).sum()

            train_total += images.size(0)

            # Progression
            if (batch_idx + 1) % 20 == 0:

                percentage = (
                    (batch_idx + 1)
                    / len(train_loader)
                    * 100
                )

                print(
                    f"Progression : {percentage:.1f}%"
                )

        # ----------------------------------------------------
        # TRAIN METRICS
        # ----------------------------------------------------

        train_loss = (
            train_loss / train_total
        ).item()

        train_accuracy = (
            train_correct / train_total
        ).item()

        print()
        print(
            f"Train Loss     : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy : {train_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        model.eval()

        test_loss = torch.zeros(1, device=device)
        test_correct = torch.zeros(1, device=device)
        test_total = 0

        y_true = []
        y_prob = []
        y_pred = []

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.float().unsqueeze(1).to(
                    device,
                    non_blocking=True
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(device.type == "cuda")
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

                probabilities = torch.sigmoid(outputs)

                predictions = probabilities >= 0.5

                test_loss += (
                    loss.detach()
                    * images.size(0)
                )

                test_correct += (
                    predictions == labels.bool()
                ).sum()

                test_total += images.size(0)

                # CPU uniquement pour les métriques finales
                y_true.extend(
                    labels.cpu().numpy().ravel()
                )

                y_prob.extend(
                    probabilities.cpu().numpy().ravel()
                )

                y_pred.extend(
                    predictions.cpu().numpy().ravel()
                )

        # ----------------------------------------------------
        # TEST METRICS
        # ----------------------------------------------------

        test_loss = (
            test_loss / test_total
        ).item()

        test_accuracy = (
            test_correct / test_total
        ).item()

        scheduler.step(test_loss)

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = np.array(y_pred)

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        # ----------------------------------------------------
        # HISTORIQUES
        # ----------------------------------------------------

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # ----------------------------------------------------
        # AFFICHAGE
        # ----------------------------------------------------

        print()
        print(
            f"Test Loss      : {test_loss:.4f}"
        )

        print(
            f"Test Accuracy  : {test_accuracy:.4f}"
        )

        print(
            f"Precision      : {precision:.4f}"
        )

        print(
            f"Recall         : {recall:.4f}"
        )

        print(
            f"F1-score       : {f1:.4f}"
        )

        end_epoch = time.perf_counter()

        print()
        print(
            f"Temps epoch : "
            f"{end_epoch - start_epoch:.2f} secondes"
        )

        # ========================================================
        # EARLY STOPPING
        # ========================================================

        if test_loss < best_test_loss:

            best_test_loss = test_loss
            epochs_without_improvement = 0

            # Sauvegarde du meilleur modèle
            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                f"Nouvelle meilleure test loss : "
                f"{best_test_loss:.4f}"
            )

            best_test_accuracy = test_accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1

        else:

            epochs_without_improvement += 1

            print(
                f"Pas d'amélioration depuis "
                f"{epochs_without_improvement} epoch(s)"
            )

            if epochs_without_improvement >= PATIENCE:

                print()
                print("=" * 60)
                print("EARLY STOPPING")
                print(
                    f"Arrêt après {epoch + 1} epochs."
                )
                print(
                    f"Meilleure test loss : "
                    f"{best_test_loss:.4f}"
                )
                print("=" * 60)

                break
            
    print("best test loss:", best_test_loss)

    print(
        f"Test Accuracy  : {best_test_accuracy:.4f}"
    )

    print(
        f"Precision      : {best_precision:.4f}"
    )

    print(
        f"Recall         : {best_recall:.4f}"
    )

    print(
        f"F1-score       : {best_f1:.4f}"
    )
    # ========================================================
    # SAUVEGARDE
    # ========================================================

    torch.save(
        model.state_dict(),
        "photo_classifier.pth"
    )

    print()
    print("Modèle sauvegardé.")

    # ========================================================
    # GRAPHIQUES
    # ========================================================

    epochs = range(
        1,
        len(train_losses) + 1
    )

    # Loss
    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Training loss"
    )

    plt.plot(
        epochs,
        test_losses,
        label="Test loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and test loss")
    plt.legend()
    plt.grid()

    plt.show()

    # Accuracy
    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        train_accuracies,
        label="Training accuracy"
    )

    plt.plot(
        epochs,
        test_accuracies,
        label="Test accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and test accuracy")
    plt.legend()
    plt.grid()

    plt.show()


# ============================================================
# WINDOWS
# ============================================================

if __name__ == "__main__":
    main()