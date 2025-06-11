import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MedicalImageCNN(nn.Module):
    def __init__(self, output_dim=128):
        super(MedicalImageCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # [B, 64, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):  # x: [B, 1, H, W]
        x = self.features(x)
        x = self.fc(x)
        return x  # shape: [B, output_dim]


class CNNToRNA(nn.Module):
    """
    CNN to RNA regressor. Takes a batch of images and outputs gene expression values.
    Meant to be used as a wrapper for a CNN encoder
    """
    def __init__(self, cnn_encoder, embedding_dim=128, output_dim=1000):
        super().__init__()
        self.encoder = cnn_encoder
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, images):  # [B, N, 1, H, W]
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)  # [B*N, 1, H, W]

        features = self.encoder(images)      # [B*N, emb_dim]
        features = features.view(B, N, -1)   # [B, N, emb_dim]

        pooled = features.mean(dim=1)        # [B, emb_dim]
        return self.regressor(pooled)        # [B, num_genes]


class CNNClassifier(nn.Module):
    def __init__(self, cnn_encoder, embedding_dim=128, num_classes=3):  # or 2 for binary
        super().__init__()
        self.encoder = cnn_encoder
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, images):  # images: [B, N, 1, H, W]
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)

        features = self.encoder(images)  # [B*N, emb_dim]
        features = features.view(B, N, -1)  # [B, N, emb_dim]
        pooled = features.mean(dim=1)      # [B, emb_dim]

        return self.classifier(pooled)     # [B, num_classes]

from torch.amp import autocast, GradScaler

from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, device="cuda",
                num_epochs=50, patience=5, min_delta=0.001):
    model.to(device)

    train_losses = []
    val_losses = []
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        for images, _, gene_values, _ in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(device)
            gene_values = gene_values.to(device)

            optimizer.zero_grad()

            with autocast("cuda"):  # Mixed precision context
                outputs = model(images)
                loss = criterion(outputs, gene_values)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

            # Optional: free up memory
            del outputs, loss
            torch.cuda.empty_cache()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0

        for images, _, gene_values, _ in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            gene_values = gene_values.to(device)

            with torch.no_grad():
                with autocast("cuda"):
                    outputs = model(images)
                    val_loss = criterion(outputs, gene_values)
                    running_val_loss += val_loss.item()

            del outputs, val_loss
            torch.cuda.empty_cache()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # 🔁 Early Stopping
        if best_val_loss - epoch_val_loss > min_delta:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    return train_losses, val_losses
