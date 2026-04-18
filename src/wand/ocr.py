"""Symbol recognition CNN and circle detector.

Small CNN (~30K params) that recognizes 8 base magic symbols
from 64x64 grayscale images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .synth_data import NUM_SYMBOLS, SYMBOL_NAMES


class SymbolCNN(nn.Module):
    """Small CNN for symbol classification.

    Input: (batch, 1, 64, 64) grayscale images
    Output: (batch, NUM_SYMBOLS) logits
    """

    def __init__(self, n_classes: int = NUM_SYMBOLS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),   # 64x64 → 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → 32x32

            nn.Conv2d(16, 32, 3, padding=1),   # → 32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → 16x16

            nn.Conv2d(32, 32, 3, padding=1),   # → 16x16
            nn.ReLU(),
            nn.MaxPool2d(2),                   # → 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def predict(self, img: np.ndarray) -> tuple[str, float]:
        """Predict symbol from a single 64x64 image."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            logits = self(x)
            probs = F.softmax(logits, dim=-1)[0]
            idx = probs.argmax().item()
            return SYMBOL_NAMES[idx], probs[idx].item()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CircleDetector(nn.Module):
    """Binary classifier: closed circle vs open/no circle.

    Input: (batch, 1, 64, 64)
    Output: (batch, 1) probability of closed circle
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(self.features(x)))

    def is_closed(self, img: np.ndarray, threshold: float = 0.5) -> tuple[bool, float]:
        """Check if image contains a closed circle."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            prob = self(x)[0, 0].item()
            return prob > threshold, prob


def train_symbol_cnn(images: np.ndarray, labels: np.ndarray,
                     epochs: int = 20, batch_size: int = 64,
                     lr: float = 1e-3, val_split: float = 0.15) -> tuple[SymbolCNN, dict]:
    """Train symbol CNN on synthetic data. Returns (model, metrics)."""
    n = len(images)
    perm = np.random.permutation(n)
    val_n = int(n * val_split)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    x_train = torch.from_numpy(images[train_idx]).float().unsqueeze(1)
    y_train = torch.from_numpy(labels[train_idx]).long()
    x_val = torch.from_numpy(images[val_idx]).float().unsqueeze(1)
    y_val = torch.from_numpy(labels[val_idx]).long()

    model = SymbolCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(x_train[idx])
            loss = loss_fn(logits, y_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item()

        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

    return model, history


def train_circle_detector(n_samples: int = 800, epochs: int = 15,
                          batch_size: int = 32, lr: float = 1e-3) -> tuple[CircleDetector, dict]:
    """Train circle detector on synthetic data."""
    from .synth_data import generate_circle

    images = []
    labels = []
    for _ in range(n_samples // 2):
        images.append(generate_circle(closed=True))
        labels.append(1.0)
        images.append(generate_circle(closed=False))
        labels.append(0.0)

    images = np.array(images)
    labels = np.array(labels, dtype=np.float32)

    n = len(images)
    perm = np.random.permutation(n)
    val_n = int(n * 0.15)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    x_train = torch.from_numpy(images[train_idx]).float().unsqueeze(1)
    y_train = torch.from_numpy(labels[train_idx]).float().unsqueeze(1)
    x_val = torch.from_numpy(images[val_idx]).float().unsqueeze(1)
    y_val = torch.from_numpy(labels[val_idx]).float().unsqueeze(1)

    model = CircleDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        perm_t = torch.randperm(len(x_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(x_train), batch_size):
            idx = perm_t[i:i + batch_size]
            pred = model(x_train[idx])
            loss = loss_fn(pred, y_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_acc = ((val_pred > 0.5).float() == y_val).float().mean().item()

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_acc"].append(val_acc)

    return model, history
