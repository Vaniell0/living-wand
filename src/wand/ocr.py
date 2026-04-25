"""Symbol recognition and circle detector.

SymbolNet: CoordConv + SE attention + multi-scale features for rune recognition.
AdaptiveOCR: wrapper that learns from SpellFormer feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from .synth_data import NUM_SYMBOLS, SYMBOL_NAMES


class SEBlock(nn.Module):
    """Squeeze-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excite(w).view(b, c, 1, 1)
        return x * w


class SymbolNet(nn.Module):
    """Smart symbol classifier with CoordConv + SE attention + multi-scale features.

    Key features:
    1. CoordConv: x,y coordinate channels → position/size awareness
    2. Squeeze-Excitation: channel attention per conv block
    3. Multi-scale: skip connection from 16x16 features → fine detail
    4. BatchNorm: stable training, better generalization

    Input: (batch, 1, 64, 64) grayscale
    Output: (batch, n_classes) logits
    """

    def __init__(self, n_classes: int = NUM_SYMBOLS):
        super().__init__()
        self.n_classes = n_classes

        # Block 1: CoordConv (3 input channels: image + x + y)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        self.se1 = SEBlock(24)
        self.pool1 = nn.MaxPool2d(2)  # 64→32

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.se2 = SEBlock(48)
        self.pool2 = nn.MaxPool2d(2)  # 32→16

        # Block 3: dilated conv for larger receptive field (stays 16x16)
        self.block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, dilation=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.se4 = SEBlock(64)
        self.pool3 = nn.MaxPool2d(2)  # 16→8

        # Multi-scale skip: downsample 16x16 features to 8x8
        self.skip_pool = nn.AdaptiveAvgPool2d(8)

        # Classifier: (48 skip + 64 main) * 8 * 8 = 7168
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((48 + 64) * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

        # Pre-compute coordinate grids (registered as buffer)
        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, 64),
            torch.linspace(0, 1, 64),
            indexing='ij',
        ))  # (2, 64, 64)
        self.register_buffer('coord_grid', coords)

    def _add_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Add x,y coordinate channels to input."""
        b = x.shape[0]
        coords = self.coord_grid.unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([x, coords], dim=1)  # (B, 3, 64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CoordConv: add coordinate channels
        x = self._add_coords(x)  # (B, 3, 64, 64)

        x = self.block1(x)
        x = self.se1(x)
        x = self.pool1(x)    # (B, 24, 32, 32)

        x = self.block2(x)
        x = self.se2(x)
        x = self.pool2(x)    # (B, 48, 16, 16)

        # Save 16x16 features for skip connection
        skip = self.block3(x)  # (B, 48, 16, 16) — dilated, larger RF

        x = self.block4(skip)
        x = self.se4(x)
        x = self.pool3(x)     # (B, 64, 8, 8)

        # Multi-scale fusion
        skip_down = self.skip_pool(skip)  # (B, 48, 8, 8)
        x = torch.cat([x, skip_down], dim=1)  # (B, 112, 8, 8)

        return self.classifier(x)

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


def train_symbol_net(images: np.ndarray, labels: np.ndarray,
                     epochs: int = 30, batch_size: int = 64,
                     lr: float = 1e-3, val_split: float = 0.15) -> tuple[SymbolNet, dict]:
    """Train SymbolNet on synthetic data."""
    n = len(images)
    perm = np.random.permutation(n)
    val_n = int(n * val_split)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    x_train = torch.from_numpy(images[train_idx]).float().unsqueeze(1)
    y_train = torch.from_numpy(labels[train_idx]).long()
    x_val = torch.from_numpy(images[val_idx]).float().unsqueeze(1)
    y_val = torch.from_numpy(labels[val_idx]).long()

    model = SymbolNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        perm_t = torch.randperm(len(x_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(x_train), batch_size):
            idx = perm_t[i:i + batch_size]
            logits = model(x_train[idx])
            loss = loss_fn(logits, y_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item()

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  OCR epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.4f} acc={val_acc:.1%}")

    return model, history


class AdaptiveOCR:
    """OCR wrapper that learns from SpellFormer feedback.

    After each cast, SpellFormer produces an effect. If the effect strongly
    suggests certain runes (high element/domain intensity), but OCR predicted
    something different, the correction is stored and OCR fine-tunes.

    The base CNN features stay frozen. Only a small adaptation layer trains.
    """

    def __init__(self, base_cnn: SymbolNet, lr: float = 1e-3, buffer_size: int = 100):
        self.base = base_cnn
        self.buffer: deque[tuple[np.ndarray, int]] = deque(maxlen=buffer_size)

        # Freeze everything except classifier
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.base.classifier.parameters():
            param.requires_grad = True

        # Optimizer for classifier only
        self.optimizer = torch.optim.Adam(
            self.base.classifier.parameters(), lr=lr
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.corrections = 0

    def predict(self, img: np.ndarray) -> tuple[str, float]:
        """Predict with base CNN."""
        return self.base.predict(img)

    def predict_top_k(self, img: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k predictions."""
        self.base.eval()
        with torch.no_grad():
            x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            logits = self.base(x)
            probs = F.softmax(logits, dim=-1)[0]
            top_k = torch.topk(probs, k)
            return [(SYMBOL_NAMES[idx.item()], prob.item())
                    for idx, prob in zip(top_k.indices, top_k.values)]

    def correct(self, img: np.ndarray, correct_label: int):
        """Record a correction and fine-tune.

        Called when SpellFormer or the player indicates the OCR prediction
        was wrong for this image.
        """
        self.buffer.append((img.copy(), correct_label))
        self.corrections += 1
        self._fine_tune()

    def feedback_from_spell(self, images: list[np.ndarray],
                            ocr_predictions: list[str],
                            spell_context: dict):
        """Receive feedback from SpellFormer about what runes were likely intended.

        Args:
            images: the drawn symbol images
            ocr_predictions: what OCR predicted for each
            spell_context: dict with 'element_intensity' (6,) and 'domain_intensity' (6,)
                          from SpellFormer output — tells us which runes the magic
                          model thinks are present
        """
        elem_int = spell_context.get("element_intensity")
        dom_int = spell_context.get("domain_intensity")
        if elem_int is None or dom_int is None:
            return

        # Map strong intensities to likely rune names
        ELEM_NAMES = ["fire", "water", "air", "earth", "light", "dark"]  # indices 0-5
        DOM_NAMES = ["body", "mind", "matter", "space", "time", "life"]  # indices 6-11 in SYMBOL_NAMES

        likely_runes = set()
        for i, intensity in enumerate(elem_int):
            if intensity > 0.5:
                likely_runes.add(ELEM_NAMES[i])
        for i, intensity in enumerate(dom_int):
            if intensity > 0.5:
                likely_runes.add(DOM_NAMES[i])

        if not likely_runes:
            return

        # Check each OCR prediction — if it's NOT in likely_runes but a likely
        # rune IS, and the OCR confidence was low, this might be a correction
        for img, pred_name in zip(images, ocr_predictions):
            if pred_name in likely_runes:
                # OCR agrees with magic — reinforce
                if pred_name in SYMBOL_NAMES:
                    label = SYMBOL_NAMES.index(pred_name)
                    self.buffer.append((img.copy(), label))
            else:
                # OCR disagrees — check top predictions
                top = self.predict_top_k(img, k=5)
                for alt_name, alt_prob in top[1:]:  # skip first (already predicted)
                    if alt_name in likely_runes and alt_prob > 0.1:
                        # Alternative prediction matches magic context — correct
                        label = SYMBOL_NAMES.index(alt_name)
                        self.correct(img, label)
                        break

    def _fine_tune(self, epochs: int = 2):
        """Fine-tune classifier on replay buffer."""
        if len(self.buffer) < 5:
            return

        imgs = np.array([b[0] for b in self.buffer])
        labels = np.array([b[1] for b in self.buffer])

        x = torch.from_numpy(imgs).float().unsqueeze(1)
        y = torch.from_numpy(labels).long()

        self.base.train()
        for _ in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), 16):
                idx = perm[i:i + 16]
                logits = self.base(x[idx])
                loss = self.loss_fn(logits, y[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Re-freeze features (in case something unfroze)
        for param in self.base.features.parameters():
            param.requires_grad = False

    def save_buffer(self, path: str):
        """Save replay buffer to disk."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = [(img, label) for img, label in self.buffer]
        torch.save(data, path)

    def load_buffer(self, path: str):
        """Load replay buffer from disk."""
        import os
        if not os.path.exists(path):
            return
        data = torch.load(path, weights_only=False)
        for img, label in data:
            self.buffer.append((img, label))


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
