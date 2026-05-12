from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()

        self._pca: PCA | None = None

        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )

        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)

        n_components = min(
            32,
            X_scaled.shape[0] - 1,
            X_scaled.shape[1],
        )

        self._pca = PCA(n_components=n_components)

        X_processed = self._pca.fit_transform(X_scaled)

        self._build_network(X_processed.shape[1])

        X_t = torch.from_numpy(X_processed).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)],
            dtype=torch.float32,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=300,
        )

        best_loss = float("inf")
        patience = 30
        patience_counter = 0

        self.train()

        for _ in range(80):
            optimizer.zero_grad()

            logits = self(X_t)

            loss = criterion(logits, y_t)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            scheduler.step()

            current_loss = float(loss.item())

            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        self.eval()

        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":

        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(
            np.concatenate(
                [
                    probs,
                    np.linspace(0.05, 0.95, 181),
                ]
            )
        )

        best_threshold = 0.5
        best_f1 = -1.0

        for t in candidates:
            y_pred_t = (probs >= t).astype(int)

            score = f1_score(
                y_val,
                y_pred_t,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (
            self.predict_proba(X)[:, 1] >= self._threshold
        ).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)

        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)

        X_t = torch.from_numpy(X_scaled).float()

        with torch.no_grad():
            logits = self(X_t)

            prob_pos = torch.sigmoid(logits).numpy()

        return np.stack(
            [
                1.0 - prob_pos,
                prob_pos,
            ],
            axis=1,
        )