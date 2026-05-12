"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None 
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._lda: LDA | None = None
        self._threshold: float = 0.5 
        self._use_pca: bool = True
        self._use_lda: bool = False

    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        self._net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
        )

    def _apply_dimensionality_reduction(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Apply PCA and optionally LDA for dimensionality reduction.
        
        Args:
            X: Feature matrix of shape (n_samples, feature_dim)
            y: Labels for LDA (required if use_lda=True)
            
        Returns:
            Reduced feature matrix
        """
        if self._pca is None:
            self._pca = PCA(n_components=0.95, random_state=42)
            self._pca.fit(X)
        
        X_reduced = self._pca.transform(X)
        
        if self._use_lda and y is not None and len(np.unique(y)) > 1:
            n_classes = len(np.unique(y))
            max_components = min(X_reduced.shape[1], n_classes - 1)
            
            if max_components >= 1:
                if self._lda is None:
                    self._lda = LDA(n_components=max_components)
                    self._lda.fit(X_reduced, y)
                X_reduced = self._lda.transform(X_reduced)
        
        return X_reduced

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """

        X_reduced = self._apply_dimensionality_reduction(X, y)
        
        X_scaled = self._scaler.fit_transform(X_reduced)
        
        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20
        )
        
        n_samples = len(X_t)
        n_val = max(1, int(0.15 * n_samples))
        indices = np.random.permutation(n_samples)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, X_val = X_t[train_idx], X_t[val_idx]
        y_train, y_val = y_t[train_idx], y_t[val_idx]
        
        self.train()
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 30
        best_state_dict = None
        
        for epoch in range(500):
            optimizer.zero_grad()
            logits = self(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_logits = self(X_val)
                val_loss = criterion(val_logits, y_val)

            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    self.load_state_dict(best_state_dict)
                    break
            
            self.train()
        
        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if self._pca is not None:
            X_reduced = self._pca.transform(X)
            if self._lda is not None:
                X_reduced = self._lda.transform(X_reduced)
        else:
            X_reduced = X
        
        X_scaled = self._scaler.transform(X_reduced)
        
        X_t = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)