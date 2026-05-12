"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs stratified k-fold cross-validation.
    Each fold uses a different held-out test set, with validation taken
    from the training portion.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
                      In k-fold mode, this determines the number of folds
                      (k = 1/test_size).
        val_size:     Fraction of training samples reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Use stratified k-fold cross-validation to maximize use of limited data
        (689 samples). Each fold provides a different train/val/test split,
        and final predictions are ensembled across folds.
    """
    
    N = len(y)
    idx = np.arange(N)

    n_folds = 5
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    splits = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(idx, y)):
        y_train_val = y[train_val_idx]

        relative_val = val_size / (1.0 - test_size)

        if relative_val > 0.3:
            relative_val = 0.2 
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state + fold_idx, 
            stratify=y_train_val,
        )
        
        splits.append((train_idx, val_idx, test_idx))
    
    return splits


def simple_stratified_split(
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Simple single stratified split (original default behavior).
    
    Provided as an alternative to k-fold cross-validation.
    """
    idx = np.arange(len(y))
    
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=y[idx_train_val],
    )
    return [(idx_train, idx_val, idx_test)]


def group_aware_split(
    y: np.ndarray,
    df: pd.DataFrame,
    group_col: str = "prompt",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Group-aware split that ensures no prompt appears in both train and validation/test.
    
    Useful if the same prompt appears multiple times with different responses.
    
    Args:
        y: Labels
        df: DataFrame containing the data
        group_col: Column name to use for grouping (e.g., "prompt")
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
    """
    groups = df[group_col].unique()
    n_groups = len(groups)

    group_labels = []
    group_indices = []
    
    for group in groups:
        group_mask = df[group_col] == group
        group_y = y[group_mask]
        majority_label = 1 if group_y.sum() > len(group_y) / 2 else 0
        group_labels.append(majority_label)
        group_indices.append(np.where(group_mask)[0])

    idx_groups = np.arange(n_groups)

    train_val_groups, test_groups = train_test_split(
        idx_groups,
        test_size=test_size,
        random_state=random_state,
        stratify=group_labels,
    )

    train_labels = [group_labels[i] for i in train_val_groups]
    relative_val = val_size / (1.0 - test_size)
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=relative_val,
        random_state=random_state,
        stratify=train_labels,
    )

    train_idx = np.concatenate([group_indices[i] for i in train_groups])
    val_idx = np.concatenate([group_indices[i] for i in val_groups])
    test_idx = np.concatenate([group_indices[i] for i in test_groups])
    
    return [(train_idx, val_idx, test_idx)]