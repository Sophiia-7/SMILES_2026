from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:

    idx = np.arange(len(y))

    idx_train_full, idx_test = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    y_train_full = y[idx_train_full]

    n_splits = 5

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = []

    for train_rel_idx, val_rel_idx in skf.split(
        idx_train_full,
        y_train_full,
    ):

        idx_train = idx_train_full[train_rel_idx]
        idx_val = idx_train_full[val_rel_idx]

        folds.append(
            (
                idx_train.astype(int),
                idx_val.astype(int),
                idx_test.astype(int),
            )
        )

    return folds