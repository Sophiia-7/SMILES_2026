"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    layer_indices = [8, 16, -1]

    layer_indices = [i if i >= 0 else n_layers + i for i in layer_indices]
    
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze()
    
    if real_positions.dim() == 0:
        real_positions = real_positions.unsqueeze(0)
    
    aggregated_features = []
    
    for layer_idx in layer_indices:
        layer_hidden = hidden_states[layer_idx] 
        
        token_reps = layer_hidden[real_positions] 
        pooled = token_reps.mean(dim=0) 
        
        aggregated_features.append(pooled)
    
    feature = torch.cat(aggregated_features, dim=0)  
    
    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze()
    if real_positions.dim() == 0:
        real_positions = real_positions.unsqueeze(0)
    
    layer_indices = [8, 16, -1]
    layer_indices = [i if i >= 0 else n_layers + i for i in layer_indices]
    
    geometric_features = []
    
    for layer_idx in layer_indices:
        layer_hidden = hidden_states[layer_idx]
        token_reps = layer_hidden[real_positions]  
        
        if len(real_positions) > 1:
            token_std = token_reps.std(dim=0).mean()
            geometric_features.append(token_std)

            norms = torch.norm(token_reps, dim=1)
            probs = F.softmax(norms, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            geometric_features.append(entropy)

            max_pool = token_reps.max(dim=0)[0] 
            mean_pool = token_reps.mean(dim=0) 
            max_minus_mean = torch.norm(max_pool - mean_pool)
            geometric_features.append(max_minus_mean)
        
        mean_norm = torch.norm(token_reps.mean(dim=0))
        geometric_features.append(mean_norm)
        
        if len(real_positions) > 1 and mean_norm > 1e-6:
            cv = token_std / mean_norm
            geometric_features.append(cv)
    
    response_length = torch.tensor(len(real_positions), dtype=torch.float32) / seq_len
    geometric_features.append(response_length)
    
    if len(layer_indices) >= 2:
        for i in range(len(layer_indices) - 1):
            layer_a = hidden_states[layer_indices[i]][real_positions].mean(dim=0)
            layer_b = hidden_states[layer_indices[i+1]][real_positions].mean(dim=0)
            cos_sim = F.cosine_similarity(layer_a.unsqueeze(0), layer_b.unsqueeze(0))
            geometric_features.append(cos_sim)
    
    if geometric_features:
        return torch.stack(geometric_features)
    else:
        return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask) 

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features