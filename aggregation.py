from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    n_layers, seq_len, hidden_dim = hidden_states.shape

    layer_indices = [8, 16, -1]
    
    layer_indices_converted = []
    for i in layer_indices:
        if i >= 0:
            layer_indices_converted.append(i)
        else:
            layer_indices_converted.append(n_layers + i)
    
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze()
    
    if real_positions.dim() == 0:
        real_positions = real_positions.unsqueeze(0)
    
    aggregated_features = []
    
    for layer_idx in layer_indices_converted:
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
    """Extract fixed-size geometric features."""
    n_layers, seq_len, hidden_dim = hidden_states.shape
    
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze()
    if real_positions.dim() == 0:
        real_positions = real_positions.unsqueeze(0)
    
    layer_indices = [8, 16, -1]
    layer_indices_converted = []
    for i in layer_indices:
        if i >= 0:
            layer_indices_converted.append(i)
        else:
            layer_indices_converted.append(n_layers + i)
    
    geometric_features = []
    
    for layer_idx in layer_indices_converted:
        layer_hidden = hidden_states[layer_idx]
        token_reps = layer_hidden[real_positions]  
        
        if len(real_positions) > 1:
            token_std = token_reps.std(dim=0).mean()
        else:
            token_std = torch.tensor(0.0, device=hidden_states.device)
        geometric_features.append(token_std.unsqueeze(0))
        
        if len(real_positions) > 1:
            norms = torch.norm(token_reps, dim=1)
            probs = F.softmax(norms, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        else:
            entropy = torch.tensor(0.0, device=hidden_states.device)
        geometric_features.append(entropy.unsqueeze(0))
        
        if len(real_positions) > 1:
            max_pool = token_reps.max(dim=0)[0]
            mean_pool = token_reps.mean(dim=0)
            max_minus_mean = torch.norm(max_pool - mean_pool)
        else:
            max_minus_mean = torch.tensor(0.0, device=hidden_states.device)
        geometric_features.append(max_minus_mean.unsqueeze(0))

        mean_norm = torch.norm(token_reps.mean(dim=0))
        geometric_features.append(mean_norm.unsqueeze(0))

        if len(real_positions) > 1 and mean_norm > 1e-6:
            cv = token_std / mean_norm
        else:
            cv = torch.tensor(0.0, device=hidden_states.device)
        geometric_features.append(cv.unsqueeze(0))
    
    response_length = torch.tensor(len(real_positions), dtype=torch.float32, device=hidden_states.device) / seq_len
    geometric_features.append(response_length.unsqueeze(0))
    
    if len(layer_indices_converted) >= 2:
        for i in range(len(layer_indices_converted) - 1):
            layer_a = hidden_states[layer_indices_converted[i]][real_positions].mean(dim=0)
            layer_b = hidden_states[layer_indices_converted[i+1]][real_positions].mean(dim=0)
            cos_sim = F.cosine_similarity(layer_a.unsqueeze(0), layer_b.unsqueeze(0))
            geometric_features.append(cos_sim.unsqueeze(0))
    else:
        geometric_features.append(torch.tensor([0.0], device=hidden_states.device))
    
    if geometric_features:
        result = torch.cat(geometric_features, dim=0)
        return result
    else:
        return torch.zeros(0, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = True,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    
    geo_features = extract_geometric_features(hidden_states, attention_mask)
    
    if not hasattr(aggregation_and_feature_extraction, '_printed'):
        print(f"✓ Geometric features: {geo_features.shape[0]} features added")
        print(f"  Total feature dimension: {agg_features.shape[0]} + {geo_features.shape[0]} = {agg_features.shape[0] + geo_features.shape[0]}")
        aggregation_and_feature_extraction._printed = True
    
    if geo_features.numel() > 0:
        return torch.cat([agg_features, geo_features], dim=0)
    
    return agg_features