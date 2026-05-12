from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:

    valid_len = int(attention_mask.sum().item())

    hs = hidden_states[:, :valid_len, :]

    selected = [
        hs[-1],
        hs[-2],
        hs[-4],
    ]

    pooled = []

    for layer in selected:
        mean_pool = layer.mean(dim=0)
        last_token = layer[-1]

        pooled.append(mean_pool)
        pooled.append(last_token)

    return torch.cat(pooled, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    valid_len = int(attention_mask.sum().item())

    hs = hidden_states[:, :valid_len, :]

    device = hs.device

    features = []

    for layer in hs:
        token_norms = torch.norm(layer, dim=-1)

        features.extend(
            [
                token_norms.mean(),
                token_norms.std(),
                token_norms.max(),
                token_norms.min(),
            ]
        )

    for i in range(hs.shape[0] - 1):
        a = hs[i].mean(dim=0)
        b = hs[i + 1].mean(dim=0)

        cos_sim = F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=-1,
        ).squeeze()

        drift = torch.norm(a - b)

        features.extend([cos_sim, drift])

    final_layer = hs[-1]

    global_mean = final_layer.mean()
    global_std = final_layer.std()

    features.extend(
        [
            global_mean,
            global_std,
            torch.tensor(float(valid_len), device=device),
        ]
    )

    return torch.stack(features).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = True,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features