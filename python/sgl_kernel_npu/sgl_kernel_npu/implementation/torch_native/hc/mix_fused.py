import torch


def mix_fused(candidates, combined_weight, variance_epsilon, output_dtype):
    values = candidates.float()
    normalized = values * torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True) + variance_epsilon
    )
    scores = (normalized * combined_weight.float()).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    return (probabilities.unsqueeze(-1) * values).sum(dim=1).to(output_dtype)
