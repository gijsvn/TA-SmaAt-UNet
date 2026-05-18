import torch
from torch import nn

class TimeConditioning(nn.Module):
    """
    FiLM-style conditioning: given a global conditioning vector t (B, time_dim),
    compute per-channel scale and shift for a feature map x (B, C, H, W).

    y = (1 + gamma) * x + beta
    """
    def __init__(
        self,
        num_channels: int,
        time_dim: int=4,
        hidden_dim: int=16,
    ) -> None:
        super().__init__()

        self.time_dim = time_dim
        self.num_channels = num_channels

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_channels),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            t: (B, time_dim)
        Returns:
            (B, C, H, W) conditioned on t
        """
        # t -> (B, 2C)
        params = self.mlp(t)
        gamma, beta = params.chunk(2, dim=1)  # each (B, C)

        # reshape to broadcast over H, W
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)

        # (1 + gamma) so that initial weights near zero ≈ identity
        return x * (1.0 + gamma) + beta