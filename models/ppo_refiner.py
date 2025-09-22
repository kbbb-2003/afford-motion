# models/ppo_refiner.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPORefiner(nn.Module):
    """
    Simple actor-critic that predicts per-frame corrections (delta) to an input motion sequence.
    Input: obs tensor shape [B, seq_len, motion_dim] (will be flattened inside)
    Output: mean action [B, seq_len, motion_dim], logstd param (state-independent), value [B]
    """
    def __init__(self, seq_len:int, motion_dim:int, hidden_dim:int=1024):
        super().__init__()
        self.seq_len = seq_len
        self.motion_dim = motion_dim
        in_dim = seq_len * motion_dim

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # actor head -> per-frame mean
        self.actor_mean = nn.Linear(hidden_dim, in_dim)
        # state-independent logstd (one param per action dim)
        self.logstd = nn.Parameter(torch.zeros(in_dim))

        # critic head
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs: torch.Tensor):
        """
        obs: [B, seq_len, motion_dim]
        returns: mean [B, seq_len, motion_dim], logstd [B, seq_len, motion_dim], value [B]
        """
        b = obs.shape[0]
        x = obs.view(b, -1)
        h = self.trunk(x)
        mean = self.actor_mean(h).view(b, self.seq_len, self.motion_dim)
        logstd = self.logstd.view(1, self.seq_len, self.motion_dim).expand_as(mean)
        value = self.critic(x).squeeze(-1)
        return mean, logstd, value


