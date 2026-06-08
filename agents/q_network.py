import torch
import torch.nn as nn
from configs.env_config import FRAME_STACK


class QNetwork(nn.Module):

    def __init__(self, input_size, num_actions, lidar_len=240, gru_hidden=256):
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions
        self.frame_stack = FRAME_STACK

        assert input_size % max(1, self.frame_stack) == 0
        self.frame_size = input_size // max(1, self.frame_stack)

        self.lidar_len = lidar_len

        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        non_lidar_size = max(0, self.frame_size - self.lidar_len)
        self.non_lidar_proj = nn.Linear(non_lidar_size, 128) if non_lidar_size > 0 else None

        rnn_input_size = 128 + 128 if non_lidar_size > 0 else 128
        self.gru = nn.GRU(input_size=rnn_input_size, hidden_size=gru_hidden, batch_first=True)

        self.shared = nn.Sequential(
            nn.Linear(gru_hidden, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        bsz = x.shape[0]
        if self.frame_stack > 1:
            frames = x.view(bsz, self.frame_stack, self.frame_size)
        else:
            frames = x.unsqueeze(1)

        per_frame_embs = []
        for t in range(frames.shape[1]):
            frame = frames[:, t, :]
            if self.lidar_len > 0:
                lidar = frame[:, -self.lidar_len:]
                lidar = lidar.view(bsz, 1, self.lidar_len)
                le = self.lidar_encoder(lidar).view(bsz, -1)
            else:
                le = torch.zeros(bsz, 128, device=frame.device)

            if self.non_lidar_proj is not None:
                non_lidar = frame[:, : (self.frame_size - self.lidar_len)] if (self.frame_size - self.lidar_len) > 0 else frame[:, :0]
                ne = self.non_lidar_proj(non_lidar)
            else:
                ne = torch.zeros(bsz, 128, device=frame.device)

            per_frame_embs.append(torch.cat([le, ne], dim=1))

        seq = torch.stack(per_frame_embs, dim=1) 

        _, h = self.gru(seq)
        h = h.squeeze(0)

        shared = self.shared(h)

        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)

        q = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q