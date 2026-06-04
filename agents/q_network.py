import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
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
        shared = self.shared(x)

        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)

        q = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q