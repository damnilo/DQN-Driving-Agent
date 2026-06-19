import os
import torch

class CheckpointManager:

    def save(self, filepath, agent, optimizer, step, episode):
        """Persists both network state dicts, the optimizer state, global step, and
        episode number to the given file path."""


        os.makedirs(os.path.dirname(filepath) or ".", exist_ok = True)

        torch.save({
            "online_net": agent.online_net.state_dict(),
            "target_net": agent.target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "episode": episode
        }, filepath)

    def load(self, filepath, agent, optimizer):
        """Restores online/target network weights and optimizer state from a checkpoint
        and returns the saved global step."""

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)

        agent.online_net.load_state_dict(
            checkpoint["online_net"]
        )

        agent.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        return checkpoint["step"]