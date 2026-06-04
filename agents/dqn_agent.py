import torch
import numpy as np
import random
from agents.q_network import QNetwork

class DQNAgent:
    
    def __init__(self, input_size, num_actions, epsilon_scheduler):
        
        self.online_net = QNetwork(
            input_size, 
            num_actions
        )   

        self.target_net = QNetwork(
            input_size,
            num_actions,
        )
        self.num_actions = num_actions
        self.epsilon_scheduler = epsilon_scheduler

        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state, step, training):

        epsilon = self.epsilon_scheduler.get_epsilon(step)

        if random.random() < epsilon and training:
            action = random.randint(0, self.num_actions - 1)

            return action

        device = next(self.online_net.parameters()).device
        
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_net(state_t)

            base_action = torch.argmax(q_values, dim=1).item()
            
            noise = random.choice([-2, -1, 1, 2])
            action = np.clip(base_action + noise, 0, self.num_actions - 1)

        return action
    
    def update_target_network(self):

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.target_net.eval()