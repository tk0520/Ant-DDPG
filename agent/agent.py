from torch.optim import RMSprop
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
import torch
import copy

from agent.model import DDPGActorModel, DDPGCriticModel
from agent.memory import ReplayMemory
from agent.computation import QLerning
import settings

class RobotArmAgent:
    def __init__(self, state_space, action_space):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_space = state_space
        self.action_space = action_space

        self.actor_model = DDPGActorModel(
            self.state_space, 
            self.action_space, 
            self.device
        ).to(self.device)
        self.old_actor_model = copy.deepcopy(self.actor_model).to(self.device)

        self.critic_model = DDPGCriticModel(
            self.state_space, 
            self.action_space, 
            self.device
        ).to(self.device)
        self.old_critic_model = copy.deepcopy(self.critic_model).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), 
            lr=settings.LEARNING_RATE
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), 
            lr=settings.LEARNING_RATE
        )

        self.replay_memory = ReplayMemory()
        self.computation = QLerning(self)        

        self.env_steps = 0
        self.last_sync_step = 0

    def set_old_actor_model(self):
        self.old_actor_model.load_state_dict(self.actor_model.state_dict())

    def set_old_critic_model(self):
        self.old_critic_model.load_state_dict(self.critic_model.state_dict())
    
    def select_actions(self, observations):
        actions = self.actor_model(observations)
        return actions.detach().cpu().numpy().astype(np.float32)

    def select_old_actions(self, observations):
        actions = self.old_actor_model(observations)
        return actions
    
    def current_evaluate(self, observations):
        return self.critic_model(observations)

    def old_evaluate(self, observations):
        return self.old_critic_model(observations)

    def soft_update(self, target, source, tau=0.005):
        for target, source in zip(target.parameters(), source.parameters()):
            target.data.copy_(tau * source.data + (1.0 - tau) * target.data)

    def reset_noise(self):
        for module in self.actor_model.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()
    
    def update(self, samples):
        observations, actions, rewards, next_observations, terminated = zip(*samples)

        observations = torch.tensor(
            [obs[0] if isinstance(obs, tuple) else obs for obs in observations],
            dtype=torch.float32
        ).to(self.device)
        next_observations = torch.tensor(
            [obs[0] if isinstance(obs, tuple) else obs for obs in next_observations],
            dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            [action[0] if isinstance(action, tuple) else action for action in actions],
            dtype=torch.float32
        ).to(self.device)
        rewards = torch.tensor(
            [reward[0] if isinstance(reward, tuple) else reward for reward in rewards],
            dtype=torch.float32
        ).to(self.device)
        terminated = torch.tensor(
            [term[0] if isinstance(term, tuple) else term for term in terminated],
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            rewards = rewards.unsqueeze(1)
            terminated = terminated.unsqueeze(1)

            next_actions = self.select_old_actions(next_observations)
            next_q_values = self.old_evaluate(torch.cat((next_observations, next_actions), dim=1))
            targets_critic = rewards + (1 - terminated) * settings.DISCOUNT * next_q_values

        # Critic Loss ===============================================================
        current_q = self.current_evaluate(torch.cat((observations, actions), dim=1))
        critic_loss = torch.mean((targets_critic - current_q) ** 2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Critic Loss ===============================================================

        # Actor Loss ===============================================================
        # Prepare new Q-value
        new_actions = self.actor_model(observations)
        new_q_values = self.current_evaluate(torch.cat((observations, new_actions), dim=1))
        avg_grad_objective = -new_q_values.mean()

        # Backward Propagation for only Actor
        self.actor_optimizer.zero_grad()
        avg_grad_objective.backward()
        self.actor_optimizer.step()
        # Actor Loss ===============================================================

        # Target Models Update
        self.soft_update(self.old_actor_model, self.actor_model)
        self.soft_update(self.old_critic_model, self.critic_model)
    