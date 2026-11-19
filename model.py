import torch.nn as nn
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module):
    def __init__(self, num_inputs=8, num_actions=4):
        super(ActorCritic, self).__init__()
        # THE ACTOR: Takes the state (8), outputs probabilities for each button (4)
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)  # Forces outputs to be valid probabilities (summing to 1.0)
        )
        # THE CRITIC: Takes the state (8), outputs a single number guessing the score (1)
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def get_value(self,state):
        """Used during Phase 1 (Data Collection) to calculate the GAE surprises."""
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        """Used during Phase 2 (Optimization) to get everything needed for Equation 9."""
        action_probs=self.actor(state)
        dist=Categorical(action_probs)
        # If we are collecting data, sample a new action.
        # If we are training, evaluate the old action we passed in.
        if action is None:
            action = dist.sample()
        # We need log probabilities for the probability ratio r_t(theta)
        log_prob = dist.log_prob(action)
        # PyTorch calculates the S (Entropy Bonus) for us automatically!
        entropy = dist.entropy()
        # Get the critic's guess V(s)
        value = self.critic(state)
        return action, log_prob, entropy, value