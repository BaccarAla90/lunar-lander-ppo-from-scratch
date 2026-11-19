import torch
class RolloutBuffer:
    def __init__(self):
        # These lists store the gameplay data.
        # self.logprobs is where our pi_old data lives!
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []  # The Critic's guesses V(s)
        self.is_terminals = []  # True if the ship crashed or landed (game over)

    def clear(self):
        """Empties the memory after the network finishes training on it."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]

    def compute_gae(self, next_value, gamma=0.99, lam=0.95):
        """Calculates Equation 11 & 12 (The Surprise Chain)"""
        advantages = []
        gae = 0
        # We must loop BACKWARDS through time to chain the future rewards
        for step in reversed(range(len(self.rewards))):
            # If we are at the very last step, we look at the 'next_value' we passed in
            if step == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.is_terminals[-1]
                next_val = next_value
            # Otherwise, we look at the next step in our saved memory
            else:
                next_non_terminal = 1.0 - self.is_terminals[step + 1]
                next_val = self.values[step + 1]

            # EQUATION 12: The 1-step surprise (delta)
            # Reality (Reward + expected future) - What the Critic guessed
            delta = self.rewards[step] + gamma * next_val * next_non_terminal - self.values[step]
            # EQUATION 11: Chaining the surprises together with lambda
            gae = delta + gamma * lam * next_non_terminal * gae
            # Insert at the front of the list because we are looping backwards
            advantages.insert(0, gae)
        # Convert lists to PyTorch tensors for the Optimizer
        advantages = torch.tensor(advantages, dtype=torch.float32)
        values_tensor = torch.tensor(self.values, dtype=torch.float32)
        # Calculate the true total scores (Returns) so we can train the Critic later
        # True Score = How much better we did than expected (Advantage) + What we expected (Value)
        returns = advantages + values_tensor

        # Normalization: A standard deep learning trick to keep math stable (mean 0, std 1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns