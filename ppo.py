import torch
import torch.nn as nn
from model import ActorCritic


class PPOAgent:
    def __init__(self, state_dim=8, action_dim=4, lr=0.002, gamma=0.999, lam=0.95, clip_ratio=0.2, K_epochs=10, c1=1, c2=0,batch_size=64):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.K_epochs = K_epochs
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size

        # Initialize the Brain and the Optimizer
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, next_value, device):
        advantages, returns = memory.compute_gae(next_value, self.gamma, self.lam)
        advantages = advantages.to(device)
        returns = returns.to(device)

        old_states = torch.stack(memory.states).squeeze(1).detach().to(device)
        old_actions = torch.stack(memory.actions).squeeze(-1).detach().to(device)
        old_logprobs = torch.stack(memory.logprobs).squeeze(-1).detach().to(device)

        dataset_size = old_states.size(0)

        # Accumulators for true averages
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.K_epochs):
            # Shuffle indices for mini-batching
            indices = torch.randperm(dataset_size, device=device)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                _, new_logprobs, entropy, state_values = self.policy.get_action_and_value(
                    old_states[mb_idx], old_actions[mb_idx]
                )
                state_values = torch.squeeze(state_values)

                ratios = torch.exp(new_logprobs - old_logprobs[mb_idx])

                surr1 = ratios * advantages[mb_idx]
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = self.MseLoss(state_values, returns[mb_idx])
                entropy_bonus = entropy.mean()

                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_bonus.item()
                num_updates += 1

        avg_actor = total_actor_loss / num_updates
        avg_critic = total_critic_loss / num_updates
        avg_entropy = total_entropy / num_updates
        print(f"Update Complete | Actor Loss: {avg_actor:.4f} | Critic Loss: {avg_critic:.4f} | Entropy: {avg_entropy:.4f}")
        memory.clear()
        return avg_actor, avg_critic, avg_entropy
