import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def plot_training(history):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # Episode Rewards + 20-episode Moving Average
    rewards = history["ep_rewards"]
    axs[0, 0].plot(rewards, alpha=0.3, label="Raw Reward")
    if len(rewards) >= 20:
        ma = np.convolve(rewards, np.ones(20)/20, mode="valid")
        axs[0, 0].plot(range(19, len(rewards)), ma, color="red", label="20-Ep Moving Avg")
    axs[0, 0].axhline(200, color="green", linestyle="--", label="Solved (+200)")
    axs[0, 0].set_title("Episode Rewards")
    axs[0, 0].legend()

    # Actor Loss
    axs[0, 1].plot(history["actor_loss"], color="orange")
    axs[0, 1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axs[0, 1].set_title("Average Actor Loss")

    # Critic Loss
    axs[1, 0].plot(history["critic_loss"], color="purple")
    axs[1, 0].set_title("Average Critic Loss")

    # Entropy
    axs[1, 1].plot(history["entropy"], color="teal")
    axs[1, 1].set_title("Policy Entropy")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_probs_and_value(policy, state_tensor):
    logits = policy.actor(state_tensor)
    probs = F.softmax(logits, dim=-1)
    value = policy.critic(state_tensor).squeeze(-1)
    return probs, value
# ==========================================
# 1 & 2. Value Heatmap & Decision Boundary
# ==========================================
def plot_brain_maps(policy, device):
    x_vals = np.linspace(-1.0, 1.0, 150)
    y_vals = np.linspace(0.0, 1.4, 150)
    X, Y = np.meshgrid(x_vals, y_vals)

    # State vector: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
    # Simulate a gentle downward descent (vy = -0.2) with level attitude
    grid_states = np.zeros((X.size, 8), dtype=np.float32)
    grid_states[:, 0] = X.ravel()
    grid_states[:, 1] = Y.ravel()
    grid_states[:, 3] = -0.2

    state_tensor = torch.tensor(grid_states, device=device)
    probs, values = get_probs_and_value(policy, state_tensor)

    V_grid = values.detach().cpu().numpy().reshape(X.shape)
    A_grid = torch.argmax(probs, dim=-1).cpu().numpy().reshape(X.shape)

    # --- Plot 1: Critic Value Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(X, Y, V_grid, levels=40, cmap="magma")
    plt.colorbar(cf, ax=ax, label="Predicted Value $V(s)$")
    ax.axvline(0, color="white", linestyle="--", alpha=0.4)
    ax.plot([-0.2, 0.2], [0.01, 0.01], color="lime", linewidth=4, label="Landing Pad")
    ax.set_title("Critic Value Heatmap $V(x, y)$ (Level Descent, $v_y = -0.2$)")
    ax.set_xlabel("Horizontal Position ($x$)")
    ax.set_ylabel("Altitude ($y$)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"1_value_heatmap.png", dpi=300)
    plt.close()

    # --- Plot 2: Actor Decision Boundary ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2b2d42", "#ef233c", "#06d6a0", "#ffd166"]
    cmap = ListedColormap(colors)
    ax.pcolormesh(X, Y, A_grid, cmap=cmap, shading="auto", vmin=0, vmax=3)
    ax.plot([-0.2, 0.2], [0.01, 0.01], color="white", linewidth=4)

    legend_elements = [
        Patch(facecolor=colors[0], label="0: Do Nothing"),
        Patch(facecolor=colors[1], label="1: Left Engine (Push Right)"),
        Patch(facecolor=colors[2], label="2: Main Engine (Brake)"),
        Patch(facecolor=colors[3], label="3: Right Engine (Push Left)"),
    ]
    ax.set_title("Actor Policy Decision Map (Level Descent, $v_y = -0.2$)")
    ax.set_xlabel("Horizontal Position ($x$)")
    ax.set_ylabel("Altitude ($y$)")
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.show()
    plt.savefig("2_decision_boundary.png", dpi=300)
    plt.close()





def plot_trajectories(trajectories, rewards):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6))

    for traj, r in zip(trajectories, rewards):
        color = "#00f5d4" if r >= 200 else "#ff595e"
        alpha = 0.25 if r >= 200 else 0.6
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=1.2)

    # Landing pad & flags
    ax.plot([-0.2, 0.2], [0, 0], color="yellow", linewidth=4, label="Landing Pad")
    ax.axvline(-0.2, ymin=0, ymax=0.1, color="yellow", linestyle="--", alpha=0.8)
    ax.axvline(0.2, ymin=0, ymax=0.1, color="yellow", linestyle="--", alpha=0.8)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.05, 1.5)
    ax.set_title(f"{len(trajectories)}-Episode Flight Trajectories (Deterministic)")
    ax.set_xlabel("Horizontal Position ($x$)")
    ax.set_ylabel("Altitude ($y$)")

    legend_elements = [
        Patch(facecolor="#00f5d4", label=f"Solved >= 200 ({np.sum(rewards >= 200)}/{len(rewards)})"),
        Patch(facecolor="#ff595e", label=f"Below 200 ({np.sum(rewards < 200)}/{len(rewards)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig("/3_trajectories.png", dpi=300)
    plt.close()
    plt.show()
    plt.style.use("default")




def run_evaluation(policy, device,env, deterministic=True, n_episodes=100):

    rewards, trajectories = [], []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0
        traj = []

        while not done:
            traj.append((state[0], state[1]))
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            probs, _ = get_probs_and_value(policy, state_t)

            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                action = torch.distributions.Categorical(probs).sample().item()

            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        trajectories.append(np.array(traj))

    env.close()
    return np.array(rewards), trajectories


# ==========================================
def plot_telemetry(policy, device,env, seed=42):
   
    state, _ = env.reset(seed=seed)
    done = False

    altitudes, v_ys, angles, values, probs_hist = [], [], [], [], []
    total_reward = 0.0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        probs, val = get_probs_and_value(policy, state_t)

        altitudes.append(state[1])
        v_ys.append(state[3])
        angles.append(np.degrees(state[4]))
        values.append(val.item())
        probs_hist.append(probs.squeeze(0).detach().cpu().numpy())

        action = torch.argmax(probs, dim=-1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    steps = np.arange(len(altitudes))
    probs_hist = np.array(probs_hist)

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Flight Telemetry (Seed {seed} | Total Reward: {total_reward:.1f} | {len(steps)} Steps)", fontsize=13)

    # 1. Altitude & Vertical Velocity
    axs[0].plot(steps, altitudes, color="#1d3557", label="Altitude ($y$)")
    axs[0].set_ylabel("Altitude")
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(steps, v_ys, color="#e63946", linestyle="--", alpha=0.8, label="Vertical Vel ($v_y$)")
    ax0_twin.set_ylabel("Vertical Vel", color="#e63946")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper left")
    ax0_twin.legend(loc="upper right")

    # 2. Tilt Angle
    axs[1].plot(steps, angles, color="#457b9d")
    axs[1].axhline(0, color="black", linestyle=":", alpha=0.5)
    axs[1].set_ylabel("Tilt (deg)")
    axs[1].grid(True, alpha=0.3)

    # 3. Critic Value Prediction
    axs[2].plot(steps, values, color="#6a4c93", linewidth=2)
    axs[2].set_ylabel("Critic $V(s_t)$")
    axs[2].grid(True, alpha=0.3)

    # 4. Action Probabilities (Stacked Area)
    axs[3].stackplot(
        steps, probs_hist.T,
        labels=["Do Nothing", "Left Thruster", "Main Engine", "Right Thruster"],
        colors=["#2b2d42", "#ef233c", "#06d6a0", "#ffd166"],
        alpha=0.85
    )
    axs[3].set_ylabel("Action Prob")
    axs[3].set_xlabel("Time Step")
    axs[3].set_ylim(0, 1)
    axs[3].legend(loc="upper right", ncol=4, fontsize=8)

    plt.tight_layout()
    plt.savefig("4_flight_telemetry.png", dpi=300)
    plt.show()
    plt.close()
