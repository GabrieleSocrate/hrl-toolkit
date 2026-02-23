import gymnasium as gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os

def set_seed(seed):
    """In Python there isn't a single "global" random  generator.
    random, NumPy, PyTorch and the enviroment all use their own randomness.
    We set the same seed everywhere so the code behaves the same every time we run it"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(env_name, seed = 0):
    env = gym.make(env_name)
    env.reset(seed = seed)
    env.action_space.seed(seed)
    return env

def to_tensor(obs, device = None):
    obs = np.asarray(obs, dtype = np.float32)
    t = torch.as_tensor(obs, dtype = torch.float32)
    if device is not None:
        t = t.to(device)
    return t

def one_hot_option(option, num_options):
    v = np.zeros(num_options, dtype = np.float32)
    v[int(option)] = 1.0
    return v

def soft_update(net, net_targ, tau: float):
    """
    Soft-update target network parameters:
        θ_targ <- (1 - tau) * θ_targ + tau * θ

    Why:
    - target network changes slowly
    - TD targets become more stable
    - training becomes less noisy / less likely to diverge
    """
    with torch.no_grad():
        for p, p_targ in zip(net.parameters(), net_targ.parameters()):
            p_targ.data.mul_(1.0 - tau)
            p_targ.data.add_(tau * p.data)

def moving_average(x, window=50):
    """
    Return a list with the moving average of x using a fixed window size.
    Example: window=3 -> [avg(x[0:3]), avg(x[1:4]), ...]
    """
    x = list(x)

    # Not enough points -> no moving average
    if len(x) < window:
        return []

    averages = []

    # For each window of length 'window', compute its mean
    for i in range(len(x) - window + 1):
        chunk = x[i : i + window]          # take window values
        avg = sum(chunk) / window          # compute average
        averages.append(avg)

    return averages

def save_plots(run_dir, rows, ma_window=50):
    if not rows:
        print("No episode rows to plot.")
        return

    episodes = [r["episode"] for r in rows]
    ep_return = [r["ep_return"] for r in rows]

    # 1) Return + moving average
    plt.figure()
    plt.plot(episodes, ep_return)
    ma = moving_average(ep_return, window=ma_window)
    if len(ma) > 0:
        # moving average is shorter -> align it to the end
        plt.plot(episodes[-len(ma):], ma)
    plt.title("Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig(os.path.join(run_dir, "return.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Losses (NaN when missing)
    def col(name):
        out = []
        for r in rows:
            v = r.get(name, None)
            out.append(np.nan if v is None else float(v))
        return out

    plt.figure()
    plt.plot(episodes, col("critic_loss"))
    plt.plot(episodes, col("actor_loss"))
    plt.plot(episodes, col("optv_loss"))
    plt.plot(episodes, col("term_loss"))
    plt.title("Losses (may be sparse / NaN when not updated)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend(["critic", "actor", "optv", "term"])
    plt.savefig(os.path.join(run_dir, "losses.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3) HRL stats per episode
    plt.figure()
    plt.plot(episodes, [r["terminations"] for r in rows])
    plt.plot(episodes, [r["switches"] for r in rows])
    plt.title("HRL stats per episode")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.legend(["terminations", "switches"])
    plt.savefig(os.path.join(run_dir, "hrl_stats.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4) Average option duration per episode
    plt.figure()
    avg_term = col("avg_term_steps")

    # allineamento corretto: prendo solo (ep, val) validi
    ep_valid = [ep for ep, v in zip(episodes, avg_term) if not np.isnan(v)]
    val_valid = [v  for v  in avg_term if not np.isnan(v)]

    # plot solo dei punti validi (niente “linea a buchi”)
    plt.plot(ep_valid, val_valid)

    ma2 = moving_average(val_valid, window=ma_window)
    if len(ma2) > 0:
        plt.plot(ep_valid[-len(ma2):], ma2)

    plt.title("Avg option duration at termination (mean TERM EVENT) per episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend(["avg_term_steps", "moving average"])
    plt.savefig(os.path.join(run_dir, "avg_option_duration_term.png"), dpi=150, bbox_inches="tight")
    plt.close()