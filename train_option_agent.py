import argparse
import numpy as np
import torch
import os
import csv
import matplotlib.pyplot as plt


from utils import set_seed, make_env
from experience_replay import ReplayBuffer
from ddpg import DDPG
from td3 import TD3
from option_agent import OptionAgent

"""The structure of this file is very similar to other training file"""

def get_noise_std(ep):
    """At the beginning the noise will be high (exploration),
    then we decrease it (more exploitation)."""
    if ep < 1000:
        return 0.5
    elif ep < 1500:
        return 0.1
    else:
        return 0.01

###################################
# Modifiche per grafici  
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


####################################


def train(args):
    set_seed(args.seed)

    env = make_env(args.env, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    algo = args.algo.lower().strip()
    if algo == "ddpg":
        low_level = DDPG(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_limit=act_limit,
            device=args.device,
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            hidden=args.hidden,
            num_options = args.num_options
        )
    elif algo == "td3":
        low_level = TD3(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_limit=act_limit,
            device=args.device,
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            hidden=args.hidden,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
            num_options = args.num_options
        )
    else:
        raise ValueError("--algo must be 'ddpg' or 'td3'")

    agent = OptionAgent(
        obs_dim=obs_dim,
        num_options=args.num_options,
        low_level_agent=low_level,
        device=args.device,
        hidden=args.hidden,
        eps_option=args.eps_option,
        terminate_deterministic=args.terminate_deterministic,
        min_option_steps = args.min_option_steps
    )

    buffer = ReplayBuffer(max_size=args.buffer_size)

    obs, info = env.reset(seed=args.seed)
    agent.reset(obs)  # pick an initial option at the start of episode

    ep_return, ep_len = 0.0, 0
    episodes = 0

    total_actor_updates_seen = 0

    ###################################
    # Modifiche per grafici
    # ---- minimal logging to CSV + plots ----
    run_name = f"{args.env}_{algo}_seed{args.seed}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "episodes.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode",
            "t",
            "ep_len",
            "ep_return",
            "critic_loss",
            "actor_loss",
            "optv_loss",
            "term_loss",
            "terminations",
            "switches",
            "avg_term_steps",
        ],
    )
    csv_writer.writeheader()

    episode_rows = []
    last_critic_loss, last_actor_loss = None, None
    last_optv_loss, last_term_loss = None, None

    # agent.get_stats() is cumulative: keep prev values to log per-episode deltas
    prev_cum_terminations = 0
    prev_cum_switches = 0

    ep_term_steps_sum = 0
    ep_term_steps_count = 0
    ################################################

    for t in range(args.total_steps):
        noise_std = get_noise_std(episodes)
        action, option, did_terminate, term_steps = agent.act(obs, noise_std=noise_std, greedy_option=False) # term_steps è MODIFICA PER DEBUG

        next_obs, reward, terminated, truncated, info = env.step(action)
        """
        did_terminate refers to the option if it ended
        terminated is True if the episodes finishes due to goal reach (always false)
        truncated is True if the episode is forced to terminate (True after 200 step)
        """
        done = bool(terminated or truncated) # I need bool for the condition under here
        ##################
        # Modifica debug
        if did_terminate and (term_steps is not None) and (not done):
            ep_term_steps_sum += int(term_steps)
            ep_term_steps_count += 1
        ##################

        """
        We also want the deliberation cost to affect the replay buffer reward:
        If we terminated the option and the episode is not ending , we pay a cost
        """
        done_float = float(done)
        reward_eff = float(reward) - float(did_terminate) * float(agent.delib_cost) * (1.0 - done_float)

        # Store HRL transition 
        buffer.push(
            obs=obs,
            action=action,
            reward=reward_eff,
            next_obs=next_obs,
            done=done_float,
            option=option,
            terminated=float(did_terminate),
        )

        obs = next_obs
        ep_return += float(reward)
        ep_len += 1

        # we update after warmup and if there are enough samples to create a batch
        if t >= args.start_steps and len(buffer) >= args.batch_size:
            low_out, optv_loss, term_loss = agent.update(
                buffer,
                batch_size=args.batch_size,
                update_iteration=args.update_iteration,
            )
            # we update critic and actor using a batch from the buffer
            """update_iteration = how many "gradient steps" you do for each iteration with the enviroment,
            if it's too high you risk to overfit the replay buffer"""
            
            critic_loss, actor_loss = None, None
            did_actor_update = False

            # TD3 returns (critic_loss, actor_loss, did_actor_update)
            # DDPG returns (critic_loss, actor_loss)
            if isinstance(low_out, tuple) and len(low_out) == 3:
                critic_loss, actor_loss, did_actor_update = low_out
            else:
                critic_loss, actor_loss = low_out
                did_actor_update = True  # DDPG updates actor every update() call

            ###########################
            # Modifiche grafici
             # keep last seen losses for episode-level logging
            try:
                last_critic_loss = None if critic_loss is None else float(critic_loss)
            except Exception:
                last_critic_loss = None
            try:
                last_actor_loss = None if actor_loss is None else float(actor_loss)
            except Exception:
                last_actor_loss = None
            try:
                last_optv_loss = None if optv_loss is None else float(optv_loss)
            except Exception:
                last_optv_loss = None
            try:
                last_term_loss = None if term_loss is None else float(term_loss)
            except Exception:
                last_term_loss = None
            ############################

            if did_actor_update:
                total_actor_updates_seen += 1
                if (total_actor_updates_seen % args.print_actor_every) == 0:
                    try:
                        print(f"[ACTOR UPDATED] t={t} | actor_loss={float(actor_loss):.4f} | actor_updates={total_actor_updates_seen}")
                    except Exception:
                        print(f"[ACTOR UPDATED] t={t} | actor_updates={total_actor_updates_seen}")

            if (t % args.log_every) == 0:
                stats = agent.get_stats()
                actor_loss_str = "NA" if actor_loss is None else f"{float(actor_loss):.4f}"
                critic_loss_str = "NA" if critic_loss is None else f"{float(critic_loss):.4f}"
                optv_loss_str = "NA" if optv_loss is None else f"{float(optv_loss):.4f}"
                term_loss_str = "NA" if term_loss is None else f"{float(term_loss):.4f}"

                print(
                    f"t={t} | algo={algo}"
                    f" | critic_loss={critic_loss_str}"
                    f" | actor_loss={actor_loss_str}"
                    f" | buffer={len(buffer)}"
                    f" | opt={stats['current_option']}"
                    f" | opt_steps={stats['option_steps']}"
                    f" | terminations={stats['num_terminations']}"
                    f" | switches={stats['num_option_switches']}"
                    f" | optv_loss={optv_loss_str}"
                    f" | term_loss={term_loss_str}"
                )

        if done:
            episodes += 1
            print(f"ep {episodes} done | len={ep_len} | return={ep_return:.2f} | t={t}")

            #######################
            # Modifica per grafici
            # log per-episode stats (convert cumulative stats to episode deltas)
            stats = agent.get_stats()
            cum_terms = int(stats.get("num_terminations", 0))
            cum_switch = int(stats.get("num_option_switches", 0))
            ep_terms = max(0, cum_terms - prev_cum_terminations)
            ep_switch = max(0, cum_switch - prev_cum_switches)
            prev_cum_terminations = cum_terms
            prev_cum_switches = cum_switch

            if ep_term_steps_count > 0:
                avg_term_steps = ep_term_steps_sum / ep_term_steps_count
            else:
                avg_term_steps = None


            row = {
                "episode": episodes,
                "t": t,
                "ep_len": ep_len,
                "ep_return": float(ep_return),
                "critic_loss": last_critic_loss,
                "actor_loss": last_actor_loss,
                "optv_loss": last_optv_loss,
                "term_loss": last_term_loss,
                "terminations": ep_terms,
                "switches": ep_switch,
                "avg_term_steps": avg_term_steps,
            }
            csv_writer.writerow(row)
            csv_file.flush()
            episode_rows.append(row)
            ep_term_steps_sum = 0
            ep_term_steps_count = 0

            ###################################
            
            obs, info = env.reset()
            agent.reset(obs)

            ep_return, ep_len = 0.0, 0

    env.close()

    csv_file.close()
    save_plots(run_dir, episode_rows, ma_window=50)
    print("Saved logs to:", csv_path)
    print("Saved plots to:", run_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total_steps", type=int, default=500000)
    p.add_argument("--buffer_size", type=int, default=1000000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--start_steps", type=int, default=5000)      
    p.add_argument("--update_iteration", type=int, default=1)     
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_lr", type=float, default=1e-3)
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--algo", type=str, default="ddpg", choices=["ddpg", "td3"])
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--policy_delay", type=int, default=2)
    p.add_argument("--num_options", type=int, default=2)
    p.add_argument("--eps_option", type=float, default=0.0)
    p.add_argument("--terminate_deterministic", action="store_true") # is false by default if you write in command line it becames true 
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--print_actor_every", type=int, default=200) 
    p.add_argument("--min_option_steps", type=int, default=50) # each option has to last for at least 50 steps then it can be changed
 

    args = p.parse_args()
    train(args)