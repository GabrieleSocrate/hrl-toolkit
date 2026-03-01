import argparse
import numpy as np
import torch
import os
import csv
from datetime import datetime
import json
import sys
import shlex

from utils import set_seed, make_env, save_plots
from experience_replay import ReplayBuffer
from ddpg import DDPG
from td3 import TD3
from option_agent import OptionAgent

"""The structure of this file is very similar to other training file"""

def get_noise_std(ep):
    """At the beginning the noise will be high (exploration),
    then we decrease it (more exploitation)."""
    if ep < 500: # prima era 1000
        return 0.5
    elif ep < 800: # prima era 1500
        return 0.1
    else:
        return 0.01 

def save_checkpoint(run_dir, agent, t, episodes, total_actor_updates_seen, prev_cum_terminations, prev_cum_switches, save_last=False):
    """Save a checkpoint WITHOUT overwriting previous checkpoints.
    If save_last=True, also writes/overwrites checkpoints/last.pt (optional convenience).
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    payload = {
        "agent": agent.state_dict(),
        "t": int(t),
        "episodes": int(episodes),
        "total_actor_updates_seen": int(total_actor_updates_seen),
        "prev_cum_terminations": int(prev_cum_terminations),
        "prev_cum_switches": int(prev_cum_switches),
    }

    path = os.path.join(ckpt_dir, f"ckpt_ep{episodes:05d}_t{t:07d}.pt")
    torch.save(payload, path)

    if save_last:
        torch.save(payload, os.path.join(ckpt_dir, "last.pt"))

    return path

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
            hidden=args.hidden_low,
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
            hidden=args.hidden_low,
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
        hidden=args.hidden_high,
        eps_option=args.eps_option,
        terminate_deterministic=args.terminate_deterministic,
        min_option_steps = args.min_option_steps,
        optv_lr = args.optv_lr,
        term_lr = args.term_lr
    )

    agent.delib_cost = args.delib_cost

    buffer = ReplayBuffer(max_size=args.buffer_size)

    #########################################
    # Modifiche per grafici
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env}_{algo}_seed{args.seed}_{timestamp}"
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save run configuration for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save the exact command line used for this run (associated to this run_dir)
    # This is useful when you look at plots and want to remember how you launched the run.
    cmd_list = sys.argv[:]  # raw argv list
    cmd_str = " ".join(shlex.quote(x) for x in cmd_list)

    with open(os.path.join(run_dir, "cmd.txt"), "w", encoding="utf-8") as f:
        f.write(cmd_str + "\n")

    with open(os.path.join(run_dir, "cmd.json"), "w", encoding="utf-8") as f:
        json.dump({"argv": cmd_list, "cmd": cmd_str}, f, indent=2)

    csv_path = os.path.join(run_dir, "episodes.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode","t_step","ep_return","ep_len",
        "critic_loss","actor_loss","optv_loss","term_loss",
        "terminations","switches","avg_term_steps"
    ])
    csv_writer.writeheader()
    ###################################

    obs, info = env.reset(seed=args.seed)
    agent.reset(obs)  # pick an initial option at the start of episode

    ep_return, ep_len = 0.0, 0
    episodes = 0

    total_actor_updates_seen = 0

    episode_rows = []
    last_critic_loss, last_actor_loss = None, None
    last_optv_loss, last_term_loss = None, None

    # agent.get_stats() is cumulative: keep prev values to log per-episode deltas
    prev_cum_terminations = 0
    prev_cum_switches = 0

    ep_term_steps_sum = 0
    ep_term_steps_count = 0

    for t in range(args.total_steps):
        noise_std = get_noise_std(episodes)
        action, option, did_terminate, term_steps = agent.act(obs, noise_std=noise_std, greedy_option=False) # term_steps è MODIFICA PER DEBUG

        next_obs, reward, terminated, truncated, info = env.step(action)
        """
        did_terminate refers to the option if it ended
        terminated is True if the episodes finishes due to goal reach (always false)
        truncated is True if the episode is forced to terminate (True after 200 step)
        """
        episode_done = bool(terminated or truncated) # I need bool for the condition under here
        terminal_float = float(terminated) # I separate temrinated form truncated for Bellman update
        ##################
        # Modifica debug
        if did_terminate and (term_steps is not None) and (not episode_done):
            ep_term_steps_sum += int(term_steps)
            ep_term_steps_count += 1
        ##################

        """
        We also want the deliberation cost to affect the replay buffer reward:
        If we terminated the option and the episode is not ending , we pay a cost
        """
        reward_eff = float(reward) - float(did_terminate) * float(agent.delib_cost) * (1.0 - terminal_float)

        # Store HRL transition 
        buffer.push(
            obs=obs,
            action=action,
            reward=reward_eff,
            next_obs=next_obs,
            done=terminal_float,
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

        if episode_done:
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
                "t_step": t,
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

    # Save final checkpoint (unique filename, so it won't overwrite previous ones)
    ckpt_path = save_checkpoint(
        run_dir=run_dir,
        agent=agent,
        t=t,
        episodes=episodes,
        total_actor_updates_seen=total_actor_updates_seen,
        prev_cum_terminations=prev_cum_terminations,
        prev_cum_switches=prev_cum_switches,
        save_last=args.save_last,
    )
    print("Saved final checkpoint:", ckpt_path)

    save_plots(run_dir, episode_rows, ma_window=50)
    print("Saved logs to:", csv_path)
    print("Saved plots to:", run_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="LunarLanderContinuous-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--total_steps", type=int, default=500000)
    p.add_argument("--buffer_size", type=int, default=1000000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--start_steps", type=int, default=5000)      
    p.add_argument("--update_iteration", type=int, default=1)     
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_lr", type=float, default=1e-3) # actor lr
    p.add_argument("--critic_lr", type=float, default=1e-3) # critic lr
    p.add_argument("--optv_lr", type=float, default=1e-3)   # option-value lr
    p.add_argument("--term_lr", type=float, default=1e-3)   # termination lr 
    p.add_argument("--hidden_low", type=int, default=256)   # low-level Actor/Critic
    p.add_argument("--hidden_high", type=int, default=256)  # high-level OptionValue/Termination
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
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--save_last", action="store_true") # optional: also write checkpoints/last.pt (this overwrites last.pt)
    p.add_argument("--delib_cost", type=float, default=0.5)  # deliberation cost (high-level)
 

    args = p.parse_args()
    train(args)