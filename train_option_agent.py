import argparse
import numpy as np
from utils import set_seed, make_env
from experience_replay import ReplayBuffer
from ddpg import DDPG
from td3 import TD3
from option_agent import OptionAgent

"""The structure of this file is very similar to other training file"""

def get_noise_std(ep: int) -> float:
    """At the beginning the noise will be high (exploration),
    then we decrease it (more exploitation)."""
    if ep < 1000:
        return 0.5
    elif ep < 1500:
        return 0.1
    else:
        return 0.01


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
    )

    buffer = ReplayBuffer(max_size=args.buffer_size)

    obs, info = env.reset(seed=args.seed)
    agent.reset(obs)  # pick an initial option at the start of episode

    ep_return, ep_len = 0.0, 0
    episodes = 0

    total_actor_updates_seen = 0

    for t in range(args.total_steps):
        noise_std = get_noise_std(episodes)
        action, option, did_terminate = agent.act(obs, noise_std=noise_std, greedy_option=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)

        # Store HRL transition 
        buffer.push(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            option=option,
            terminated=float(did_terminate),
        )

        obs = next_obs
        ep_return += float(reward)
        ep_len += 1

        # we update after warmup and if there are enough samples to create a batch
        if t >= args.start_steps and len(buffer) >= args.batch_size:
            update_out = agent.update(
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
            if isinstance(update_out, tuple) and len(update_out) == 3:
                critic_loss, actor_loss, did_actor_update = update_out
            else:
                critic_loss, actor_loss = update_out
                did_actor_update = True  # DDPG updates actor every update() call

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

                print(
                    f"t={t} | algo={algo}"
                    f" | critic_loss={critic_loss_str}"
                    f" | actor_loss={actor_loss_str}"
                    f" | buffer={len(buffer)}"
                    f" | opt={stats['current_option']}"
                    f" | opt_steps={stats['option_steps']}"
                    f" | terminations={stats['num_terminations']}"
                    f" | switches={stats['num_option_switches']}"
                )

        if done:
            episodes += 1
            print(f"ep {episodes} done | len={ep_len} | return={ep_return:.2f} | t={t}")

            obs, info = env.reset()
            agent.reset(obs)

            ep_return, ep_len = 0.0, 0

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--buffer_size", type=int, default=1000000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--start_steps", type=int, default=5000)      
    p.add_argument("--update_iteration", type=int, default=1)     
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_lr", type=float, default=1e-3)
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--algo", type=str, default="td3", choices=["ddpg", "td3"])
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--policy_delay", type=int, default=2)
    p.add_argument("--num_options", type=int, default=4)
    p.add_argument("--eps_option", type=float, default=0.0)
    p.add_argument("--terminate_deterministic", action="store_true")
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--print_actor_every", type=int, default=200)  

    args = p.parse_args()
    train(args)