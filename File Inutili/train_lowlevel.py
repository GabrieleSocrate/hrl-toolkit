# train_lowlevel.py
import argparse
from utils import set_seed, make_env
from experience_replay import ReplayBuffer
from agents import make_agent

import csv, os
os.makedirs("runs", exist_ok=True)
csv_path = "runs/lowlevel_metrics.csv"

csv_file = open(csv_path, "w", newline="")
writer = csv.DictWriter(csv_file, fieldnames=[
    "episode", "t", "ep_len", "ep_return",
    "critic_loss", "actor_loss"
])
writer.writeheader()

last_critic_loss, last_actor_loss = None, None


def get_noise_std(ep):
    """At the beginning the noise will be high, so high exploration,
    then I decrease it, so more exploitation"""
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

    agent = make_agent(
        args.algo,
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        device=args.device,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden=args.hidden,
        # TD3-only params (ignored by DDPG)
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        num_options=args.num_options
    )

    buffer = ReplayBuffer(max_size=args.buffer_size)

    obs, info = env.reset(seed=args.seed)
    ep_return, ep_len = 0.0, 0
    episodes = 0

    actor_updates = 0
    actor_print_every = args.actor_print_every

    for t in range(args.total_steps):
        noise_std = get_noise_std(episodes)
        option = 0
        action = agent.act(obs, noise_std=noise_std, option = option) # the agent choose an action, we add some noise for exploration

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)

        buffer.push(obs, action, reward, next_obs, done, option = option) # save the transition in the replay buffer

        obs = next_obs
        ep_return += float(reward)
        ep_len += 1

        # we update after warmup and if there are enough samples to create a batch
        if t >= args.start_steps and len(buffer) >= args.batch_size:
            out = agent.update(
                buffer,
                batch_size=args.batch_size,
                update_iteration=args.update_iteration
            )
            last_critic_loss = float(critic_loss)
            last_actor_loss = None if actor_loss is None else float(actor_loss)

            # we update critic and actor using a batch from the buffer
            """update_iteration = how many "gradient steps" you do for each iteration with the enviroment,
            if it's too high you risk to overfit the replay buffer"""

            # TD3 returns (critic_loss, actor_loss, did_actor_update)
            # DDPG returns (critic_loss, actor_loss)
            if isinstance(out, tuple) and len(out) == 3:
                critic_loss, actor_loss, did_actor_update = out
            else:
                critic_loss, actor_loss = out
                did_actor_update = True  # DDPG updates actor every update() call

            # print only every N actor updates 
            if did_actor_update:
                actor_updates += 1
                if actor_print_every > 0 and (actor_updates % actor_print_every == 0):
                    if actor_loss is None:
                        print(f"[ACTOR UPDATED] t={t} | actor_loss=NA | actor_updates={actor_updates}")
                    else:
                        print(f"[ACTOR UPDATED] t={t} | actor_loss={actor_loss:.4f} | actor_updates={actor_updates}")

            if (t % args.log_every) == 0:
                actor_str = f"{actor_loss:.4f}" if actor_loss is not None else "NA"
                print(
                    f"algo={args.algo} | t={t} | critic_loss={critic_loss:.4f} | actor_loss={actor_str} "
                    f"| actor_update={did_actor_update} | buffer={len(buffer)}"
                )

        if done:
            episodes += 1
            print(f"ep {episodes} done | len={ep_len} | return={ep_return:.2f} | t={t}")
            obs, info = env.reset()
            ep_return, ep_len = 0.0, 0
            writer.writerow({
                "episode": episodes,
                "t": t,
                "ep_len": ep_len,
                "ep_return": ep_return,
                "critic_loss": last_critic_loss if last_critic_loss is not None else "",
                "actor_loss": last_actor_loss if last_actor_loss is not None else "",
                })
            csv_file.flush()


    env.close()
    csv_file.close()
    print("Saved metrics to:", csv_path)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="td3", choices=["ddpg", "td3"])
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

    p.add_argument("--log_every", type=int, default=2000)

    # TD3-only (safe to pass also for DDPG)
    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--policy_delay", type=int, default=2)

    
    p.add_argument("--actor_print_every", type=int, default=200)
    p.add_argument("--num_options", type=int, default=1)


    args = p.parse_args()
    train(args)
