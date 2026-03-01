import argparse
import os
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from ddpg import DDPG
from td3 import TD3
from option_agent import OptionAgent


def put_text(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw text on an RGB frame (H,W,3) using Pillow."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # font default (ok cross-platform). If you want nicer:
    # font = ImageFont.truetype("arial.ttf", 18)
    font = ImageFont.load_default()

    # black box background
    x, y = 10, 10
    padding = 4
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding],
        fill=(0, 0, 0)
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return np.array(img)


def build_env(env_id: str, seed: int):
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def build_agent(obs_dim, act_dim, act_limit, args):
    algo = args.algo.lower()

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
            num_options=args.num_options,
        )
    else:
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
            num_options=args.num_options,
        )

    agent = OptionAgent(
        obs_dim=obs_dim,
        num_options=args.num_options,
        low_level_agent=low_level,
        device=args.device,
        hidden=args.hidden_high,
        eps_option=0.0,  # eval: no epsilon
        terminate_deterministic=args.terminate_deterministic,
        min_option_steps=args.min_option_steps,
    )
    agent.delib_cost = args.delib_cost
    return agent


def load_checkpoint(agent, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
    else:
        agent.load_state_dict(ckpt)
    return ckpt


@torch.no_grad()
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    env = build_env(args.env, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = build_agent(obs_dim, act_dim, act_limit, args)
    load_checkpoint(agent, args.ckpt, device=args.device)

    ###################
    # --- save video in the SAME run folder (where plots are) ---
    ckpt_path = args.ckpt
    checkpoints_dir = os.path.dirname(ckpt_path)   # .../runs/<run_name>/checkpoints
    run_dir = os.path.dirname(checkpoints_dir)     # .../runs/<run_name>  (plots are here)
    out_path = os.path.join(run_dir, args.out_name)
    writer = imageio.get_writer(out_path, fps=args.fps)
    ####################
    obs, _ = env.reset(seed=args.seed)
    agent.reset(obs)

    done = False
    t = 0
    prev_option = None

    while not done and t < args.max_steps:
        # policy (eval): noise=0, greedy options
        action, option, did_terminate, term_steps = agent.act(
            obs, noise_std=0.0, greedy_option=True
        )

        if prev_option is None:
            prev_option = option
        if did_terminate:
            print(f"[TERM] t={t} | option(after)={option} | term_steps={term_steps}")

        if option != prev_option:
            print(f"[SWITCH] t={t} {prev_option} -> {option} | did_terminate={did_terminate} | option_steps={term_steps}")
            prev_option = option

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()  # RGB array
        frame = put_text(frame, f"Option: {int(option)} | term={int(did_terminate)} | term_steps={term_steps} | t={t} | r={reward:+.2f}")
        writer.append_data(frame)

        obs = next_obs
        t += 1

    writer.close()
    env.close()
    print(f"\nSaved annotated video: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="LunarLanderContinuous-v3")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="videos")
    p.add_argument("--out_name", type=str, default="lander_annotated.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--algo", type=str, default="td3", choices=["ddpg", "td3"])
    p.add_argument("--num_options", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden_low", type=int, default=256)
    p.add_argument("--hidden_high", type=int, default=256)

    p.add_argument("--terminate_deterministic", action="store_true")
    p.add_argument("--min_option_steps", type=int, default=0)
    p.add_argument("--delib_cost", type=float, default=0.5)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_lr", type=float, default=1e-3)
    p.add_argument("--critic_lr", type=float, default=1e-3)

    p.add_argument("--policy_noise", type=float, default=0.2)
    p.add_argument("--noise_clip", type=float, default=0.5)
    p.add_argument("--policy_delay", type=int, default=2)

    args = p.parse_args()
    main(args)