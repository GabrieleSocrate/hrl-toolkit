import argparse
from utils import set_seed, make_env


def run(args):
    set_seed(args.seed)
    env = make_env(args.env, seed=args.seed)

    print("OBS space:", env.observation_space)
    print("ACT space:", env.action_space)

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    ep_steps = 0
    episodes = 0

    for t in range(args.total_steps):
        action = env.action_space.sample()  # generae a random action from the action space
        obs, reward, terminated, truncated, info = env.step(action) # make a step in the enviroment using that action and receives reward...

        total_reward += float(reward)
        ep_steps += 1

        if terminated or truncated:
            episodes += 1
            print(f"ep {episodes} done | steps={ep_steps} | return={total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
            ep_steps = 0

    env.close()

if __name__ == "__main__": # it means execute this code only if I'm executing directly the main file 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--env", type=str, default="Pendulum-v1") # to choose the enviroment
    parser.add_argument("--seed", type=int, default=0) # to choose the seed
    parser.add_argument("--total_steps", type=int, default=2000) # to choose the number of total steps
    args = parser.parse_args()
    run(args)