import argparse
from utils import set_seed, make_env
from experience_replay import ReplayBuffer


def run(args):
    set_seed(args.seed)
    env = make_env(args.env, seed=args.seed)

    print("OBS space:", env.observation_space)
    print("ACT space:", env.action_space)

    buffer = ReplayBuffer(max_size = args.buffer_size)

    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    ep_steps = 0
    episodes = 0

    for t in range(args.total_steps):
        action = env.action_space.sample()  # generate a random action from the action space
        next_obs, reward, terminated, truncated, info = env.step(action) # make a step in the enviroment using that action and receives reward...
        done = float(terminated or truncated)

        buffer.push(obs, action, reward, next_obs, done)

        total_reward += float(reward)
        ep_steps += 1
        obs = next_obs

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
    parser.add_argument("--buffer_size", type=int, default=100000) # to choose buffer size
    parser.add_argument("--batch_size", type=int, default=32) # to choose batch size
    args = parser.parse_args()
    run(args)