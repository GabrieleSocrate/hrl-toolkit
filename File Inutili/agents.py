from ddpg import DDPG
from td3 import TD3


def make_agent(algo: str, **kwargs):
    """
    Function to build a low-level continuous-control agent.
    algo: "ddpg" or "td3"
    kwargs: common args (obs_dim, act_dim, act_limit, device, gamma, tau, actor_lr, critic_lr, hidden, ...)
    """
    algo = algo.lower().strip()

    if algo == "ddpg":
        return DDPG(
            obs_dim=kwargs["obs_dim"],
            act_dim=kwargs["act_dim"],
            act_limit=kwargs["act_limit"],
            device=kwargs.get("device", "cpu"),
            gamma=kwargs.get("gamma", 0.99),
            tau=kwargs.get("tau", 0.005),
            actor_lr=kwargs.get("actor_lr", 1e-3),
            critic_lr=kwargs.get("critic_lr", 1e-3),
            hidden=kwargs.get("hidden", 256),
             num_options=kwargs.get("num_options", 1),
        )

    if algo == "td3":
        return TD3(
            obs_dim=kwargs["obs_dim"],
            act_dim=kwargs["act_dim"],
            act_limit=kwargs["act_limit"],
            device=kwargs.get("device", "cpu"),
            gamma=kwargs.get("gamma", 0.99),
            tau=kwargs.get("tau", 0.005),
            actor_lr=kwargs.get("actor_lr", 1e-3),
            critic_lr=kwargs.get("critic_lr", 1e-3),
            hidden=kwargs.get("hidden", 256),
            policy_noise=kwargs.get("policy_noise", 0.2),
            noise_clip=kwargs.get("noise_clip", 0.5),
            policy_delay=kwargs.get("policy_delay", 2),
             num_options=kwargs.get("num_options", 1),
        )

    raise ValueError(f"Unknown algo='{algo}'. Use 'ddpg' or 'td3'.")
