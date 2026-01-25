import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import Actor, Critic


def soft_update(net, target_net, tau):
    with torch.no_grad():
        for p, p_targ in zip(net.parameters(), target_net.parameters()):
            p_targ.data.copy_((1.0 - tau) * p_targ.data + tau * p.data)


class TD3:
    """key differences with DDPG:
     1) Two critics(Q1, Q2): in TD3 we have two critics (Q1, Q2) and take min(Q1_target, Q2_target)
     2) Delayed policy update (the D in TD3): the critics are updated at every training step, but the actor
     is updated only once every policy_delay critic updates. The reason is that the actor update depends on 
     critic gradients; updating the actor too frequently while critics are still inaccurate/noisy can destabilize training.
     So TD3 lets the critics improve for a few steps before moving the actor
     3) Target policy smoothing: the target action a' used in the TD target is perturbed with small clipped noise"""
    
    def __init__(
            self,
            obs_dim,
            act_dim,
            act_limit,
            device = "cpu",
            gamma = 0.99,
            tau = 0.005,
            actor_lr = 1e-3,
            critic_lr = 1e-3,
            hidden = 256,
            policy_noise = 0.2,
            noise_clip = 0.5,
            policy_delay = 2
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = float(act_limit)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.actor = Actor(obs_dim, act_dim, act_limit, hidden=hidden).to(self.device)
        self.actor_targ = Actor(obs_dim, act_dim, act_limit, hidden=hidden).to(self.device)

        self.actor_targ.load_state_dict(self.actor.state_dict())

        for p in self.actor_targ.parameters():
            p.requires_grad = False

        # TD3 uses TWO critics to reduce overestimation bias:
        # in the target we take min(Q1, Q2).
        self.critic1 = Critic(obs_dim, act_dim, hidden=hidden).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, hidden=hidden).to(self.device)

        self.critic1_targ = Critic(obs_dim, act_dim, hidden=hidden).to(self.device)
        self.critic2_targ = Critic(obs_dim, act_dim, hidden=hidden).to(self.device)

        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_targ.parameters():
            p.requires_grad = False
        for p in self.critic2_targ.parameters():
            p.requires_grad = False

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.mse = nn.MSELoss()

        self._n_critic_updates = 0

    def act(self, obs, noise_std):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                obs_t = obs.to(self.device).float().unsqueeze(0)

            a = self.actor(obs_t).squeeze(0)

            if noise_std is not None and noise_std > 0:
                a = a + float(noise_std) * torch.randn_like(a)

            a = torch.clamp(a, -self.act_limit, self.act_limit)

            return a.cpu().numpy()
    
    def update(self, replay_buffer, batch_size = 256, update_iteration = 1):
         
        policy_loss = []
        value_loss = []
        did_actor_update = False # a flag to see if the actor has been updated or not in this iteration

        for it in range(update_iteration):
            state, next_state, action, reward, done, _, _ = replay_buffer.sample(batch_size)  # _ is option and terminated that are useless here

            # to torch tensors
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)            # (B, obs_dim)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)  # (B, obs_dim)
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)         # (B, act_dim)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)         # (B,1) or (B,)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device)             # (B,1) or (B,)

            with torch.no_grad():
                next_action = self.actor_targ(next_state)

                # TD3 difference: target policy smoothing 
                # Add small clipped noise 
                noise = self.policy_noise * torch.randn_like(next_action)
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_action = next_action + noise
                next_action = torch.clamp(next_action, -self.act_limit, self.act_limit)

                # TD3 difference: double Q target
                # target_Q_next = min(Q1_target(s',a'), Q2_target(s',a'))
                target_Q_next1 = self.critic1_targ(next_state, next_action)
                target_Q_next2 = self.critic2_targ(next_state, next_action)
                target_Q_next = torch.min(target_Q_next1, target_Q_next2)

                # Bellman target 
                target_Q = reward + self.gamma * (1.0 - done) * target_Q_next

            # current Q estimates (two critics)
            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)

            # critic losses (optimized separately)
            critic1_loss = self.mse(current_Q1, target_Q)
            critic2_loss = self.mse(current_Q2, target_Q)
            critic_loss = critic1_loss + critic2_loss

            value_loss.append(float(critic_loss.item()))

            # optimize critic 1
            self.critic1_opt.zero_grad()
            critic1_loss.backward()
            self.critic1_opt.step()

            # optimize critic 2
            self.critic2_opt.zero_grad()
            critic2_loss.backward()
            self.critic2_opt.step()

            # TD3 difference: delayed actor 
            # Count critic updates; update actor only every policy_delay steps.
            self._n_critic_updates += 1

            if self._n_critic_updates % self.policy_delay == 0:
                # actor loss 
                actor_loss = -self.critic1(state, self.actor(state)).mean()
                policy_loss.append(float(actor_loss.item()))
                did_actor_update = True

                # optimize actor
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                # soft update targets 
                soft_update(self.critic1, self.critic1_targ, self.tau)
                soft_update(self.critic2, self.critic2_targ, self.tau)
                soft_update(self.actor, self.actor_targ, self.tau)

        critic_loss_mean = float(np.mean(value_loss)) if len(value_loss) > 0 else 0.0
        actor_loss_mean = float(np.mean(policy_loss)) if len(policy_loss) > 0 else None

        return critic_loss_mean, actor_loss_mean, did_actor_update