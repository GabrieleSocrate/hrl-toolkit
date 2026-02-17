import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import Actor, Critic
from utils import one_hot_option, soft_update

class DDPG:
    """
    We keep use a SINGLE actor and a SINGLE critic.
    The option influences the action by being concatenated to the state:
        s_aug = [s, one_hot(option)], this is the AUGMENTED STATE

    So both actor and critic are functions of (state, option).
    """

    def __init__(
            self,
            obs_dim,
            act_dim,
            act_limit,
            device="cpu",
            gamma=0.99,          # discount factor
            tau=0.005,
            actor_lr=1e-3,       # learning rate actor
            critic_lr=1e-3,      # learning rate critic
            hidden=256,
            num_options=0,
        ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = float(act_limit)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.num_options = int(num_options)
        if self.num_options <= 0:
            raise ValueError("MODE B requires num_options > 0")

        # augmented observation dimension: state + option
        self.obs_dim_aug = self.obs_dim + self.num_options

        self.actor = Actor(self.obs_dim_aug, act_dim, act_limit, hidden=hidden).to(self.device)
        # Actor takes (s, option) and returns a continuous action

        self.critic = Critic(self.obs_dim_aug, act_dim, hidden=hidden).to(self.device)
        # Critic takes (s, a, option) and returns Q(s, a, o)

        self.actor_targ = Actor(self.obs_dim_aug, act_dim, act_limit, hidden=hidden).to(self.device)
        self.critic_targ = Critic(self.obs_dim_aug, act_dim, hidden=hidden).to(self.device)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        # I'll not update target with backpropagation but I'll use soft_update
        # so I do not have to compute gradients for these
        for p in self.actor_targ.parameters():
            p.requires_grad = False
        for p in self.critic_targ.parameters():
            p.requires_grad = False

        # Initialization of optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # loss
        self.mse = nn.MSELoss()

    def augment_obs(self, obs, option):
        """
        Concatenate the one-hot encoded option to the state.
        """
        if option is None:
            raise ValueError("Error: there are no options")

        if isinstance(obs, np.ndarray):
            opt_oh = one_hot_option(int(option), self.num_options) # (K,)
            return np.concatenate([obs, opt_oh], axis=0) # (obs_dim + K,)
        else:
            opt_oh = torch.as_tensor(
                one_hot_option(int(option), self.num_options), # (K,)
                dtype=torch.float32,
                device=obs.device
            )
            return torch.cat([obs, opt_oh], dim=0) # (obs_dim + K,)

    def act(self, obs, noise_std=0.1, option=None):
        """
        Choose an action using the actor.
        The action depends on both the state and the current option.
        """
        with torch.no_grad():  # we are just choosing an action so no gradients are required
            obs = self.augment_obs(obs, option)

            if isinstance(obs, np.ndarray):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                obs_t = obs.to(self.device).float().unsqueeze(0)

            a = self.actor(obs_t).squeeze(0)  # action chosen by actor

            if noise_std > 0:
                a = a + noise_std * torch.randn_like(a)  # Adding some gaussian noise for exploration

            a = torch.clamp(a, -self.act_limit, self.act_limit)
            return a.cpu().numpy()

    def update(self, replay_buffer, batch_size=256, update_iteration=1):
        """
        Updates actor and critic using a batch of samples from the replay buffer.

        For each update step:
        - sample (state, next_state, action, reward, done, option, terminated)
        - build augmented states [s, option]
        - build target Q using target critic + target actor
        - optimize critic with MSE(Q(s,a,o), target_Q)
        - optimize actor by maximizing Q(s, a, o) (minimize - Q)
        - soft-update target networks

        Returns:
        (critic_loss_mean, actor_loss_mean)
        """

        policy_loss = []
        value_loss = []

        for it in range(update_iteration):
            # sample from buffer
            state, next_state, action, reward, done, option, _ = replay_buffer.sample(batch_size) # _ is for terminated that I don't need now
            """Shapes returned by ReplayBuffer.sample (with batch_size = B):
               state:      (B, obs_dim)
               next_state: (B, obs_dim)
               action:     (B, act_dim)
               reward:     (B, 1)
               done:       (B, 1)
               option:     (B,)          
               _:          (B, 1) terminated (not used here)"""
            
            if option is None:
                raise ValueError("Error: there are no options")
            
            # build one-hot options opt_oh: (B, K)
            opt_oh = np.stack(
                [one_hot_option(int(o), self.num_options) for o in option],
                axis=0
            ) # (B, K)

            """For each element o_i in option (i=1..B), one_hot_option(o_i, K) returns an array of shape (K,)
            np.stack stacks B vectors (K,) into a matrix (B, K)
            So now:
            state:   (B, obs_dim)
            opt_oh:  (B, K)
            and we concatenate them to obtain the augmented state
            -> state_aug: (B, obs_dim + K)"""

            state_aug = np.concatenate([state, opt_oh], axis=1) # (B, obs_dim + K)
            next_state_aug = np.concatenate([next_state, opt_oh], axis=1) # (B, obs_dim + K)

            # to torch tensors
            state = torch.as_tensor(state_aug, dtype=torch.float32, device=self.device) # (B, obs_dim + K)
            next_state = torch.as_tensor(next_state_aug, dtype=torch.float32, device=self.device) # (B, obs_dim + K)
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device) # (B, act_dim)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device) # (B, 1)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device) # (B, 1)

            with torch.no_grad():
                next_action = self.actor_targ(next_state) # action in the next state obtained using actor target
                target_Q_next = self.critic_targ(next_state, next_action) # use critic target to estimate the Q' value of the pair (next_state, next_action)
                target_Q = reward + self.gamma * target_Q_next # I compute target Q using Bellman equation
                # DA MODIFICARE

            # current Q estimate
            current_Q = self.critic(state, action)

            # critic loss
            critic_loss = self.mse(current_Q, target_Q)
            value_loss.append(float(critic_loss.item()))

            # optimize critic
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # actor loss (maximize Q)
            actor_loss = -self.critic(state, self.actor(state)).mean()
            policy_loss.append(float(actor_loss.item()))

            # optimize actor
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            """
            The critic learns Q(s,a,o): how good action a is in state s under option o.
            We train the critic with TD-learning using a Bellman target:
                y = r + gamma*(1-done)*Q_target(s', pi_target(s',o), o)

            The actor does not fit a target value: it chooses actions that maximize Q.
            So we minimize:
            So we define the actor loss as the negative expected Q: L_actor = -E[ Q(s, pi(s,o), o) ]
            Minimizing this loss is the same as maximizing Q
            """

            # soft update targets
            soft_update(self.critic, self.critic_targ, self.tau)
            soft_update(self.actor, self.actor_targ, self.tau)

        # if you do multiple update_iteration (update_iteration > 1), return average losses 
        critic_loss_mean = float(np.mean(value_loss)) if len(value_loss) > 0 else 0.0
        actor_loss_mean = float(np.mean(policy_loss)) if len(policy_loss) > 0 else 0.0

        return critic_loss_mean, actor_loss_mean
