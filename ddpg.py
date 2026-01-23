import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import Actor, Critic

def soft_update(net, target_net, tau):
    with torch.no_grad():
        for p, p_targ in zip(net.parameters(), target_net.parameters()):
            p_targ.data.copy_((1.0 - tau) * p_targ.data + tau * p.data)

class DDPG:
    def __init__(
            self,
            obs_dim,
            act_dim,
            act_limit,             
            device = "cpu",       
            gamma = 0.99,          # discount factor
            tau = 0.005,           
            actor_lr = 1e-3,       # learning rate actor
            critic_lr = 1e-3,      # learning rate critic
            hidden = 256,           
        ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = float(act_limit)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, act_dim, act_limit, hidden = hidden).to(self.device)
        # Actor takes s and returns a continuous action
        self.critic = Critic(obs_dim, act_dim, hidden = hidden).to(self.device)
        # Critic takes (s, a) and returns Q(s, a)

        self.actor_targ = Actor(obs_dim, act_dim, act_limit, hidden = hidden).to(self.device)
        self.critic_targ = Critic(obs_dim, act_dim, hidden = hidden).to(self.device)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        # I'll not update target with backpropagation but I'll use soft_update so do not have to compute gradients for these
        for p in self.actor_targ.parameters():
            p.requires_grad = False
        for p in self.critic_targ.parameters():
            p.requires_grad = False
        
        # Inizialization of optimaizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr = critic_lr)

        # loss
        self.mse = nn.MSELoss()

    def act(self, obs, noise_std = 0.1):
        with torch.no_grad(): # we are just choosing an action so no gradients are required
            if isinstance(obs, np.ndarray): # if obs is an array I convert it into a tensor
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                obs_t = obs.to(self.device).float().unsqueeze(0) # take the obs to device

            a = self.actor(obs_t).squeeze(0) # action choosen by actor

            if noise_std > 0:
                a = a + noise_std * torch.randn_like(a) # I add some gaussian noise for exploration

            a = torch.clamp(a, -self.act_limit, self.act_limit)
            return a.cpu().numpy()
    
    def update(self, replay_buffer, batch_size = 256, update_iteration = 1):
        """
        Updates actor and critic using a batch of samples from the replay buffer.

        For each update step:
        - sample (state, next_state, action, reward, done)
        - build target Q using target critic + target actor
        - optimize critic with MSE(Q(s,a), target_Q)
        - optimize actor by maximizing Q(s, actor(s))  (=> minimize -Q)
        - soft-update target networks

        Returns:
        (critic_loss_mean, actor_loss_mean) as python floats.
        """

        policy_loss = []
        value_loss = []

        for it in range(update_iteration):
            # sample from buffer 
            state, next_state, action, reward, done, _ = replay_buffer.sample(batch_size) # _ is for the option that I don't need now

            # to torch tensors 
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device) # (B, obs_dim)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device) # (B, obs_dim)
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device) # # (B, act_dim)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)  # (B,1)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device)      # (B,1) 1 if episode ended

            with torch.no_grad():
                next_action = self.actor_targ(next_state) # action in the next state obtained using actor target
                target_Q_next = self.critic_targ(next_state, next_action) # use critic target to estimate the Q' value of the pair (next_state, next_action)
                target_Q = reward + self.gamma * (1.0 - done) * target_Q_next # I compute target Q using Bellman equation

            # current Q estimate 
            current_Q = self.critic(state, action)

            # critic loss 
            critic_loss = self.mse(current_Q, target_Q) # i want to make current_Q closer to target_Q
            value_loss.append(float(critic_loss.item()))

            # optimize critic 
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # actor loss (maximize Q) 
            actor_loss = -self.critic(state, self.actor(state)).mean() # pytorch minimize so I put - (maximization problem)
            policy_loss.append(float(actor_loss.item()))

            # optimize actor 
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            """The critic learns to approximate Q(s,a): how good an action a is in a state s.
            We train the critic with TD-learning: we build a Bellman target
            y = r + gamma*(1-done)*Q_target(s', pi_target(s')) and minimize MSE( Q(s,a), y ).
            The actor does NOT fit a target value: its goal is to choose actions that maximize Q.
            So we define the actor loss as the negative expected Q: L_actor = -mean( Q(s, actor(s)) ).
            Minimizing this loss is the same as maximizing Q"""

            # soft update targets 
            soft_update(self.critic, self.critic_targ, self.tau)
            soft_update(self.actor, self.actor_targ, self.tau)

        # if you do multiple update_iteration (update_iteration > 1), return average losses 
        critic_loss_mean = float(np.mean(value_loss)) if len(value_loss) > 0 else 0.0
        actor_loss_mean = float(np.mean(policy_loss)) if len(policy_loss) > 0 else 0.0

        return critic_loss_mean, actor_loss_mean