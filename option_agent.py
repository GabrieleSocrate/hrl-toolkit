import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import soft_update

from option_policies import Termination
from networks import OptionValue
# These two are two networks one for termination function and one for high level

class OptionAgent:
    """
    Here we find the logic of HRL above the continuous RL in low level (DDPG/TD3)
    So choice of a discrete option using a policy for options and deciding when to change the option using the termination function
    """

    def __init__(
        self,
        obs_dim,
        num_options,
        low_level_agent,
        device = "cpu",
        tau = 0.005,
        hidden = 256,
        eps_option = 0.0,
        terminate_deterministic = False,
        min_option_steps = 1
    ):
        """
        obs_dim: state dimension
        num_options: number of discrete options K
        low_level_agent: instance of DDPG or TD3 (already created)
        eps_option: optional epsilon-greedy on option selection 
        terminate_deterministic: if True, terminate when beta>0.5 instead of sampling Bernoulli
        """
        self.obs_dim = obs_dim
        self.num_options = num_options
        self.low_level = low_level_agent
        self.device = torch.device(device)
        self.tau = tau

        # We have two different networks, one to choose options and one for the termination
        """The OptionValue network estimates how good it is to pick option o in state s.
        We'll train it with TD targets"""
        self.option_value = OptionValue(obs_dim, num_options, hidden = hidden).to(self.device)
        
        # Target network: is a slowly updated copy used only to build stable targets
        self.option_value_targ = OptionValue(obs_dim, num_options, hidden = hidden).to(self.device)
        self.option_value_targ.load_state_dict(self.option_value.state_dict())

        for p in self.option_value_targ.parameters():
            p.requires_grad = False
        
        # We initialize the optimizer
        self.option_value_opt = torch.optim.Adam(self.option_value.parameters(), lr = 1e-3)

        # Loss
        self.mse = nn.MSELoss()
        
        """The Termination network outputs beta(s, o) that are probabilities (in [0, 1]) that the option o terminates in state s"""
        self.termination = Termination(obs_dim, num_options, hidden=hidden).to(self.device)

        # We initialize the optimizer
        self.termination_opt = torch.optim.Adam(self.termination.parameters(), lr = 1e-4)

        # DELIBERATION COST
        # Adding +c inside the termination update discourages switching frequently 
        self.delib_cost = 0.01 # LATER WE SHOULD TUNE IT

        self.eps_option = float(eps_option)
        self.terminate_deterministic = bool(terminate_deterministic)

        # Current option state 
        self.current_option = None
        self.option_steps = 0  # how many env steps since the current option was selected

        self.num_terminations = 0
        self.num_option_switches = 0

        self.min_option_steps = min_option_steps

    def obs_to_torch(self, obs): 
        """
        Convert a single observation (np array) to a torch tensor shaped (1, obs_dim).
        """
        if isinstance(obs, np.ndarray):
            s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            # if user passes a torch tensor
            s = obs.to(self.device).float().unsqueeze(0)
        return s

    def select_option(self, obs, greedy = False):
        """
        To select the option we use Harb-style selecting the option with the highest Q-value Q(s, o) associated
        If greedy = True: we use argmax over Q(s, o) and select o associated with the highest
        Else epsilon-greedy on Q-values
        """
        with torch.no_grad():
            s = self.obs_to_torch(obs)  # (1, obs_dim)
            q = self.option_value(s)  # (1, K)

            if greedy:
                o = int(torch.argmax(q).item())
                return o

            # epsilon-greedy on options
            if self.eps_option > 0.0 and np.random.rand() < self.eps_option: # this is for epsilon greedy but when we explore
                return int(np.random.randint(self.num_options))

            return int(torch.argmax(q).item()) # This is for epsilon greedy but when we exploit 

    
    def should_terminate(self, obs, option):
        """
        Decide whether to terminate current option using beta(s,o).
        If terminate_deterministic is False:
            sample terminate ~ Bernoulli(beta)
        else:
            terminate = (beta > 0.5)
        """
        with torch.no_grad():
            s = self.obs_to_torch(obs)  # (1, obs_dim)
            opt = torch.tensor([option], dtype=torch.int64, device=self.device)  # (1,)

            beta = self.termination.beta(s, opt)  # (1,1) 
            b = float(beta.item())

            if self.terminate_deterministic:
                return b > 0.5
            else:
                # sample Bernoulli(beta)
                term = torch.bernoulli(beta).item()
                """Option termination:
                The termination function β(s, o) ∈ [0, 1] defines the probability that
                the currently active option o terminates in state s.
                Termination is therefore a binary random event:
                terminate ∈ {0, 1}, naturally modeled as a Bernoulli random variable"""

                return bool(term)

    def reset(self, obs=None):
        """
        Call this at the beginning of each episode.
        If obs is given, we immediately pick an option based on obs.
        """
        self.option_steps = 0
        self.current_option = None

        if obs is not None:
            self.current_option = self.select_option(obs, greedy=False)
            self.num_option_switches += 1

    def act(self, obs, noise_std=None, greedy_option = False):
        """
        Choose action from low-level agent, but also considering option termination.

        Returns:
            action: np.ndarray (act_dim,)
            option: int (current option after possible termination)
            did_terminate: bool (True if beta terminated the previous option at this step)
        """
        # If first step of episode, select an option
        if self.current_option is None:
            self.current_option = self.select_option(obs, greedy=greedy_option)
            self.num_option_switches += 1
            self.option_steps = 0

        did_terminate = False

        can_terminate = self.option_steps >= self.min_option_steps # allow termination only after min_option_steps

        # Check termination BEFORE selecting low-level action 
        if can_terminate and self.should_terminate(obs, self.current_option): # it's true if the option terminated and if the option lasted long enough
            did_terminate = True 
            self.num_terminations += 1

            # pick a new option
            self.current_option = self.select_option(obs, greedy=greedy_option)
            self.num_option_switches += 1
            self.option_steps = 0

        # Low-level action 
        action = self.low_level.act(obs, noise_std=noise_std, option = self.current_option)

        # Track option duration
        self.option_steps += 1

        return action, int(self.current_option), did_terminate

    def update(self, replay_buffer, batch_size=256, update_iteration=1):
        """
        1) We train high-level OptionValue QΩ(s, o) with TD targets
           The TD target depends on whether the option terminated after the transition

            - If the option did NOT terminate, next value continues with the SAME option: Q(s', o)
            - If the option terminated, next value switches to the BEST next option: max_o' Q(s', o')

            We then keep training the low-level controller (TD3/DDPG) exactly as always.
        
        2) We also train the Termination using the term: mean( beta(s',o) * (A(s',o) + delib_cost) )
           where A(s',o) = Q(s',o) - V(s') and V(s') = max_o' Q(s',o').
           

        Returns:
        low_level_out: whatever TD3/DDPG returns (critic loss, etc.)
        optv_loss_mean: mean OptionValue loss over update_iteration steps
        term_loss_mean: mean loss used to update termination beta
        """

        value_loss = []
        term_losses = []

        for it in range(update_iteration):
            # Sample from replay buffer 
            state, next_state, action, reward, done, option, terminated = replay_buffer.sample(batch_size)

            """Shapes returned by ReplayBuffer.sample (with batch_size = B):
               state:      (B, obs_dim)
               next_state: (B, obs_dim)
               action:     (B, act_dim)      not used for OptionValue
               reward:     (B, 1)
               done:       (B, 1)
               option:     (B,)              option index for each transition
               terminated: (B, 1)            1 if option ended after the step, else 0
            """

            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)            # (B, obs_dim)
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)  # (B, obs_dim)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)          # (B, 1)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device)              # (B, 1)
            terminated = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)  # (B, 1)
            option = torch.as_tensor(option, dtype=torch.int64, device=self.device)            # (B,)

            B = state.shape[0]
            idx = torch.arange(B, device=self.device)

            # Compute target Q-values at next state using Target OptionValue net
            with torch.no_grad():
                # Q-values for all options at next state: (B, K)
                target_Q_next_all = self.option_value_targ(next_state)
                
                # Keep the same option so Q(s', o)
                target_Q_next_same = target_Q_next_all[idx, option].unsqueeze(1) # (B, 1)
                
                # Option ended so choose the best option max_o'(Q(s', o'))
                target_Q_next_max = target_Q_next_all.max(dim = 1, keepdim = True).values # (B, 1)
            
            """
            Term: mean( beta(s',o) * ( (Q(s',o) - V(s')) + delib_cost ) )
            - Q(s',o) - V(s') is an advantage-like signal:
                negative -> staying with current option is worse than switching -> encourage termination
                positive -> staying is not worse -> discourage termination

            - delib_cost shifts the decision to avoid too frequent switching.
            """
            adv_next = (target_Q_next_same - target_Q_next_max).detach() # (B, 1)

            # beta(s', o) is the termination probability predicted by the termination network
            beta_next = self.termination.beta(next_state, option) # (B, 1) in [0, 1]

            term_loss = (beta_next * (adv_next + self.delib_cost)).mean()
            term_losses.append(float(term_loss.item()))

            self.termination_opt.zero_grad()
            term_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.termination.parameters(), max_norm=1.0)
            self.termination_opt.step()
            
            with torch.no_grad():

                """
                Combine the two cases using the terminated flag as a switch:
                - terminated = 0, keep same option
                - terminated = 1, switch to max option
                """
                target_Q_next = (1.0 - terminated) * target_Q_next_same + terminated * target_Q_next_max

                # Standard TD target with episode termination mask (1-done)
                target_Q = reward + self.low_level.gamma * (1.0 - done) * target_Q_next  # (B, 1)

            # option_value(state) returns Q(s, o) for all o 
            current_Q_all = self.option_value(state)
            
            current_Q = current_Q_all[idx, option].unsqueeze(1)  # select the option actually used in the transition (B,1)

            # Make current_Q match the target_Q (exactly like critic training in TD3/DDPG).
            optv_loss = self.mse(current_Q, target_Q)
            value_loss.append(float(optv_loss.item()))

            self.option_value_opt.zero_grad()
            optv_loss.backward()
            self.option_value_opt.step()

            soft_update(self.option_value, self.option_value_targ, self.tau)

        # Mean high-level and termination losses (same pattern as TD3)
        optv_loss_mean = float(np.mean(value_loss)) if len(value_loss) > 0 else 0.0
        term_loss_mean = float(np.mean(term_losses)) if len(term_losses) > 0 else 0.0

        # Low-level update (unchanged, as requested)
    
        low_level_out = self.low_level.update(
            replay_buffer,
            batch_size=batch_size,
            update_iteration=update_iteration
        )
        return low_level_out, optv_loss_mean, term_loss_mean
    
    def get_stats(self):
        """
        This function is just to monitoring what is going on
        """
        return {
            "current_option": self.current_option,
            "option_steps": self.option_steps,
            "num_terminations": self.num_terminations,
            "num_option_switches": self.num_option_switches,
        }