# hrl/option_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from option_policies import OptionPolicy, Termination


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

        # We have two different networks, one to choose options and one for the termination
        self.option_policy = OptionPolicy(obs_dim, num_options, hidden=hidden).to(self.device)
        self.termination = Termination(obs_dim, num_options, hidden=hidden).to(self.device)

        for p in self.option_policy.parameters():
            p.requires_grad = False
        for p in self.termination.parameters():
            p.requires_grad = False

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
        Sample an option using pi(o|s) (categorical from softmax logits) with optional epsilon-greedy.
        if greedy=True then is argmax over options 
        """
        with torch.no_grad():
            s = self.obs_to_torch(obs)  # (1, obs_dim)
            logits = self.option_policy(s)  # (1, K)

            if greedy:
                o = int(torch.argmax(logits, dim=1).item())
                return o

            # epsilon-greedy on options
            if self.eps_option > 0.0 and np.random.rand() < self.eps_option:
                return int(np.random.randint(self.num_options))

            probs = torch.softmax(logits, dim=1)  # (1,K) from logits we obtain probabilities
            o = torch.multinomial(probs, num_samples=1)  # (1,1) 
            """Option selection:
            The high-level policy defines a discrete distribution over K options:
            π(o | s),  with  o ∈ {0, ..., K−1}.
            At each decision point, exactly ONE option must be chosen.
            This corresponds to a categorical random variable, obtained by applying
            a softmax to the logits and sampling one index.
            In practice, this is implemented via torch.multinomial(probs, num_samples=1),
            which performs a categorical (multinomial with one trial) draw."""

            return int(o.item())

    
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
        """PER ORA LA DECISIONE DELL'AZIONE LOW LEVEL NON DIPENDE DALL'OPZIONE IN CUI MI TROVO"""
        action = self.low_level.act(obs, noise_std=noise_std)

        # Track option duration
        self.option_steps += 1

        return action, int(self.current_option), did_terminate

    def update(self, replay_buffer, batch_size=256, update_iteration=1):
        """DA FINIRE UNA VOLTA CHE SI FA IL TRAIN DELLA TERMINATION FUNCTION E POLICY SU OPZIONI"""
        """
        For now we only train the low-level agent (DDPG/TD3).
        The replay buffer may already store option, but low-level ignores it.

        Returns:
            Whatever low_level.update returns (typically critic_loss, actor_loss, ...)
        """
        return self.low_level.update(
            replay_buffer,
            batch_size=batch_size,
            update_iteration=update_iteration
        )

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