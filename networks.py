import torch
import torch.nn as nn
import torch.nn.functional as F

"""The Actor and Critic structure is for the low level, while for high level we use just a simple network called OptionValue"""

class Actor(nn.Module):
    """Actor takes an observation (state) from the enviroment and produce an action"""
    def __init__(self, obs_dim, act_dim, act_limit, hidden = 256):
        super().__init__()
        self.act_limit = float(act_limit) # in the case of pendolum act_limit is 2 since action space is between -2 and 2
        self.affine1 = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, act_dim)
    
    def forward(self, x):
        """x is a tensor (B, obs_dim), where B is the batch size"""
        x = self.affine1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = self.ln2(x)
        x = F.relu(x)
        a = torch.tanh(self.out(x)) # so now a are number in [-1, 1] but we need [-2, 2]
        return self.act_limit * a
    
class Critic(nn.Module):
    """Critic takes (obs, action) and return Q(obs, action), so how good is that action in sta state (observation)"""
    def __init__(self, obs_dim, act_dim, hidden = 256):
        super().__init__()
        self.affine1 = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.affine2 = nn.Linear(hidden + act_dim, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)
    
    def forward(self, x, a):
        """
        x is (B, obs_dim)
        a is (B, act_dim)
        """
        x = self.affine1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = torch.cat([x, a], dim = 1)
        x = self.affine2(x)
        x = self.ln2(x)
        x = F.relu(x)
        q = self.out(x)
        return q

"""Now we create e network to estimate the Q value Q(s, o) for all the options in each state"""
class OptionValue(nn.Module):
    def __init__(self, obs_dim, num_options, hidden = 256):
        super().__init__()
        self.affine1 = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, num_options)
    
    def forward(self, x):
        x = self.affine1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = self.ln2(x)
        x = F.relu(x)
        return self.out(x) # (B, num_options)