import torch 
import torch.nn as nn
import torch.nn.functional as F
# INCLUDI QUESTO FILE IN NETWORKS

class OptionPolicy(nn.Module):
    """This is the high level policy over options pi(o|s)
    The output will be logits over K options"""
    def __init__(self, obs_dim, num_options, hidden = 256):
        super().__init__()
        self.affine1 = nn.Linear(obs_dim, hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, num_options)

    def forward(self, s):
        x = F.relu(self.affine1(s))
        x = F.relu(self.affine2(x))
        return self.logits(x)
    """the input is a tensor of states (B, obs_dim) and the output is (B, K) so for each batch you obtain a vector of K numbers,
    the higher the number the more that option is preferred"""

class Termination(nn.Module):
    """This is the termination function beta(s, o): probability of terminating option o in state s
    The output will be K logits so each logit is how likely the respective option will end"""
    def __init__(self, obs_dim, num_options, hidden = 256):
        super().__init__()
        self.affine1 = nn.Linear(obs_dim, hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, num_options)

    def forward(self, s):
        x = F.relu(self.affine1(s))
        x = F.relu(self.affine2(x))
        return self.logits(x) # the output is (B, K)
    
    def beta(self, s, option):
        logits = self.forward(s) # (B, K) terminations for all the options
        # For each batch row i, we want the logit at column option[i]
        B = logits.size(0)
        rows = torch.arange(B, device=logits.device)   
        opt_logits = logits[rows, option]              
        opt_logits = opt_logits.unsqueeze(1)           

        return torch.sigmoid(opt_logits) # in this way from logits we get probability (if logits is big then Beta close to 1 and viceversa)