import numpy as np

capacity = 1000000 # max size of the replay buffer (how many transactions you want to keep in memory)
class ReplayBuffer():
    def __init__(self, max_size = capacity):
        self.storage = [] # every element (tuple) will be a transaction
        self.max_size = max_size
        self.ptr = 0 # pointer that points where to overwrite when the buffer is full
    
    def push(self, obs, action, reward, next_obs, done, option = None):
        """
        Store a transition.
        - obs: np.array shape (obs_dim,)
        - action: np.array shape (act_dim,)  (continuous)
        - reward: float
        - next_obs: np.array shape (obs_dim,)
        - done: float or bool (1.0 if episode ended else 0.0)
        - option: optional int (for HRL/Option-Critic style buffers)
        """
        data = (
        np.asarray(obs, dtype=np.float32),
        np.asarray(next_obs, dtype=np.float32),
        np.asarray(action, dtype=np.float32),
        float(reward),
        float(done),
        option,
        )

        if len(self.storage) == self.max_size: # if buffer is full
            self.storage[self.ptr] = data # overwrite the transaction at position ptr
            self.ptr = (self.ptr + 1) % self.max_size # increase ptr by one and % max_size move it back to 0 when ptr = last element of buffer
        else:
            self.storage.append(data)
    
    def sample(self, batch_size):
        """
        Takes a random batch of transactions from buffer
        Shapes:
          state:      (B, obs_dim)
          next_state: (B, obs_dim)
          action:     (B, act_dim)
          reward:     (B, 1)
          done:       (B, 1)
          option:     (B,) int64 or None
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)

        state, next_state, action, reward, done, option = [], [], [], [], [], []

        for i in ind:
            st, n_st, act, rew, dn, opt = self.storage[i]
            state.append(st)
            next_state.append(n_st)
            action.append(act)
            reward.append(rew)
            done.append(dn)
            option.append(-1 if opt is None else int(opt))

        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32).reshape(-1, 1)
        done = np.array(done, dtype=np.float32).reshape(-1, 1)

        option = np.array(option, dtype=np.int64)
        if np.all(option == -1):
            option = None

        return state, next_state, action, reward, done, option

    def __len__(self,):
        return len(self.storage)