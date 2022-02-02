import collections
import numpy as np

"""
---------------------------- Multi-purpose Experience Buffer ----------------------------
"""
BiddingExperience = collections.namedtuple('Experience', field_names=['obs', 'action', 'reward'])
PlayingExperience = collections.namedtuple(
    'Experience', field_names=['obs', 'action', 'reward', 'done', 'next_obs', 'mask', 'next_mask'])
SupervisedExperience = collections.namedtuple('Experience', field_names=['input', 'target'])


class ExperienceBuffer:
    def __init__(self, capacity, seed):
        self.buffer = collections.deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample_bidding(self, batch_size):
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        obs_batch, action_batch, reward_batch = zip(*[self.buffer[idx] for idx in indices])
        return np.array(obs_batch), np.array(action_batch), np.array(reward_batch)

    def sample_playing(self, batch_size):
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        obs_batch, action_batch, reward_batch, done_batch, next_obs_batch, mask_batch, next_mask_batch\
            = zip(*[self.buffer[idx] for idx in indices])
        return np.array(obs_batch), np.array(action_batch), np.array(reward_batch), np.array(done_batch), \
               np.array(next_obs_batch), np.array(mask_batch), np.array(next_mask_batch)

    def sample_supervised(self, batch_size):
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        input_batch, target_batch = zip(*[self.buffer[idx] for idx in indices])
        return np.array(input_batch), np.array(target_batch)


"""
---------------------------- PER = Prioritized Experience Buffer ----------------------------
PER is independent of the kind of experience
"""
BETA_START = 0.4


class PrioBuffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.beta = BETA_START

    def update_beta(self, idx, expected_updates):
        # Increase Beta up to Beta = 1
        v = BETA_START + idx * (1.0 - BETA_START) / expected_updates
        self.beta = min(1.0, v)
        return self.beta

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0

        # If buffer is not full yet, simply append experience
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        # If buffer is full, overwrite experience at current position
        else:
            self.buffer[self.pos] = experience

        # New samples always get max priority
        self.priorities[self.pos] = max_prio
        # Adjust current position
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        # Calculate Probabilities P(i) from Priorities p_i
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        # Chose a batch according to the probs
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)

        # Calculate the weights used in SGD training
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        # Update priorities after each SGD backward pass
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
