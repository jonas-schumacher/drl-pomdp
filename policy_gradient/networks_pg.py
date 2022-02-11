import torch
import torch.nn as nn
from torch.distributions import Categorical


class A2CCritic(nn.Module):
    """
    Critic which approximates the value of the input observation
    """
    def __init__(self, num_inputs, hidden, hps):
        super(A2CCritic, self).__init__()

        torch.manual_seed(hps['env']['SEED'])

        self.value = nn.Sequential(
            nn.Linear(num_inputs, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.value(x)


class A2CActor(nn.Module):
    """
    Actor class which decides on the action to take
    Being a policy-gradient method, the agent chooses the action with according to the probabilities
    """

    def __init__(self, num_inputs, num_actions, hidden, hps):
        super(A2CActor, self).__init__()

        torch.manual_seed(hps['env']['SEED'])

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], num_actions),
        )

    def forward(self, x):
        return self.layers(x)

    @torch.no_grad()
    def sample_action(self, obs):
        """
        This is for (unrestricted) bidding
        """
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)

        logits = self(obs).squeeze(0)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        return action.item()

    @torch.no_grad()
    def sample_masked_action(self, obs, action_mask):
        """
        This is for (restricted) playing
        """
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)

        mask = torch.from_numpy(action_mask).bool()
        logits = self(obs).squeeze(0)

        probs_raw_masked = logits[mask]
        action_dist = Categorical(logits=probs_raw_masked)
        if torch.max(torch.abs(logits)) > 70:
            print(logits)
            # print(action_dist.probs)
            # print(action_dist.entropy())
        action_masked_index = action_dist.sample()
        action = torch.arange(len(mask))[mask][action_masked_index]

        return action.item()
