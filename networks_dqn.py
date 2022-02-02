import torch
import torch.nn as nn
import numpy as np

"""
---------------------------- DQN Network for Both Bidding and Playing ----------------------------
"""


class DQNNetwork(nn.Module):
    """
    This class models the DQN network which assigns a value to all possible actions the agent can take
    Being a value-based method, the agent then simply picks the highest-valued action
    """

    def __init__(self, num_inputs, num_actions, hidden, do_history_preprocessing, hps):
        """
        Fully-connected neural network
        :param num_inputs: neural input
        :param num_actions: neural output
        :param hidden: shape of the hidden layers
        """

        super(DQNNetwork, self).__init__()

        torch.manual_seed(hps['env']['SEED'])
        self.hps = hps

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], num_actions)
        )

        # HISTORIC PREPROCESSING
        self.do_history_preprocessing = do_history_preprocessing
        if do_history_preprocessing:
            self.history_preprocessing = nn.Sequential(
                nn.Linear(in_features=self.hps['hist']['HIDDEN_SIZE'],
                          out_features=self.hps['hist']['HIDDEN_SIZE']),
                nn.ReLU(),
                nn.Linear(in_features=self.hps['hist']['HIDDEN_SIZE'],
                          out_features=self.hps['hist']['HIDDEN_SIZE'])
            )

    def forward(self, x):
        regular_input = x
        history_input = torch.tensor([])
        if self.do_history_preprocessing:
            # Extract and preprocess historic input
            history_input = self.history_preprocessing(regular_input[:, -self.hps['hist']['HIDDEN_SIZE']:])
            # Remove history input from regular input
            regular_input = regular_input[:, :-self.hps['hist']['HIDDEN_SIZE']]

        combined_input = torch.cat(tensors=(regular_input, history_input), dim=1)
        return self.fc(combined_input)

    @torch.no_grad()
    def sample_action(self, obs):
        """
        BIDDING: return action with highest value (there are no restrictions in bidding)
        :param obs: one single observation
        :return: action with highest value
        """
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)
        q_values = self(obs)
        q_values = q_values.squeeze(0)
        action = torch.argmax(q_values).item()

        return action

    @torch.no_grad()
    def sample_masked_action(self, obs, action_mask):
        """
        PLAYING: return action with highest value among all admissible actions
        :param obs: one single observation
        :param action_mask: mask output of neural network
        :return:
        """
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)
        mask = torch.from_numpy(action_mask).bool()
        q_values = self(obs)
        q_values = q_values.squeeze(0)
        q_values_masked = q_values[mask]
        action_masked_index = torch.argmax(q_values_masked)
        action = torch.arange(len(mask))[mask][action_masked_index].item()

        return action

    @torch.no_grad()
    def sample_masked_action_and_value(self, obs, action_mask):
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)
        mask = torch.from_numpy(action_mask).bool()
        q_values = self(obs)
        q_values = q_values.squeeze(0)
        q_values_masked = q_values[mask]
        action_masked_index = torch.argmax(q_values_masked)
        action = torch.arange(len(mask))[mask][action_masked_index].item()

        return action, torch.max(q_values_masked).item()


    @torch.no_grad()
    def get_max_value_masked(self, obs_batch, action_mask_batch):
        """
        PLAYING: return max value of all feasible actions
        :param obs_batch: batch of observations
        :param action_mask_batch: batch of masks
        :return:
        """
        q_values = self(obs_batch)
        q_max_masked = torch.tensor(
            [torch.max(q_values[i][action_mask_batch[i]]) for i in range(len(action_mask_batch))])

        return q_max_masked

    @torch.no_grad()
    def get_max_action_masked(self, obs_batch, action_mask_batch):
        q_values = self(obs_batch)
        action_max_masked = torch.tensor([torch.arange(len(action_mask_batch[i]))
                                          [action_mask_batch[i]][torch.argmax(q_values[i][action_mask_batch[i]])]
                                          for i in range(len(action_mask_batch))])

        return action_max_masked

    @torch.no_grad()
    def get_all_q_values(self, obs):
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)
        q_values = self(obs)
        q_values = q_values.squeeze(0)

        return q_values.numpy()

    @torch.no_grad()
    def get_all_q_values_masked(self, obs, action_mask):
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)
        mask = torch.from_numpy(action_mask).bool()
        q_values = self(obs)
        q_values = q_values.squeeze(0)
        q_values_masked = q_values[mask]

        return q_values_masked.numpy()

"""
---------------------------- DQN WITH DUELING ARCHITECTURE ----------------------------
"""


class DQNDueling(DQNNetwork):

    def __init__(self, num_inputs, num_actions, hidden, do_history_preprocessing, hps):
        super(DQNDueling, self).__init__(num_inputs, num_actions, hidden, do_history_preprocessing, hps)

        torch.manual_seed(hps['env']['SEED'])
        self.hps = hps

        self.trunk = nn.Sequential(
            nn.Linear(num_inputs, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], 1)
        )

    def forward(self, x):
        regular_input = x
        history_input = torch.tensor([])
        if self.do_history_preprocessing:
            history_input = self.history_preprocessing(regular_input[:, -self.hps['hist']['HIDDEN_SIZE']:])
            regular_input = regular_input[:, :-self.hps['hist']['HIDDEN_SIZE']]
        combined_input = torch.cat(tensors=(regular_input, history_input), dim=1)

        trunk_out = self.trunk(combined_input)
        a = self.policy(trunk_out)
        v = self.value(trunk_out)
        v = v.expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q


"""
---------------------------- STATE ESTIMATOR ----------------------------
"""


class StateEstimator(DQNNetwork):
    """
    Use same architecture and forward-method as base class
    """
    def __init__(self, num_inputs, num_estimations, hidden, do_history_preprocessing, hps):
        super(StateEstimator, self).__init__(num_inputs, num_estimations, hidden, do_history_preprocessing, hps)

    @torch.no_grad()
    def get_state_prediction(self, obs):
        obs = torch.from_numpy(obs).float()
        obs = obs.unsqueeze(0)
        estimation = self(obs)
        estimation = estimation.numpy()

        return estimation
