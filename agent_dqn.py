import collections

import numpy as np
from scipy.special import softmax
import torch

from agent import OptimizedPlayer, copy_weights, calc_grad, GamePhase, AgentMode, calc_weight
from simulation import Simulation
from networks_dqn import DQNNetwork, DQNDueling, StateEstimator
from networks_history import HistoryNet
from replay_buffer import ExperienceBuffer, BiddingExperience, PlayingExperience, SupervisedExperience, PrioBuffer


def epsilon_decay_schedule(decay_type, total_steps, init_epsilon, min_epsilon, decay_share):
    """
    Calculate epsilon schedule in advance
    Base case: constant epsilon
    :param decay_type:
    :param total_steps:
    :param init_epsilon:
    :param min_epsilon:
    :param decay_share:
    :return: epsilon schedule
    """
    decay_steps = int(decay_share * total_steps)

    epsilons = np.full(shape=total_steps, fill_value=min_epsilon)
    if decay_type == "linear":
        epsilons = 1 - np.arange(total_steps) / decay_steps
        epsilons = min_epsilon + (init_epsilon - min_epsilon) * epsilons
        epsilons = np.clip(epsilons, min_epsilon, init_epsilon)
    if decay_type == "exponential":
        decay = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        decay = min_epsilon + (init_epsilon - min_epsilon) * decay
        epsilons[:decay_steps] = decay
    if decay_type == "exponential_skip":
        skip_share = 0.2
        decay_steps_skip = int(skip_share * total_steps)
        epsilons[:decay_steps_skip] = init_epsilon
        decay = 0.01 / np.logspace(-2, 0, decay_steps - decay_steps_skip, endpoint=False) - 0.01
        decay = min_epsilon + (init_epsilon - min_epsilon) * decay
        epsilons[decay_steps_skip:decay_steps] = decay

    return epsilons


class DQNPlayer(OptimizedPlayer):
    """
    This player uses Deep Q-Learning (DQN, value-based) for both bidding and playing
    Various features from Rainbow DQN can optionally be added
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps)

        torch.manual_seed(hps['env']['SEED'])
        self.master = master

        # Only the master player creates networks, replay buffers and epsilon schedules
        if self.master is None:

            """
            Optionally extend the input size with (pretrained) historic information
            """
            if self.hps['agent']['HISTORY_PREPROCESSING']:
                self.input_size_playing_ext = self.input_size_playing + self.hps['hist']['HIDDEN_SIZE']
                self.history_net = HistoryNet(
                    input_dim=self.hps['hist']['INPUT'],
                    hidden_dim=self.hps['hist']['HIDDEN_SIZE'],
                    output_dim=self.hps['hist']['OUTPUT'],
                    hps=self.hps)
                self.history_net.load_state_dict(torch.load(self.hps['hist']['CHECKPOINT'] + ".pt")['actor'])
                self.hidden_h = torch.zeros(1, 1, self.hps['hist']['HIDDEN_SIZE'])
                self.hidden_c = torch.zeros(1, 1, self.hps['hist']['HIDDEN_SIZE'])
            else:
                self.input_size_playing_ext = self.input_size_playing

            if self.hps['dqn']['DUELING']:
                self.bidding_actor = DQNDueling(num_inputs=self.input_size_bidding,
                                                num_actions=self.output_size_bidding,
                                                hidden=self.hps['dqn']['HIDDEN_BIDDING'],
                                                do_history_preprocessing=False,
                                                hps=self.hps)
                self.playing_actor = DQNDueling(num_inputs=self.input_size_playing_ext,
                                                num_actions=self.output_size_playing,
                                                hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                                do_history_preprocessing=self.hps['agent']['HISTORY_PREPROCESSING'],
                                                hps=self.hps)
                self.playing_actor_target = DQNDueling(num_inputs=self.input_size_playing_ext,
                                                       num_actions=self.output_size_playing,
                                                       hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                                       do_history_preprocessing=self.hps['agent'][
                                                           'HISTORY_PREPROCESSING'],
                                                       hps=self.hps)
            else:
                self.bidding_actor = DQNNetwork(num_inputs=self.input_size_bidding,
                                                num_actions=self.output_size_bidding,
                                                hidden=self.hps['dqn']['HIDDEN_BIDDING'],
                                                do_history_preprocessing=False,
                                                hps=self.hps)
                self.playing_actor = DQNNetwork(num_inputs=self.input_size_playing_ext,
                                                num_actions=self.output_size_playing,
                                                hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                                do_history_preprocessing=self.hps['agent']['HISTORY_PREPROCESSING'],
                                                hps=self.hps)
                self.playing_actor_target = DQNNetwork(num_inputs=self.input_size_playing_ext,
                                                       num_actions=self.output_size_playing,
                                                       hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                                       do_history_preprocessing=self.hps['agent'][
                                                           'HISTORY_PREPROCESSING'],
                                                       hps=self.hps)

            self.bidding_actor_opt = torch.optim.Adam(self.bidding_actor.parameters(), lr=self.hps['dqn']['LR_BIDDING'])
            self.playing_actor_opt = torch.optim.Adam(self.playing_actor.parameters(), lr=self.hps['dqn']['LR_PLAYING'])

            self.load_checkpoint() if self.hps['agent']['READ_CHECKPOINTS'] else None

            if self.hps['dqn']['PER']:
                self.replay_bidding = PrioBuffer(self.hps['dqn']['REPLAY_SIZE_BIDDING'])
                self.replay_playing = PrioBuffer(self.hps['dqn']['REPLAY_SIZE_PLAYING'])
            else:
                self.replay_bidding = ExperienceBuffer(capacity=self.hps['dqn']['REPLAY_SIZE_BIDDING'],
                                                       seed=hps['env']['SEED'])
                self.replay_playing = ExperienceBuffer(capacity=self.hps['dqn']['REPLAY_SIZE_PLAYING'],
                                                       seed=hps['env']['SEED'])

            self.epsilon_schedule_bidding = epsilon_decay_schedule(
                decay_type=self.hps['dqn']['eps']['STRATEGY_BIDDING'],
                total_steps=self.hps['agent']['GAMES'],
                init_epsilon=self.hps['dqn']['eps']['START'],
                min_epsilon=self.hps['dqn']['eps']['FINAL'],
                decay_share=self.hps['dqn']['eps']['SHARE'])
            self.epsilon_schedule_playing = epsilon_decay_schedule(
                decay_type=self.hps['dqn']['eps']['STRATEGY_PLAYING'],
                total_steps=self.hps['agent']['GAMES'],
                init_epsilon=self.hps['dqn']['eps']['START'],
                min_epsilon=self.hps['dqn']['eps']['FINAL'],
                decay_share=self.hps['dqn']['eps']['SHARE'])

        # Slave players can access networks, buffers and epsilon schedule via the master
        else:
            self.input_size_playing_ext = self.master.input_size_playing_ext
            self.bidding_actor = self.master.bidding_actor
            self.playing_actor = self.master.playing_actor
            self.replay_bidding = self.master.replay_bidding
            self.replay_playing = self.master.replay_playing
            self.epsilon_schedule_bidding = self.master.epsilon_schedule_bidding
            self.epsilon_schedule_playing = self.master.epsilon_schedule_playing

        self.epsilon_bidding = self.epsilon_schedule_bidding[0]
        self.epsilon_playing = self.epsilon_schedule_playing[0]

    # def __repr__(self):
    #     return super().__repr__() + "(DQN)"

    def set_agent_mode(self, agent_mode):
        self.agent_mode = agent_mode
        # set epsilon to Zero if agent is not training
        if agent_mode != AgentMode.TRAIN:
            self.epsilon_bidding = 0
            self.epsilon_playing = 0

    def extend_observation_with_history(self, observation):
        if self.master is None:
            historic_extension = self.hidden_c
        else:
            historic_extension = self.master.hidden_c
        observation = np.concatenate((observation, historic_extension.flatten()))
        return observation

    def bidding(self, pos_in_bidding, trump_suit_index, bids):
        observation = self.create_input_for_bidding(
            hand=self.hand,
            pos_in_bidding=pos_in_bidding,
            trump_suit_index=trump_suit_index,
            bids=bids)

        # Choose action
        if self.rng.random() < self.epsilon_bidding:
            action = self.rng.integers(low=0, high=self.num_tricks_to_play + 1)
        else:
            action = self.bidding_actor.sample_action(observation)

        if self.agent_mode == AgentMode.TRAIN:
            self.save_transition(obs=observation, action=action, mask=None, reward=None)
        self.transition()

        return action

    def playing(self, current_trick, pos_in_trick, pos_in_bidding, playable_cards, action_mask,
                trump_suit_index, follow_suit, bids, tricks, history, current_best_card):
        observation = self.create_input_for_playing(
            hand=self.hand,
            current_trick=current_trick,
            pos_in_trick=pos_in_trick,
            pos_in_bidding=pos_in_bidding,
            trump_suit_index=trump_suit_index,
            follow_suit=follow_suit,
            bids=bids,
            tricks=tricks,
            best=current_best_card)

        if self.agent_mode == AgentMode.PLAY and not self.hps['agent']['PLAYER_TYPE'] == "HUMAN":
            self.print_observation(observation)

        # Optionally integrate the Historic information:
        if self.hps['agent']['HISTORY_PREPROCESSING']:
            observation = self.extend_observation_with_history(observation)

        assert len(observation) == self.input_size_playing_ext

        if self.rng.random() < self.epsilon_playing:
            action = self.rng.choice(a=playable_cards, replace=False)
        else:
            action = self.playing_actor.sample_masked_action(observation, action_mask)

        if self.agent_mode == AgentMode.TRAIN:
            self.save_transition(obs=observation, action=action, mask=action_mask, reward=None)
        self.transition()

        return action

    def inform_about_played_card(self, card, pos_bidding, is_last):
        """
        Perform one forward pass with the LSTM and recover the hidden information
        :param card:
        :param pos_bidding:
        :param is_last:
        :return:
        """
        if self.hps['agent']['HISTORY_PREPROCESSING'] and self.master is None:
            historic_card = np.zeros(self.num_cards, dtype=int)
            historic_card[card] = 1
            historic_player = np.zeros(self.num_players, dtype=int)
            historic_player[pos_bidding] = 1
            historic_trump = np.zeros(self.num_suit_including_none, dtype=int)
            historic_trump[self.env.trump_suit_index] = 1
            historic_input = np.concatenate((historic_card, historic_player, historic_trump))
            assert len(historic_input) == self.hps['hist']['INPUT']

            with torch.no_grad():
                historic_input_sample = torch.tensor(historic_input.reshape((1, 1, -1)), dtype=torch.float32)
                prediction, hidden_tuple = self.history_net(historic_input_sample, (self.hidden_h, self.hidden_c))
            self.hidden_h = hidden_tuple[0]
            self.hidden_c = hidden_tuple[1]

            # Reset hidden state at the end of the episode
            if is_last:
                self.hidden_h = torch.zeros(1, 1, self.hps['hist']['HIDDEN_SIZE'])
                self.hidden_c = torch.zeros(1, 1, self.hps['hist']['HIDDEN_SIZE'])

    def finish_round(self, reward, history):
        self.game_phase = GamePhase.FINISH
        if self.agent_mode == AgentMode.TRAIN:
            self.save_transition(obs=None, action=None, mask=None, reward=reward)
            self.train_networks()
        self.transition()

    def save_transition(self, obs, action, mask, reward):
        """
        Save observation, action and reward
        :param obs:
        :param action:
        :param mask:
        :param reward:
        :return:
        """
        # In bidding, we need to save observation & action
        if self.game_phase == GamePhase.BIDDING:
            self.obs_bidding = obs
            self.action_bidding = action
        # In first playing step, we need to save observation & action
        elif self.game_phase == GamePhase.FIRST_PLAYING:
            self.previous_obs = obs
            self.previous_action = action
            self.previous_mask = mask
        # In ordinary playing steps we can create an experience without reward
        elif self.game_phase == GamePhase.PLAYING:
            exp = PlayingExperience(obs=self.previous_obs,
                                    action=self.previous_action,
                                    reward=0.0,
                                    done=False,
                                    next_obs=obs,
                                    mask=self.previous_mask,
                                    next_mask=mask)
            self.replay_playing.append(exp)
            self.previous_obs = obs
            self.previous_action = action
            self.previous_mask = mask
        # In the final playing step we create an experience with reward and and bidding experience
        elif self.game_phase == GamePhase.FINISH:
            exp = BiddingExperience(self.obs_bidding, self.action_bidding, reward)
            self.replay_bidding.append(exp)
            exp = PlayingExperience(obs=self.previous_obs,
                                    action=self.previous_action,
                                    reward=reward,
                                    done=True,
                                    next_obs=np.ones_like(a=self.previous_obs),
                                    mask=self.previous_mask,
                                    next_mask=np.ones_like(a=self.previous_mask))
            self.replay_playing.append(exp)

    def train_networks(self):
        self.epsilon_bidding = self.epsilon_schedule_bidding[self.game_count]
        self.epsilon_playing = self.epsilon_schedule_playing[self.game_count]
        if self.master is None:
            if (self.game_count + 1) % self.hps['dqn']['TRAIN_INTERVAL'] == 0:
                if len(self.replay_bidding) >= self.hps['dqn']['REPLAY_START_SIZE_BIDDING']:
                    self.train_bidding()
                    self.train_count_bidding += 1
                if len(self.replay_playing) >= self.hps['dqn']['REPLAY_START_SIZE_PLAYING']:
                    self.train_playing()
                    self.train_count_playing += 1
                    if (self.game_count + 1) % self.hps['dqn']['COPY_TARGET'] == 0:
                        copy_weights(net_from=self.playing_actor,
                                     net_to=self.playing_actor_target,
                                     polyak_tau=self.hps['dqn']['TAU_TARGET'])

    def train_bidding(self):
        """
        Train bidding with DQN
        """
        assert self.master is None
        self.bidding_actor_opt.zero_grad()

        # Unwrap experience (either Prioritized or Regular)
        if self.hps['dqn']['PER']:
            per_samples, per_indices, per_weights = self.replay_bidding.sample(self.hps['dqn']['BATCH_SIZE_BIDDING'])
            obs_batch, action_batch, reward_batch = [], [], []
            for exp in per_samples:
                obs_batch.append(exp.obs)
                action_batch.append(exp.action)
                reward_batch.append(exp.reward)
            obs_batch = np.array(obs_batch)
            action_batch = np.array(action_batch)
            reward_batch = np.array(reward_batch)
        else:
            obs_batch, action_batch, reward_batch = self.replay_bidding.sample_bidding(
                self.hps['dqn']['BATCH_SIZE_BIDDING'])

        obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)

        raw_values = self.bidding_actor(obs_batch)
        q_values = raw_values.gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        target_q_values = reward_batch  # Use pure Monte Carlo estimates

        # Calculate loss
        if self.hps['dqn']['PER']:
            batch_weights_v = torch.tensor(per_weights)
            l = (q_values - target_q_values) ** 2
            losses = batch_weights_v * l
            loss = losses.mean()
            sample_prios = (losses + 1e-5).data.cpu().numpy()
        else:
            loss = torch.nn.MSELoss()(q_values, target_q_values)

        loss.backward()
        self.bidding_actor_opt.step()

        if self.hps['dqn']['PER']:
            self.replay_bidding.update_priorities(per_indices, sample_prios)
            beta = self.replay_bidding.update_beta(
                self.game_count, self.hps['agent']['BATCHES'] * self.hps['agent']['ITERATIONS_PER_BATCH'])

        avg_values = np.mean(raw_values.detach().numpy(), axis=1)
        adv_values = q_values.detach().numpy() - avg_values

        self.logging_dqn(phase="bidding",
                         buffer=len(self.replay_bidding),
                         epsilon=self.epsilon_bidding,
                         grad=calc_grad(self.bidding_actor)[0],
                         loss=loss.item(),
                         q_avg=avg_values.mean(),
                         q_adv=adv_values.mean(),
                         weight=calc_weight(self.bidding_actor))

    def train_playing(self):
        """
        Train playing with DQN
        """
        assert self.master is None
        self.playing_actor_opt.zero_grad()

        # Unwrap experience (either Prioritized or Regular)
        if self.hps['dqn']['PER']:
            per_samples, per_indices, per_weights = self.replay_playing.sample(self.hps['dqn']['BATCH_SIZE_PLAYING'])
            obs_batch, action_batch, reward_batch, done_batch, next_obs_batch, mask_batch, next_mask_batch \
                = [], [], [], [], [], [], []
            for exp in per_samples:
                obs_batch.append(exp.obs)
                action_batch.append(exp.action)
                reward_batch.append(exp.reward)
                done_batch.append(exp.done)
                next_obs_batch.append(exp.next_obs)
                mask_batch.append(exp.mask)
                next_mask_batch.append(exp.next_mask)
            obs_batch = np.array(obs_batch)
            action_batch = np.array(action_batch)
            reward_batch = np.array(reward_batch)
            done_batch = np.array(done_batch)
            next_obs_batch = np.array(next_obs_batch)
            mask_batch = np.array(mask_batch)
            next_mask_batch = np.array(next_mask_batch)
        else:
            obs_batch, action_batch, reward_batch, done_batch, next_obs_batch, mask_batch, next_mask_batch = \
                self.replay_playing.sample_playing(self.hps['dqn']['BATCH_SIZE_PLAYING'])
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        done_batch = torch.BoolTensor(done_batch)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32)
        mask_batch = torch.BoolTensor(mask_batch)
        next_mask_batch = torch.BoolTensor(next_mask_batch)

        # A: Evaluate training net
        raw_values = self.playing_actor(obs_batch)
        q_values = raw_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # B: Evaluate target net (without gradients) in order to bootstrap training net
        with torch.no_grad():
            # Double DQN: action selection is done by online network and action evaluation by target network
            if self.hps['dqn']['DOUBLE']:
                target_best_actions = self.playing_actor.get_max_action_masked(next_obs_batch, next_mask_batch)
                target_raw_values = self.playing_actor_target(next_obs_batch)
                target_q_values = target_raw_values.gather(1, target_best_actions.unsqueeze(1)).squeeze(1)
            # Normal DQN: both action selection and evaluation is done by target network
            else:
                target_q_values = self.playing_actor_target.get_max_value_masked(next_obs_batch, next_mask_batch)
            target_q_values[done_batch] = 0.0
            target_q_values = reward_batch + self.hps['agent']['GAMMA'] * target_q_values.detach()

        # Calculate loss
        if self.hps['dqn']['PER']:
            batch_weights_v = torch.tensor(per_weights)
            l = (q_values - target_q_values) ** 2
            losses = batch_weights_v * l
            loss = losses.mean()
            sample_prios = (losses + 1e-5).data.cpu().numpy()
        else:
            loss = torch.nn.MSELoss()(q_values, target_q_values)

        loss.backward()
        self.playing_actor_opt.step()

        if self.hps['dqn']['PER']:
            self.replay_playing.update_priorities(per_indices, sample_prios)
            beta = self.replay_playing.update_beta(
                self.game_count,
                self.hps['agent']['BATCHES'] * self.hps['agent']['ITERATIONS_PER_BATCH'])

        avg_values = np.array([raw_values[i][mask_batch[i]].mean().item()
                               for i in range(self.hps['dqn']['BATCH_SIZE_PLAYING'])])
        adv_values = q_values.detach().numpy() - avg_values

        self.logging_dqn(phase="playing",
                         buffer=len(self.replay_playing),
                         epsilon=self.epsilon_playing,
                         grad=calc_grad(self.playing_actor)[0],
                         loss=loss.item(),
                         q_avg=avg_values.mean(),
                         q_adv=adv_values.mean(),
                         weight=calc_weight(self.playing_actor))

    def logging_dqn(self, phase, buffer, epsilon, grad, loss, q_avg, q_adv, weight):
        self.writer.add_scalar("buffer/" + phase, buffer, self.game_count)
        self.writer.add_scalar("epsilon/" + phase, epsilon, self.game_count)
        self.writer.add_scalar("grad/" + phase, grad, self.game_count)
        self.writer.add_scalar("loss/" + phase, loss, self.game_count)
        self.writer.add_scalar("q_values_average/" + phase, q_avg, self.game_count)
        self.writer.add_scalar("q_values_advantage/" + phase, q_adv, self.game_count)
        self.writer.add_scalar("weight/" + phase, weight, self.game_count)

    def load_checkpoint(self):
        if self.master is None:
            if self.hps['agent']['HISTORY_PREPROCESSING']:
                checkpoint = torch.load(self.hps['dqn']['CHECKPOINT_HIST'] + ".pt")
            else:
                checkpoint = torch.load(self.hps['dqn']['CHECKPOINT'] + ".pt")
            self.bidding_actor.load_state_dict(checkpoint['dqn_bidding'])
            self.bidding_actor_opt.load_state_dict(checkpoint['dqn_bidding_opt'])
            self.playing_actor.load_state_dict(checkpoint['dqn_playing'])
            self.playing_actor_opt.load_state_dict(checkpoint['dqn_playing_opt'])
            self.playing_actor_target.load_state_dict(checkpoint['dqn_playing_target'])

    def save_checkpoint(self, current_time):
        assert self.master is None
        checkpoint = {'dqn_bidding': self.bidding_actor.state_dict(),
                      'dqn_bidding_opt': self.bidding_actor_opt.state_dict(),
                      'dqn_playing': self.playing_actor.state_dict(),
                      'dqn_playing_opt': self.playing_actor_opt.state_dict(),
                      'dqn_playing_target': self.playing_actor_target.state_dict()}
        if self.hps['agent']['HISTORY_PREPROCESSING']:
            torch.save(checkpoint, self.hps['dqn']['CHECKPOINT_HIST'] + "-" + current_time + ".pt")
        else:
            torch.save(checkpoint, self.hps['dqn']['CHECKPOINT'] + "-" + current_time + ".pt")


class ModelPlayer(DQNPlayer):
    """
    This player uses a state estimator to perform a tree search in the playing phase
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps, master)

        self.previous_state = None

        if master is None:

            # Use pre-trained baseline (memoryless) DQN networks for regular decision-making
            checkpoint = torch.load(self.hps['dqn']['CHECKPOINT'] + ".pt")
            # For bidding, we can simply overwrite the network weights
            self.bidding_actor.load_state_dict(checkpoint['dqn_bidding'])
            # For playing we need a new network without history preprocessing (which is also used as rollout policy)
            # because the architecture might be different
            self.playing_rollout = DQNNetwork(num_inputs=self.input_size_playing,
                                              num_actions=self.output_size_playing,
                                              hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                              do_history_preprocessing=False,
                                              hps=self.hps)
            self.playing_rollout.load_state_dict(checkpoint['dqn_playing'])

            self.estimator = StateEstimator(num_inputs=self.input_size_playing_ext,
                                            num_estimations=self.hps['model']['OUTPUT'],
                                            hidden=self.hps['model']['HIDDEN'],
                                            do_history_preprocessing=self.hps['agent']['HISTORY_PREPROCESSING'],
                                            hps=self.hps)
            self.estimator_opt = torch.optim.Adam(self.estimator.parameters(), lr=self.hps['model']['LR'])

            self.load_checkpoint_model() if self.hps['agent']['READ_CHECKPOINTS'] else None

            self.replay_estimator = ExperienceBuffer(capacity=self.hps['model']['REPLAY_SIZE'],
                                                     seed=hps['env']['SEED'])

            self.certainty_queue = {}
            self.certainty_queue = [collections.deque(maxlen=100) for _ in range(self.num_players * self.num_tricks_to_play)]

            self.mcts_count_total = 0
            self.mcts_count_mcts = 0
            self.mcts_count_samples = 0
            self.mcts_count_overwritten = 0
            self.mcts_count_correct = 0
            self.mcts_rewards = []
            self.mcts_deviation = 0

            self.loss_queue = collections.deque(maxlen=100)
            self.loss_mean = 0

        else:
            self.estimator = self.master.estimator
            self.replay_estimator = self.master.replay_estimator
            self.playing_rollout = self.master.playing_rollout

        # Always bid and play greedily
        self.epsilon_bidding = 0.0
        self.epsilon_playing = 0.0

    # def __repr__(self):
    #     return Player.__repr__(self) + "(Model)"

    def load_checkpoint_model(self):
        if self.master is None:
            if self.hps['agent']['HISTORY_PREPROCESSING']:
                checkpoint = torch.load(self.hps['model']['CHECKPOINT_HIST'] + ".pt")
            else:
                checkpoint = torch.load(self.hps['model']['CHECKPOINT'] + ".pt")
            self.estimator.load_state_dict(checkpoint['model'])
            self.estimator_opt.load_state_dict(checkpoint['model_opt'])

    def save_checkpoint(self, current_time):
        assert self.master is None
        checkpoint = {'model': self.estimator.state_dict(),
                      'model_opt': self.estimator_opt.state_dict()}
        if self.hps['agent']['HISTORY_PREPROCESSING']:
            torch.save(checkpoint, self.hps['model']['CHECKPOINT_HIST'] + "-" + current_time + ".pt")
        else:
            torch.save(checkpoint, self.hps['model']['CHECKPOINT'] + "-" + current_time + ".pt")


    def playing(self, current_trick, pos_in_trick, pos_in_bidding, playable_cards, action_mask,
                trump_suit_index, follow_suit, bids, tricks, history, current_best_card):

        observation = self.create_input_for_playing(
            hand=self.hand,
            current_trick=current_trick,
            pos_in_trick=pos_in_trick,
            pos_in_bidding=pos_in_bidding,
            trump_suit_index=trump_suit_index,
            follow_suit=follow_suit,
            bids=bids,
            tricks=tricks,
            best=current_best_card)

        if self.agent_mode == AgentMode.PLAY:
            self.print_observation(observation)

        """
        STEP 1: Greedy DQN Decision Making (without history) 
        """
        action, action_reward = self.playing_rollout.sample_masked_action_and_value(observation, action_mask)

        """
        STEP 2: Model-Based Decision Update (with or without history)
        """
        # Optionally integrate the historic information into the state Estimation:
        if self.hps['agent']['HISTORY_PREPROCESSING']:
            observation = self.extend_observation_with_history(observation)

        # Process Certainty information
        certainty_index = self.num_players * current_trick + pos_in_trick
        logits = self.estimator.get_state_prediction(observation)
        logits = logits.reshape((self.num_cards, -1))
        estimation = softmax(logits, axis=1)
        certainty_believed = np.mean(np.amax(estimation, axis=1))
        known_cards = self.env.ground_truth[:, self.hps['env']['GT_TRUMP']:].sum() + len(self.hand)
        certainty_true = known_cards / self.num_cards
        threshold = certainty_true > self.hps['model']['CERTAINTY']

        if self.master is None and self.agent_mode == AgentMode.TRAIN:
            self.certainty_queue[certainty_index].append(certainty_believed)
            # if self.game_count % 100 == 0:
            #     self.writer.add_scalar("certainty-believed/" + str(certainty_index),
            #                            np.mean(self.certainty_queue[certainty_index]),
            #                            self.game_count)
            #     self.writer.add_scalar("certainty-true/" + str(certainty_index),
            #                            certainty_true,
            #                            self.game_count)

        # MCTS Decision Making & Reward Comparison
        if self.hps['model']['SEARCH'] and self.master is None and len(playable_cards) > 1:
            self.mcts_count_total += 1
            if threshold:
                self.mcts_count_mcts += 1
                mean_rewards = self.mcts(logits=logits,
                                         current_trick=current_trick,
                                         position_in_trick=pos_in_trick,
                                         position_in_bidding=pos_in_bidding,
                                         playable_cards=playable_cards,
                                         history=history)
                mcts_action_index = np.argmax(mean_rewards)
                mcts_action = playable_cards[mcts_action_index]
                mcts_reward = mean_rewards[mcts_action_index]
                regular_action_index = np.where(playable_cards == action)[0]
                regular_reward = mean_rewards[regular_action_index]
                # print("Regular action {} with reward {}, mcts action {} with reward {}".format(action,
                #                                                                                action_reward,
                #                                                                                mcts_action,
                #                                                                                mcts_reward))
                if mcts_reward > regular_reward + 0.05 and regular_reward >= -1:
                    self.mcts_count_overwritten += 1
                    self.mcts_rewards.append(mcts_reward)
                    if self.agent_mode == AgentMode.PLAY:
                        print("Overwrite regular action {} with reward {}, valued by mcts as {} "
                              "with mcts action {} with reward {}".format(action,
                                                                          action_reward,
                                                                          regular_reward,
                                                                          mcts_action,
                                                                          mcts_reward))
                    action = mcts_action
                if self.agent_mode == AgentMode.TRAIN:
                    self.writer.add_scalar("mcts_overwritten", self.mcts_count_overwritten, self.game_count)

        # Print information about prediction
        if self.agent_mode == AgentMode.PLAY:
            print("Certainty believed {} vs true {}, Deviation {}, Prediction vs. Truth \n {}".format(
                np.round(certainty_believed, decimals=4),
                np.round(certainty_true, decimals=4),
                np.round(np.sum(np.abs(estimation - self.env.ground_truth)), decimals=2),
                np.round(np.concatenate((estimation, self.env.ground_truth), axis=1), decimals=1)))

        # Save experience in experience buffer
        if self.agent_mode == AgentMode.TRAIN:
            state_ground_truth = self.env.ground_truth.copy().flatten()
            exp = SupervisedExperience(input=observation,
                                       target=state_ground_truth)
            self.replay_estimator.append(exp)

        self.transition()
        return action

    def mcts(self, logits, current_trick, position_in_trick, position_in_bidding, playable_cards, history):
        """
        We call this function MCTS, but it is not the MCTS algorithm in the stricter sense
        1) Create artificial simulated environment
        2) Sample multiple states pretending to have perfect information
        3) Run multiple games in simulated environment
        4) Evaluate alternative actions
        """

        # 1: Create artifical environment for simulation
        q_values = dict.fromkeys(playable_cards, 0)
        games_played_with = dict.fromkeys(playable_cards, 0)
        sim = Simulation(env=self.env,
                         current_trick=current_trick,
                         position_in_trick=position_in_trick,
                         position_in_bidding=position_in_bidding,
                         history=history,
                         hps=self.hps)

        # 2: Sample multiple states
        for _ in range(self.hps['model']['SAMPLES_TO_GENERATE']):
            sampled_state = np.zeros_like(a=logits, dtype=int)
            # Calculate the number of cards to be sampled for each column:
            target_shape = logits.shape[1]
            target = np.zeros(shape=target_shape, dtype=int)
            for p in range(self.num_players):
                # Cards played
                target[self.hps['env']['GT_PLAYED'] + p] = np.sum(history[:, 1] == p)
                # Cards in hand
                target[self.hps['env']['GT_HAND'] + p] = self.num_tricks_to_play - np.sum(history[:, 1] == p)
            # Trump card
            if np.sum(self.env.ground_truth[:, self.hps['env']['GT_TRUMP']]) > 0:
                target[self.hps['env']['GT_TRUMP']] = 1
            else:
                target[self.hps['env']['GT_TRUMP']] = 0
            # All other cards must still be in the deck
            target[self.hps['env']['GT_DECK']] = self.num_cards - target[1:].sum()

            if self.hps['model']['SAMPLING_TYPE'] == "TRUTH":
                # A: Theoretical bound: replace sampled state with ground truth:
                sampled_state[:, :] = self.env.ground_truth[:, :]
            elif self.hps['model']['SAMPLING_TYPE'] == "UNIFORM":
                # B: Uniform distribution:
                # B1: use own cards, trump card and played cards from ground truth
                sampled_state[:, self.hps['env']['GT_HAND'] + position_in_bidding] = \
                    self.env.ground_truth[:, self.hps['env']['GT_HAND'] + position_in_bidding]
                sampled_state[:, self.hps['env']['GT_TRUMP']] = self.env.ground_truth[:, self.hps['env']['GT_TRUMP']]
                sampled_state[:, self.hps['env']['GT_PLAYED']:self.hps['env']['GT_PLAYED']+self.num_players] = \
                    self.env.ground_truth[:, self.hps['env']['GT_PLAYED']:self.hps['env']['GT_PLAYED']+self.num_players]
                # B2: The cards in the deck and the other players cards are sampled from a uniform distribution
                cards_to_be_sampled = np.arange(self.num_cards)[np.sum(sampled_state, axis=1) == 0]
                self.rng.shuffle(cards_to_be_sampled)
                low = 0
                high = target[self.hps['env']['GT_DECK']]
                sampled_state[cards_to_be_sampled[low:high], self.hps['env']['GT_DECK']] = 1
                for p in range(self.num_players):
                    if p != position_in_bidding:
                        low = high
                        high = high + target[self.hps['env']['GT_HAND'] + p]
                        sampled_state[cards_to_be_sampled[low:high], self.hps['env']['GT_HAND'] + p] = 1
            elif self.hps['model']['SAMPLING_TYPE'] == "SAMPLE":
                # C: Use probabilities from model training:
                logits_masked = logits.copy()
                cards_to_be_sampled = np.arange(self.num_cards)[np.sum(sampled_state, axis=1) == 0]
                # Shuffling the cards avoids biased sampling behavior
                self.rng.shuffle(cards_to_be_sampled)
                for c in cards_to_be_sampled:
                    # Mask logits of columns where no cards should be sampled
                    logits_masked[:, sampled_state.sum(axis=0) >= target] = -np.inf
                    sample = self.rng.choice(a=target_shape, p=softmax(logits_masked[c, :]))
                    sampled_state[c, sample] = 1
            assert (sampled_state.sum(axis=1) == np.ones(shape=sampled_state.shape[0])).all()

            self.mcts_count_samples += 1
            self.mcts_deviation += np.sum(np.abs(sampled_state - self.env.ground_truth))

            # 3: Run multiple simulations with each possible action
            for alternative_action in playable_cards:
                # Only perform simulations, if the playable card was actually sampled to the agent's hand:
                if alternative_action in np.flatnonzero(sampled_state[:, self.hps['env']['GT_HAND'] + position_in_bidding]):
                    for _ in range(self.hps['model']['SIMULATIONS_TO_RUN']):
                        reward_of_game = sim.run_simulation(sampled_state=sampled_state,
                                                            card_played=alternative_action)
                        q_values[alternative_action] += reward_of_game
                        games_played_with[alternative_action] += 1

        # 4: Evaluate alternative actions
        mean_rewards = np.zeros(shape=len(playable_cards))
        for i, alternative_action in enumerate(playable_cards):
            # Only compare those actions that were actually taken
            if games_played_with[alternative_action] > 0:
                mean_rewards[i] = q_values[alternative_action] / games_played_with[alternative_action]
            else:
                mean_rewards[i] = -1.1
            if self.agent_mode == AgentMode.PLAY:
                print("Action {} yields mean reward {}".format(alternative_action, mean_rewards[i]))
        return mean_rewards

    def finish_round(self, reward, history):
        self.game_phase = GamePhase.FINISH

        if self.agent_mode == AgentMode.TRAIN:
            self.train_networks()

        if self.master is None and self.hps['model']['SEARCH']:
            if len(self.mcts_rewards) > 0:
                reward_array = np.array(self.mcts_rewards)
                self.mcts_count_correct += np.sum(reward_array < reward+0.01)
                if self.agent_mode == AgentMode.PLAY:
                    print("Overwrite Predictions = {} vs. Truth = {}".format(reward_array, reward))
                self.mcts_rewards.clear()

        self.transition()

    def finish_interaction(self):
        if self.master is None and self.hps['agent']['TRAINING_MODE']:
            print("Final average loss: {}".format(np.round(self.loss_mean, decimals=6)))
        if self.master is None and self.hps['model']['SEARCH']:
            print("Total Decisions: {}".format(self.mcts_count_total))
            share = self.mcts_count_mcts/self.mcts_count_total if self.mcts_count_total else 0
            print("Thereof tested by MCTS: {} = {}%".format(
                self.mcts_count_mcts,
                np.round(100*share, 2)))
            share = self.mcts_count_overwritten / self.mcts_count_mcts if self.mcts_count_mcts else 0
            print("Thereof overwritten: {} = {}%".format(
                self.mcts_count_overwritten,
                np.round(100 * share, 2)))
            share = self.mcts_count_correct / self.mcts_count_overwritten if self.mcts_count_overwritten else 0
            print("Thereof real reward at least as high: {} = {}%".format(
                self.mcts_count_correct,
                np.round(100 * share, 2)))
            share = self.mcts_deviation/self.mcts_count_samples if self.mcts_count_samples else 0
            print("{} samples drawn with mean deviation {}".format(
                self.mcts_count_samples,
                np.round(share, 2)))

    def train_networks(self):
        if self.master is None:
            if (self.game_count + 1) % self.hps['model']['TRAIN_INTERVAL'] == 0:
                if len(self.replay_estimator) >= self.hps['model']['REPLAY_START_SIZE']:
                    self.train_estimator()

    def train_estimator(self):
        obs_batch, state_batch = self.replay_estimator.sample_supervised(self.hps['model']['BATCH_SIZE'])
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
        state_batch = torch.tensor(state_batch, dtype=torch.float32)

        self.estimator_opt.zero_grad()
        predicted_state = self.estimator(obs_batch)
        loss = torch.nn.BCEWithLogitsLoss()(predicted_state, state_batch)
        loss.backward()
        self.estimator_opt.step()

        self.writer.add_scalar("buffer/estimator", len(self.replay_estimator), self.game_count)
        self.writer.add_scalar("grad/estimator", calc_grad(self.estimator)[0], self.game_count)
        self.writer.add_scalar("loss/estimator", loss.item(), self.game_count)
        self.writer.add_scalar("weight/estimator", calc_weight(self.estimator), self.game_count)

        self.loss_queue.append(loss.item())
        self.loss_mean = np.mean(self.loss_queue)
        self.writer.add_scalar("loss/estimator-mean", self.loss_mean, self.game_count)
