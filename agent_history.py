import collections

import numpy as np
import torch

from agent import OptimizedPlayer, calc_grad, GamePhase, AgentMode, calc_weight
from networks_history import HistoryNet
from replay_buffer import ExperienceBuffer, SupervisedExperience


class HistoryPlayer(OptimizedPlayer):
    """
    Player that learns a representation of the history using an LSTM network
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps)

        torch.manual_seed(hps['env']['SEED'])
        self.master = master

        if master is None:
            self.hist_output_dim_cards = self.num_cards * self.num_players
            self.hist_output_dim_follow = self.num_suit * self.num_players
            self.hist_output_dim_winner = self.num_players
            assert self.hps['hist']['INPUT'] == self.num_cards + self.num_players + self.num_suit_including_none
            assert self.hps['hist'][
                       'OUTPUT'] == self.hist_output_dim_cards + self.hist_output_dim_follow + self.hist_output_dim_winner
            self.history_net = HistoryNet(
                input_dim=self.hps['hist']['INPUT'],
                hidden_dim=self.hps['hist']['HIDDEN_SIZE'],
                output_dim=self.hps['hist']['OUTPUT'],
                hps=self.hps)
            self.history_net_opt = torch.optim.Adam(self.history_net.parameters(), lr=self.hps['hist']['LR'])
            self.load_checkpoint() if self.hps['agent']['READ_CHECKPOINTS'] else None
            self.replay_history = ExperienceBuffer(capacity=self.hps['hist']['REPLAY_SIZE'],
                                                   seed=hps['env']['SEED'])
            self.historic_input_sequence = []
            self.historic_output_sequence = []
            self.hidden_h = torch.zeros(1, 1, self.hps['hist']['HIDDEN_SIZE'])
            self.hidden_c = torch.zeros(1, 1, self.hps['hist']['HIDDEN_SIZE'])
            self.loss_queue = collections.deque(maxlen=100)
            self.loss_mean = 0
        else:
            self.history_net = self.master.history_net
            self.replay_history = self.master.replay_history

    def inform_about_played_card(self, card, pos_bidding, is_last):
        if self.master is None:
            historic_card = np.zeros(self.num_cards, dtype=int)
            historic_card[card] = 1
            historic_player = np.zeros(self.num_players, dtype=int)
            historic_player[pos_bidding] = 1
            historic_trump = np.zeros(self.num_suit_including_none, dtype=int)
            historic_trump[self.env.trump_suit_index] = 1
            historic_input = np.concatenate((historic_card, historic_player, historic_trump))
            assert len(historic_input) == self.hps['hist']['INPUT']

            cards_target = np.zeros(shape=(self.num_cards, self.num_players))
            cards_target[:, :] = self.env.ground_truth[:, self.num_players + 2:]

            follow_target = np.zeros(shape=(self.num_suit, self.num_players))
            follow_target[:, :] = self.env.unable_to_follow

            winner_target = np.zeros(shape=self.num_players)
            winner_target[self.env.current_trick_winner_bid_pos] = 1

            """
            In PLAY mode, we step through the sequence one by one and compare prediction with ground ground truth
            """
            if self.agent_mode == AgentMode.PLAY:
                with torch.no_grad():
                    historic_input_sample = torch.tensor(historic_input.reshape((1, 1, -1)), dtype=torch.float32)
                    prediction, hidden_tuple = self.history_net(historic_input_sample, (self.hidden_h, self.hidden_c))
                    prediction = prediction.numpy().flatten()
                    prediction = 1 / (1 + np.exp(-prediction))
                    cards_prediction = prediction[:self.hist_output_dim_cards].reshape(cards_target.shape)
                    follow_prediction = prediction[
                                        self.hist_output_dim_cards:self.hist_output_dim_cards + self.hist_output_dim_follow].reshape(
                        follow_target.shape)
                    winner_prediction = prediction[self.hist_output_dim_cards + self.hist_output_dim_follow:]
                    print("Card prediction vs. truth {}: \n {}".format(
                        np.round(np.sum(np.abs(cards_prediction - cards_target)), decimals=1),
                        np.round(np.concatenate((cards_prediction, cards_target), axis=1), decimals=1)))
                    print("Follow prediction vs. truth {}: \n {}".format(
                        np.round(np.sum(np.abs(follow_prediction - follow_target)), decimals=1),
                        np.round(np.concatenate((follow_prediction, follow_target), axis=1), decimals=1)))
                    print("Winner prediction vs. truth {}: \n {}".format(
                        np.round(np.sum(np.abs(winner_prediction - winner_target)), decimals=1),
                        np.round(np.concatenate((winner_prediction, winner_target)), decimals=1)))
                    # Recover hidden information for next iteration
                    self.hidden_h = hidden_tuple[0]
                    self.hidden_c = hidden_tuple[1]

                # Reset hidden state at the end of the episode
                if is_last:
                    self.hidden_h = torch.zeros(1, 1, self.hps['hist']['HIDDEN_SIZE'])
                    self.hidden_c = torch.zeros(1, 1, self.hps['hist']['HIDDEN_SIZE'])
            """
            In TRAIN mode, we save the historic input alongside the target values
            """
            if self.agent_mode == AgentMode.TRAIN:
                self.historic_input_sequence.append(historic_input)
                historic_output = np.concatenate((cards_target.flatten(), follow_target.flatten(), winner_target))
                assert len(historic_output) == self.hps['hist']['OUTPUT']
                self.historic_output_sequence.append(historic_output)

                # If the sequence is complete, we can add it to the replay buffer
                if is_last:
                    hist_in = np.array(self.historic_input_sequence)
                    hist_out = np.array(self.historic_output_sequence)
                    self.replay_history.append(SupervisedExperience(input=hist_in,
                                                                    target=hist_out))
                    self.historic_input_sequence.clear()
                    self.historic_output_sequence.clear()

                    # Start training
                    if (self.game_count + 1) % self.hps['hist']['TRAIN_INTERVAL'] == 0:
                        if len(self.replay_history) >= self.hps['hist']['REPLAY_START_SIZE']:
                            self.train_history()

    def load_checkpoint(self):
        if self.master is None:
            checkpoint = torch.load(self.hps['hist']['CHECKPOINT'] + ".pt")
            self.history_net.load_state_dict(checkpoint['actor'])
            self.history_net_opt.load_state_dict(checkpoint['actor_opt'])

    def save_checkpoint(self, current_time):
        checkpoint = {'actor': self.history_net.state_dict(),
                      'actor_opt': self.history_net_opt.state_dict()}
        torch.save(checkpoint, self.hps['hist']['CHECKPOINT'] + "-" + current_time + ".pt")

    def save_experience(self, is_last, reward, obs):
        pass

    def train_bidding(self):
        pass

    def train_playing(self):
        pass

    def bidding(self, pos_in_bidding, trump_suit_index, bids):
        self.transition()
        return self.rng.integers(low=0, high=self.num_tricks_to_play + 1)

    def playing(self, current_trick, pos_in_trick, pos_in_bidding, playable_cards, action_mask,
                trump_suit_index, follow_suit, bids, tricks, history, current_best_card):
        self.transition()
        return self.rng.choice(a=playable_cards, replace=False)

    def finish_round(self, reward, history):
        self.game_phase = GamePhase.FINISH
        self.transition()

    def train_history(self):
        self.history_net_opt.zero_grad()
        history, target = self.replay_history.sample_supervised(self.hps['hist']['BATCH_SIZE'])
        history = torch.tensor(history, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        prediction, _ = self.history_net(history, None)

        loss = torch.nn.BCEWithLogitsLoss()(prediction, target)
        loss.backward()
        self.history_net_opt.step()

        self.writer.add_scalar("buffer/history", len(self.replay_history), self.game_count)
        self.writer.add_scalar("loss/history", loss.item(), self.game_count)
        self.writer.add_scalar("grad/history", calc_grad(self.history_net)[0], self.game_count)
        self.writer.add_scalar("weight/history", calc_weight(self.history_net), self.game_count)

        self.loss_queue.append(loss.item())
        self.loss_mean = np.mean(self.loss_queue)
        self.writer.add_scalar("loss/history-mean", self.loss_mean, self.game_count)

    def finish_interaction(self):
        if self.master is None:
            print("Final average loss: {}".format(np.round(self.loss_mean, decimals=6)))
