from abc import abstractmethod, ABCMeta
from enum import Enum

import numpy as np


def calc_weight(net):
    """
    Calculate mean of network weights
    .data extracts the raw tensor from the parameters
    .numpy() converts from tensor to ndarray
    """
    weights = np.concatenate([p.data.numpy().flatten()
                              for p in net.parameters()])
    return np.sqrt(np.mean(np.square(weights)))


def calc_grad(net):
    """
    Calculate the norm of the parameters of a neural network
    :param net: neural network
    :return:
    """
    grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                            for p in net.parameters()
                            if p.grad is not None])
    if np.isnan(np.sum(grads)):
        print("SUPER BAD")
    grad_mean = np.linalg.norm(grads)
    grad_var = np.var(grads)
    grad_max = np.max(np.abs(grads))
    return grad_mean, grad_var, grad_max


def convert_from_binary(vector, units, size):
    """
    Convert multiple (units) one-hot encoded vectors back into an integer representation
    :param vector: multiple concatenated one-hot encoded vectors
    :param units: number one-hot encoded vectors
    :param size: size of each one-hot encoded vectors
    :return:
    """
    output = []
    for i in range(units):
        unwrapped = np.flatnonzero(vector[i * size:(i + 1) * size])
        if len(unwrapped) > 0:
            output.append(unwrapped[0])
    return output


def copy_weights(net_from, net_to, polyak_tau):
    """
    Copy network weights from net_from to net_to
    :param net_from:
    :param net_to:
    :param polyak_tau: if value is given, use Polyak averaging
    :return:
    """
    # Option A: Overwrite Network completely
    if polyak_tau is None:
        net_to.load_state_dict(net_from.state_dict())
    # Option B: Use Smooth Averaging (Polyak)
    else:
        for slave, master in zip(net_to.parameters(), net_from.parameters()):
            slave_ratio = (1.0 - polyak_tau) * slave.data
            master_ratio = polyak_tau * master.data
            mixed_weights = slave_ratio + master_ratio
            slave.data.copy_(mixed_weights)


class GamePhase(Enum):
    BIDDING = 1
    FIRST_PLAYING = 2
    PLAYING = 3
    FINISH = 4


class AgentMode(Enum):
    TRAIN = 1
    PLAY = 2
    EVAL = 3


class Player(metaclass=ABCMeta):
    """
    Abstract base class for all players (= agents)
    """

    def __init__(self, identification, player_name, hps):
        self.id = identification
        self.name = player_name
        self.hps = hps
        self.num_players = hps['env']['PLAYERS']
        self.num_tricks_to_play = hps['env']['TRICKS']
        self.num_cards = hps['env']['CARDS']
        self.num_suit = hps['env']['SUIT']
        self.num_suit_including_none = self.num_suit + 1  # The last index stands for no trump suit at all
        self.rng = np.random.default_rng(hps['env']['SEED'])
        self.writer = None
        self.next = None
        self.previous = None
        self.hand = None
        self.bidding_pos = None
        self.env = None

    def __repr__(self):
        return self.name

    def set_agent_mode(self, agent_mode):
        pass

    @abstractmethod
    def bidding(self, pos_in_bidding, trump_suit_index, bids):
        """
        Ask the player for a bid
        :return: bid
        """

    @abstractmethod
    def playing(self, current_trick, pos_in_trick, pos_in_bidding, playable_cards, action_mask,
                trump_suit_index, follow_suit, bids, tricks, history, current_best_card):
        """
        Ask the player which card from all playable cards she wants to play
        :return: card to play
        """

    def inform_about_env(self, env):
        """
        - Once this method is called, the players know they have been put in a doubly linked list
        and can access their predecessors and successors by calling get_previous_player / get_next_player
        - Additionally they can access the environment with reference "env"
        """
        self.env = env

    def inform_about_bids(self, bids):
        """
        After the bidding, this method is invoked to inform all players
        - about the final bids of all players
        """
        pass

    def inform_about_played_card(self, card, pos_bidding, is_last):
        """
        After each card played, this method is invoked to inform all players
        - about the card played
        - about the player who played the card
        """
        pass

    def finish_round(self, reward, history):
        """
        After the last card played, all players are informed about their reward and the history of the game
        Only relevant for optimized players
        """
        pass

    def finish_interaction(self):
        pass

    def set_next_player(self, player):
        self.next = player

    def set_previous_player(self, player):
        self.previous = player

    def get_hand(self):
        return self.hand

    def set_hand(self, distributed_cards):
        self.hand = distributed_cards

    @property
    def get_id(self):
        return self.id

    @property
    def get_next_player(self):
        return self.next

    @property
    def get_previous_player(self):
        return self.previous

    @property
    def get_pos_bidding(self):
        return self.bidding_pos

    def set_pos_bidding(self, pos):
        self.bidding_pos = pos

    def give_writing_access(self, writer):
        self.writer = writer


class HumanPlayer(Player):
    """
    Wait for user input for both bidding and playing
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps)
        self.last_tricks = None

    # def __repr__(self):
    #     return super().__repr__() + "(Human)"

    def bidding(self, pos_in_bidding, trump_suit_index, bids):
        wait_for_input = True
        user_action = 0
        print("You ({}) hold cards {} and see bids {}".format(str(self),
                                                              self.env.card_names[self.hand],
                                                              bids))
        while wait_for_input:
            try:
                user_action = int(input("Enter a bid from {}: ".format(np.arange(0, self.num_tricks_to_play + 1))))
                if 0 <= user_action <= self.num_tricks_to_play:
                    wait_for_input = False
                else:
                    print("Please chose an index from the given options")
            except ValueError:
                print("Please chose an index from the given options")
        return user_action

    def playing(self, current_trick, pos_in_trick, pos_in_bidding, playable_cards, action_mask,
                trump_suit_index, follow_suit, bids, tricks, history, current_best_card):
        position_in_history = np.argwhere(history[:, 0] == -1)[0][0]
        if current_trick == 0:
            print("Final bidding is {}".format(bids))
        else:
            position = np.flatnonzero(tricks - self.last_tricks)[0]
            print("Last trick: {} played by: {} went to: {}".format(
                self.env.card_names[
                    history[position_in_history - pos_in_trick - self.num_players:position_in_history - pos_in_trick,
                    0]],
                [self.env.players[i] for i in
                 history[position_in_history - pos_in_trick - self.num_players:position_in_history - pos_in_trick, 1]],
                self.env.players[self.env.pos2id_bidding[position]]
            ))
        self.last_tricks = tricks
        print("You ({}) hold cards {}, current trick score is {} and you see cards {} played by {}"
              .format(str(self),
                      self.env.card_names[self.hand],
                      tricks,
                      self.env.card_names[history[position_in_history - pos_in_trick:position_in_history, 0]],
                      [self.env.players[i] for i in
                       history[position_in_history - pos_in_trick:position_in_history, 1]]))
        wait_for_input = True
        user_action = 0
        while wait_for_input:
            try:
                user_action = int(input("Chose a playable card from {} by its index {}: "
                                        .format(self.env.card_names[playable_cards],
                                                np.arange(0, len(playable_cards)))))
                if 0 <= user_action <= len(playable_cards) - 1:
                    wait_for_input = False
                else:
                    print("Please chose an admissible card")
            except ValueError:
                print("Please chose an admissible card")
        return playable_cards[user_action]

    def finish_round(self, reward, history):
        position = np.flatnonzero(self.env.tricks[self.env.pos2id_bidding] - self.last_tricks)[0]
        print("Last trick: {} played by: {} went to: {}".format(
            self.env.card_names[history[-self.num_players:, 0]],
            [self.env.players[i] for i in history[-self.num_players:, 1]],
            self.env.players[self.env.pos2id_bidding[position]]
        ))


class RandomPlayer(Player):
    """
    This player always chooses actions at random
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps)

    # def __repr__(self):
    #     return super().__repr__() + "(Random)"

    # Random strategy: chose value between 0 and number of cards dealt
    def bidding(self, pos_in_bidding, trump_suit_index, bids):
        return self.rng.integers(low=0, high=self.num_tricks_to_play + 1)

    def playing(self, current_trick, pos_in_trick, pos_in_bidding, playable_cards, action_mask,
                trump_suit_index, follow_suit, bids, tricks, history, current_best_card):
        return self.rng.choice(a=playable_cards, replace=False)


class AnalyzePlayer(RandomPlayer):
    """
    This player is used for analyzing different characteristics of the game
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps, master)
        self.count_free_choice = np.zeros(shape=(self.num_tricks_to_play, self.num_tricks_to_play), dtype=int)

    def playing(self, current_trick, pos_in_trick, pos_in_bidding, playable_cards, action_mask,
                trump_suit_index, follow_suit, bids, tricks, history, current_best_card):
        self.count_free_choice[len(self.hand) - 1, len(playable_cards) - 1] += 1
        return self.rng.choice(a=playable_cards, replace=False)

    def get_free_choice(self):
        return self.count_free_choice

    def reset_free_choice(self):
        self.count_free_choice[:] = 0


class RuleBasedPlayer(Player):
    """
    This player chooses actions based on fixed heuristics
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps)

        self.winning_probs = None

    # def __repr__(self):
    #     return super().__repr__() + "(RuleBased)"

    def bidding(self, pos_in_bidding, trump_suit_index, bids):
        # Probability of one card to beat another if played first / second
        wins_if_played_first = 1 - np.sum(self.env.card_order[trump_suit_index], axis=1) / self.num_cards
        wins_if_played_second = np.sum(self.env.card_order[trump_suit_index], axis=0) / self.num_cards
        # Probability to beat one other card independent of position
        self.winning_probs = (wins_if_played_first + wins_if_played_second) / 2
        # Probability to beat NUM_PLAYERS-1 other cards
        self.winning_probs = np.power(self.winning_probs, self.num_players - 1)
        # Normalize probs to sum up to 1
        self.winning_probs = self.winning_probs / self.winning_probs.sum()
        # Normalize probs so that each card has a value of 1/self.num_players
        self.winning_probs = self.winning_probs * self.num_cards / self.num_players
        # Simply sum up over all cards in the hand
        hand_valuation = np.sum(self.winning_probs[self.hand])
        action = int(np.round(hand_valuation, 0))
        # ensure action is in correct bound:
        action = min(action, self.env.num_round)
        action = max(action, 0)

        return action

        # Optionally compare with random action
        # return self.rng.integers(low=0, high=self.num_tricks_to_play + 1)

    def playing(self, current_trick, pos_in_trick, pos_in_bidding, playable_cards, action_mask,
                trump_suit_index, follow_suit, bids, tricks, history, current_best_card):
        # If player has no choice, there is no need to optimize anything
        if len(playable_cards) == 1:
            action = playable_cards[0]
        else:
            # Rank the playable cards by their winning probability
            ranked_actions = playable_cards[np.argsort(self.winning_probs[playable_cards])]
            # Case A: if there are still tricks missing, try to win the trick
            if bids[pos_in_bidding] > tricks[pos_in_bidding]:
                # Play the highest card if the player starts the trick
                if current_best_card is None:
                    action = ranked_actions[-1]
                # Also play the highest card, if it is able to win the current trick
                elif self.env.card_order[trump_suit_index][current_best_card, ranked_actions[-1]] == 1:
                    action = ranked_actions[-1]
                # Play the lowest card, if no card is able to win the trick
                else:
                    action = ranked_actions[0]
            # Case B: if the player has gathered enough (or too many) tricks, try to avoid winning the trick
            else:
                # Play the lowest card if the player starts the trick
                if current_best_card is None:
                    action = ranked_actions[0]
                # If there is a card on the table already, try using the highest card still beaten
                else:
                    action = ranked_actions[0]
                    for possible_action in ranked_actions:
                        # If a higher card can be used without winning the trick, overwrite the action
                        if self.env.card_order[trump_suit_index][current_best_card, possible_action] == 0:
                            action = possible_action
        return action

        # Optionally compare with random action
        # return self.rng.choice(a=playable_cards, replace=False)


class OptimizedPlayer(Player, metaclass=ABCMeta):
    """
    Abstract base class for all child classes using neural networks
    """

    def __init__(self, identification, player_name, hps):
        super().__init__(identification, player_name, hps)

        self.agent_mode = AgentMode.TRAIN

        # Each game starts in bidding phase
        self.game_phase = GamePhase.BIDDING
        self.game_count = 0
        self.play_count = 0
        self.train_count_bidding = 0
        self.train_count_playing = 0

        # Save observation and action in bidding
        self.obs_bidding = None
        self.action_bidding = None

        # Save observation, action and mask in playing
        self.previous_obs = None
        self.previous_action = None
        self.previous_mask = None
        """
        Initialize neural networks using one-hot encoding for BIDDING:
        - pos = position in bidding (num_players)
        - hand = own cards (num_cards)
        - trump = trump suit (num_suit)
        - bids = bids of the other (n-1) players (num_players * num_bids)
        - known_cards = cards known to everyone (num_cards)
        """
        self.input_labels_bidding = ["hand",
                                     "pos",
                                     "trump",
                                     "bids"]
        input_list = [self.num_cards,
                      self.num_players,
                      self.num_suit_including_none,
                      (self.num_players - 1) * (self.num_tricks_to_play + 1)]
        self.input_array_bidding = np.array(input_list, dtype=int)
        self.input_size_bidding = self.input_array_bidding.sum()
        self.output_size_bidding = self.num_tricks_to_play + 1

        """
        Initialize neural networks using one-hot encoding for PLAYING
        - hand = hand
        - best = best card in current trick
        - pos_trick = position in trick
        - pos_bid = position in bidding
        - trump = trump suit
        - follow = follow suit
        - bids = bids of all (n) players
        - tricks = tricks of all (n) players taken so far
        """
        self.input_labels_playing = ["hand",
                                     "best",
                                     "pos_trick",
                                     "pos_bid",
                                     "trump",
                                     "follow",
                                     "bids",
                                     "tricks"]
        input_list = [self.num_cards,
                      self.num_cards,
                      self.num_players,
                      self.num_players,
                      self.num_suit_including_none,
                      self.num_suit_including_none,
                      self.num_players * (self.num_tricks_to_play + 1),
                      self.num_players * (self.num_tricks_to_play + 1)]
        self.input_array_playing = np.array(input_list, dtype=int)
        self.input_size_playing = self.input_array_playing.sum()
        self.output_size_playing = self.num_cards

    def create_input_for_bidding(self, hand, pos_in_bidding, trump_suit_index, bids):
        hand_obs = np.zeros(shape=self.num_cards, dtype=int)
        hand_obs[hand] = 1
        pos_obs = np.zeros(shape=self.num_players, dtype=int)
        pos_obs[pos_in_bidding] = 1
        trump_obs = np.zeros(shape=self.num_suit_including_none, dtype=int)
        trump_obs[trump_suit_index] = 1
        bids_obs = np.zeros(shape=(self.num_players - 1, self.num_tricks_to_play + 1), dtype=int)
        bids_obs[np.arange(pos_in_bidding), bids[:pos_in_bidding]] = 1
        bids_obs = bids_obs.flatten()

        observation = np.concatenate((hand_obs, pos_obs, trump_obs, bids_obs))
        assert len(observation) == self.input_size_bidding

        if self.agent_mode == AgentMode.PLAY and not self.hps['agent']['PLAYER_TYPE'] == "HUMAN":
            obs_dict = self.unwrap_observation_bidding(observation)
            print("Player {} | pos {} | cards {} | trump suit {} | bidding {}".format(
                self.name,
                np.flatnonzero(obs_dict["pos"]),
                np.flatnonzero(obs_dict["hand"]),
                np.flatnonzero(obs_dict["trump"]),
                convert_from_binary(vector=obs_dict["bids"],
                                    units=self.num_players - 1,
                                    size=self.num_tricks_to_play + 1)))

        return observation

    def create_input_for_playing(self, hand, current_trick, pos_in_trick, pos_in_bidding, trump_suit_index,
                                 follow_suit, bids, tricks, best):
        hand_obs = np.zeros(shape=self.num_cards, dtype=int)
        hand_obs[hand] = 1
        best_obs = np.zeros(shape=self.num_cards, dtype=int)
        if best is not None:
            best_obs[best] = 1
        pos_trick_obs = np.zeros(shape=self.num_players, dtype=int)
        pos_trick_obs[pos_in_trick] = 1
        pos_bid_obs = np.zeros(shape=self.num_players, dtype=int)
        pos_bid_obs[pos_in_bidding] = 1
        trump_obs = np.zeros(shape=self.num_suit_including_none, dtype=int)
        trump_obs[trump_suit_index] = 1
        follow_obs = np.zeros(shape=self.num_suit_including_none, dtype=int)
        if follow_suit is not None:
            follow_obs[follow_suit] = 1
        bids_obs = np.zeros(shape=(self.num_players, self.num_tricks_to_play + 1), dtype=int)
        bids_obs[np.arange(self.num_players), bids] = 1
        bids_obs = bids_obs.flatten()
        tricks_obs = np.zeros(shape=(self.num_players, self.num_tricks_to_play + 1), dtype=int)
        tricks_obs[np.arange(self.num_players), tricks] = 1
        tricks_obs = tricks_obs.flatten()

        observation = np.concatenate((hand_obs, best_obs, pos_trick_obs, pos_bid_obs, trump_obs,
                                      follow_obs, bids_obs, tricks_obs))

        assert len(observation) == self.input_size_playing

        return observation

    def unwrap_observation_bidding(self, observation):
        obs_dict = {}
        high = 0
        for i, l in enumerate(self.input_labels_bidding):
            low = high
            high = low + self.input_array_bidding[i]
            obs_dict[l] = observation[low:high]

        return obs_dict

    def unwrap_observation_playing(self, observation):
        obs_dict = {}
        high = 0
        for i, l in enumerate(self.input_labels_playing):
            low = high
            high = low + self.input_array_playing[i]
            obs_dict[l] = observation[low:high]

        return obs_dict

    def print_observation(self, observation):
        obs_dict = self.unwrap_observation_playing(observation)
        hand_flat = np.flatnonzero(obs_dict["hand"])
        output_string = "Player {} | pos in bid/trick {}/{} | bids/tricks {}/{} | cards {} | playable {}".format(
            self.name,
            np.flatnonzero(obs_dict["pos_bid"]),
            np.flatnonzero(obs_dict["pos_trick"]),
            convert_from_binary(vector=obs_dict["bids"],
                                units=self.num_players,
                                size=self.num_tricks_to_play + 1),
            convert_from_binary(vector=obs_dict["tricks"],
                                units=self.num_players,
                                size=self.num_tricks_to_play + 1),
            hand_flat,
            np.flatnonzero(self.env.get_action_mask(hand_flat, np.flatnonzero(obs_dict["follow"])[0])))

        print(output_string)

    def set_agent_mode(self, agent_mode):
        self.agent_mode = agent_mode

    def transition(self):
        if self.game_phase == GamePhase.BIDDING:
            self.game_phase = GamePhase.FIRST_PLAYING
        elif self.game_phase == GamePhase.FIRST_PLAYING:
            self.play_count += 1
            self.game_phase = GamePhase.PLAYING
        elif self.game_phase == GamePhase.PLAYING:
            self.play_count += 1
        elif self.game_phase == GamePhase.FINISH:
            self.game_count += 1
            self.game_phase = GamePhase.BIDDING

    @abstractmethod
    def train_bidding(self):
        pass

    @abstractmethod
    def train_playing(self):
        pass
