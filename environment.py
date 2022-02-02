import numpy as np
from abc import ABC, abstractmethod


class TrickTakingGame(ABC):
    """
    Base class which provides the common methods for all trick-taking games
    """

    def __init__(self, hps):

        self.SUIT_NAMES = ['Spades', 'Hearts', 'Clubs', 'Diamonds']
        self.SUIT_SYMBOLS = ['♠', '♥', '♣', '♦']
        self.RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

        self.rng = np.random.default_rng(hps['env']['SEED'])

        self.hps = hps
        self.num_suit = hps['env']['SUIT']
        self.num_cards_per_suit = hps['env']['CARDS_PER_SUIT']
        self.num_players = hps['env']['PLAYERS']

        # Variables to be set by child classes
        self.num_cards = None
        self.card_names = None
        self.card_symbols = None
        self.card_order = None
        self.suit_from_card = None
        self.cards_to_be_followed_from_suit = None

        self.players = None
        self.trump_suit_index = None
        self.bids = None
        self.tricks = None
        self.id2pos_bidding = None
        self.pos2id_bidding = None
        self.id2pos_playing = None
        self.pos2id_playing = None
        self.num_round = None
        self.ground_truth = None
        self.unable_to_follow = None
        self.lead_suit_played = None
        self.follow_suit = None
        self.current_trick_winner_bid_pos = None
        self.current_trick_winner_card = None

        self.print_statements = False

        self.random_reward = np.full(shape=31, fill_value=0.0)
        self.rule_based_reward = np.full(shape=31, fill_value=0.0)
        self.rule_based_acc = np.full(shape=31, fill_value=0.0)

        # Additional variables for architecture B
        self.start_player_round = None
        self.history = None
        self.bids_gathered = None
        self.current_player = None
        self.current_player_position = None
        self.current_trick = None
        self.start_player_trick = None
        self.playable_cards = None
        self.last_trick_winner = None

    def create_cards(self):
        """
        Create a list of card names
        :return:
        """
        card_names = np.empty(self.num_cards, dtype=object)
        card_symbols = np.empty(self.num_cards, dtype=object)
        for suit_index in range(self.num_suit):
            card_names[self.num_cards_per_suit * suit_index:self.num_cards_per_suit * (suit_index + 1)] = \
                [self.SUIT_NAMES[suit_index] + " " + str(j) for j in range(1, self.num_cards_per_suit + 1)]
            card_symbols[self.num_cards_per_suit * suit_index:self.num_cards_per_suit * (suit_index + 1)] = \
                [str(self.SUIT_SYMBOLS[suit_index]) + str(j) for j in range(1, self.num_cards_per_suit + 1)]
        return card_names, card_symbols

    def calculate_card_order(self):
        """
        Lookup table to quickly find the better card
        0 = first card played wins the trick
        1 = second card played wins the trick
        :return: dictionary containing the order for each possible trump suit
        """
        order = {}

        compare = np.zeros(shape=(self.num_cards, self.num_cards), dtype=int)
        # A: Ordinary cards are only beaten by higher cards of the same suit
        for suit_index in range(self.num_suit):
            for rank in range(0, self.num_cards_per_suit - 1):
                compare[self.num_cards_per_suit * suit_index + rank,
                self.num_cards_per_suit * suit_index + rank + 1:
                self.num_cards_per_suit * suit_index + self.num_cards_per_suit] = 1

        # B: Trump beats all ordinary suits (even when played second)
        for suit_index in range(self.num_suit):
            compare_special = compare.copy()
            # Beat all cards of suits withs lower index
            compare_special[0:suit_index * self.num_cards_per_suit,
            suit_index * self.num_cards_per_suit:(suit_index + 1) * self.num_cards_per_suit] = 1
            # Beat all cards of suits with higher index
            compare_special[(suit_index + 1) * self.num_cards_per_suit:self.num_cards,
            suit_index * self.num_cards_per_suit:(suit_index + 1) * self.num_cards_per_suit] = 1
            order[suit_index] = compare_special

        # Keep another compare table for the case without trump suit
        order[self.num_suit] = compare

        return order

    def calculate_suit_from_card(self):
        """
        Extract information about the suit given a specific card
        :return:
        """
        suit_from_card = np.empty(shape=self.num_cards, dtype=int)
        for suit_index in range(self.num_suit):
            suit_from_card[suit_index * self.num_cards_per_suit:(suit_index + 1) * self.num_cards_per_suit] = suit_index

        return suit_from_card

    def calculate_cards_to_be_followed_from_suit(self):
        """
        Calculate the cards which need to be followed
        :return:
        """
        cards_to_be_followed_from_suit = {}

        for suit_index in range(self.num_suit):
            mask = np.zeros(shape=self.num_cards, dtype=int)
            mask[suit_index * self.num_cards_per_suit:(suit_index + 1) * self.num_cards_per_suit] = 1
            cards_to_be_followed_from_suit[suit_index] = mask

        return cards_to_be_followed_from_suit

    def set_players(self, players):
        """
        Assign instances of players to the current game instance
        :param players: list of players
        """
        self.players = players
        for p in players:
            p.inform_about_env(self)

    def set_print_statements(self, print_statements):
        self.print_statements = print_statements

    @abstractmethod
    def initialize_game_round(self):
        """
        This is done in the child class:
        Choose a trump suit and distribute cards among the players
        """

    def distribute_cards(self, shuffled_cards):
        """
        Distribute cards among players
        """
        for i in range(self.num_players):
            current_hand = shuffled_cards[i * self.num_round:(i + 1) * self.num_round]
            current_hand.sort()
            self.players[i].set_hand(current_hand)
            # Move cards from the deck to the players hand
            self.ground_truth[current_hand, self.hps['env']['GT_DECK']] = 0
            self.ground_truth[current_hand, self.hps['env']['GT_HAND'] + self.players[i].get_pos_bidding] = 1
            if self.print_statements and not self.hps['agent']['PLAYER_TYPE'] == "HUMAN":
                print("{} holds {} = {}".format(self.players[i], self.card_symbols[current_hand], current_hand))
        assert (self.ground_truth.sum(axis=1) == np.ones(shape=self.ground_truth.shape[0])).all()

    def get_action_mask(self, hand, follow_suit):
        """
        Provide the players with the cards they are allowed to play
        :param hand:
        :param follow_suit:
        :return:
        """
        hand_one_hot = np.zeros(shape=self.num_cards, dtype=int)
        hand_one_hot[hand] = 1
        # Case 1: Player doesn't need to follow suit
        if follow_suit == self.num_suit:
            action_mask = hand_one_hot
        # Case 2: Player needs to follow suit = get intersection of own cards and suit to be followed
        elif np.logical_and(hand_one_hot, self.cards_to_be_followed_from_suit[follow_suit]).any():
            action_mask = np.logical_and(hand_one_hot, self.cards_to_be_followed_from_suit[follow_suit])
        # Case 3: Player can't follow suit = no restrictions on what to play
        else:
            action_mask = hand_one_hot
        return action_mask

    def process_card_played(self, card_played, current_trick_winner_bid_pos, current_trick_winner_card, follow_suit,
                            lead_suit_played, current_pos_trick, current_pos_bidding, current_pos_history, history,
                            ground_truth):
        """
        Process the card by adding it to both history and ground truth
        Also calculate the resulting follow suit and the trick winner
        """
        if not lead_suit_played:
            follow_suit = self.suit_from_card[card_played]
            # Case A: ordinary cards set the suit immediately
            if follow_suit < self.num_suit:
                lead_suit_played = True
            # Case B: special cards of type "num_suit" have no effect on follow_suit
            elif follow_suit == self.num_suit:
                lead_suit_played = False
            # Case C: special cards of type "num_suit + 1" will enable everyone to play arbitrary cards
            elif follow_suit == (self.num_suit + 1):
                follow_suit = self.num_suit
                lead_suit_played = True
            else:
                raise Exception("follow_suit should never be above " + str(self.num_suit + 1))

        # Calculate who currently holds the highest card
        # The first player to play a card always wins the trick (temporarily)
        if current_pos_trick == 0:
            current_trick_winner_bid_pos = current_pos_bidding
            current_trick_winner_card = card_played
        # All subsequent players must challenge the highest card played so far:
        elif self.card_order[self.trump_suit_index][current_trick_winner_card, card_played] == 1:
            current_trick_winner_bid_pos = current_pos_bidding
            current_trick_winner_card = card_played

        # cards_in_trick[current_player_position] = card_played

        # Add card to history and to the ground truth:
        history[current_pos_history, 0] = card_played
        history[current_pos_history, 1] = current_pos_bidding
        ground_truth[card_played, self.hps['env']['GT_HAND'] + current_pos_bidding] = 0  # remove card from players hand
        ground_truth[card_played, self.hps['env']['GT_PLAYED'] + current_pos_bidding] = 1  # add card to played cards
        assert (ground_truth.sum(axis=1) == np.ones(shape=ground_truth.shape[0])).all()
        return current_trick_winner_bid_pos, current_trick_winner_card, follow_suit, lead_suit_played, history, \
               ground_truth

    @abstractmethod
    def evaluate_round(self, bids, tricks):
        """
        Convert (predicted) bids and (actual) tricks to a reward
        :return: reward vector
        """

    @abstractmethod
    def normalize_reward(self, reward, num_round):
        """
        Shifts the reward into a range from [-1, 1]
        :param reward:
        :param num_round:
        :return: normalized reward
        """
    @abstractmethod
    def denormalize_reward(self, reward, num_round):
        """
        Shifts the reward from range [-1, 1] to original range
        :param reward:
        :param num_round:
        :return: unnormalized reward
        """

    def upper_bound_reward(self, num_round):
        """
        Calculate an upper bound for the normalized reward in case all players play perfectly
        :param num_round:
        :return:
        """
        # If all players play perfectly, they always predict the correct number of tricks and get an equal share
        bids = np.full(shape=self.num_players, fill_value=num_round // self.num_players, dtype=int)
        tricks = np.full(shape=self.num_players, fill_value=num_round // self.num_players, dtype=int)
        rewards = self.evaluate_round(bids, tricks)
        if self.hps['env']['NORMALIZE_REWARD']:
            rewards = self.normalize_reward(reward=rewards, num_round=num_round)
        return rewards

    def print_history(self, history):
        output = np.zeros_like(history, dtype=object)
        output[:, 0] = history[:, 1]
        output[:, 1] = self.card_names[history[:, 0]]
        for n in range(self.num_round):
            print("Trick {}: {}".format((n + 1),
                                        output[self.num_players * n:self.num_players * (n + 1)].flatten()))


    """
    Architecture A: return control at the end of the game round
    """
    def game_round(self, num_round, start_player_round):
        """
        Run a whole round of the game, consisting of bidding and playing phase
        :param num_round: number of cards per player
        :param start_player_round: player which starts the bidding and the first trick
        :return:
        """
        self.num_round = num_round

        """
        the ground truth contains one card per row
        the column specifies where the card is currently located
        0: in the deck (hidden to everyone) >> this is how the game starts [index in 4 player game: 0]
        1 to num_players (hidden to everyone except the owner): in players A,B,C...'s hand [index in 4 player game: 6-9]
        num_players+1: played by nature (which can only be the trump card) [index in 4 player game: 5]
        num_players+2 to 2*num_players+1: played by player A,B,C,... [index in 4 player game: 1-4]
        """
        self.ground_truth = np.zeros(shape=(self.num_cards, self.hps['env']['GT_SIZE']), dtype=int)
        self.ground_truth[:, self.hps['env']['GT_DECK']] = 1

        # For each player, keep track of the suits she is unable to follow
        self.unable_to_follow = np.zeros(shape=(self.num_suit, self.num_players), dtype=int)

        self.id2pos_bidding = np.roll(np.arange(self.num_players), start_player_round.get_id)
        self.pos2id_bidding = np.roll(np.arange(self.num_players), -start_player_round.get_id)

        """
        Initialization phase:
        - Inform players about their position in bidding:
        - Initialize history
        """
        current_player = start_player_round
        current_player_position = 0
        while current_player_position < self.num_players:
            current_player.set_pos_bidding(current_player_position)
            current_player_position += 1
            current_player = current_player.get_next_player

        assert num_round <= self.num_cards // self.num_players
        history = np.full(shape=(num_round * self.num_players, 2), fill_value=-1, dtype=int)
        if self.print_statements:
            print('--------------------------------------------------------------------------------------------------')
            print("{} starts round {} of game <{}>".format(start_player_round, num_round, self))
        self.initialize_game_round()

        """
        Bidding phase
        """
        self.bids = np.full(shape=self.num_players, fill_value=-1, dtype=int)
        bids_gathered = 0
        current_player = start_player_round
        current_player_position = 0
        while bids_gathered < self.num_players:
            self.bids[current_player_position] = current_player.bidding(pos_in_bidding=current_player_position,
                                                                        trump_suit_index=self.trump_suit_index,
                                                                        bids=self.bids)
            bids_gathered += 1
            current_player_position += 1
            current_player = current_player.get_next_player
        # Inform players about the final bids:
        for p in self.players:
            p.inform_about_bids(self.bids)
        # Re-arrange bids in order to have bidding of player 1 at position 1
        self.bids = np.roll(self.bids, start_player_round.get_id)

        """
        Trick-taking phase
        """
        self.tricks = np.zeros(shape=self.num_players, dtype=int)
        current_trick = 0
        start_player_trick = start_player_round
        while current_trick < num_round:
            start_player_trick, history = self.game_trick(start_player_trick=start_player_trick,
                                                          current_trick=current_trick,
                                                          history=history)
            current_trick += 1
            self.tricks[start_player_trick.get_id] += 1

        if self.print_statements:
            print("Card History: {} = {}".format(self.card_symbols[history[:, 0]], history[:, 0]))
            print("Player History: {} = {}".format([self.players[self.pos2id_bidding[i]] for i in history[:, 1]],
                                                   history[:, 1]))

        """
        Evaluation phase
        """
        rewards = self.evaluate_round(self.bids, self.tricks)
        rewards_normalized = self.normalize_reward(reward=rewards, num_round=num_round)
        if self.print_statements:
            print("Bids / Tricks (in player order): {} / {}".format(self.bids, self.tricks))
            print("Score Absolute / Normalized (in player order): {} / {}".format(np.round(rewards, 2), np.round(rewards_normalized, 2)))

        # Inform players about the final reward
        for i, p in enumerate(self.players):
            if self.hps['env']['NORMALIZE_REWARD']:
                p.finish_round(reward=rewards_normalized[i], history=history)
            else:
                p.finish_round(reward=rewards[i], history=history)

        # Return the results to the game master
        if self.hps['env']['NORMALIZE_REWARD']:
            return self.bids, self.tricks, rewards_normalized
        else:
            return self.bids, self.tricks, rewards

    def game_trick(self, start_player_trick, current_trick, history):
        """
        Run a single trick inside a game round
        :param start_player_trick:
        :param current_trick:
        :param history:
        :return:
        """
        current_pos_trick = 0  # Position of player in trick order
        current_player = start_player_trick
        self.follow_suit = self.num_suit  # Index num_suit stands for not having to follow suit
        self.lead_suit_played = False
        self.current_trick_winner_card = None
        while current_pos_trick < self.num_players:
            current_pos_bidding = current_player.get_pos_bidding
            current_hand = current_player.get_hand()
            action_mask = self.get_action_mask(hand=current_hand, follow_suit=self.follow_suit)
            playable_cards = np.flatnonzero(action_mask)
            assert len(playable_cards) >= 1
            # each player is asked to play a card given the suit she needs to follow and the whole history so far
            # pos2id_bidding will shift id-sorted values to pos_bidding_sorted values
            card_played = current_player.playing(current_trick=current_trick,
                                                 pos_in_trick=current_pos_trick,
                                                 pos_in_bidding=current_pos_bidding,
                                                 playable_cards=playable_cards,
                                                 action_mask=action_mask,
                                                 trump_suit_index=self.trump_suit_index,
                                                 follow_suit=self.follow_suit,
                                                 bids=self.bids[self.pos2id_bidding],
                                                 tricks=self.tricks[self.pos2id_bidding],
                                                 history=history,
                                                 current_best_card=self.current_trick_winner_card)
            assert card_played >= 0
            assert card_played in playable_cards
            current_player.set_hand(current_hand[current_hand != card_played])

            # If a lead suit had been played before, we can now check if the current player was unable to follow it
            # This is the case, if A) the card played is of a different suit than the one to follow
            # B) + C) both the card played and the card to follow are of regular type
            if self.lead_suit_played and self.suit_from_card[card_played] != self.follow_suit and \
                    self.follow_suit < self.num_suit and self.suit_from_card[card_played] < self.num_suit:
                self.unable_to_follow[self.follow_suit, current_pos_bidding] = 1

            self.current_trick_winner_bid_pos, self.current_trick_winner_card, self.follow_suit, \
            self.lead_suit_played, history, self.ground_truth = self.process_card_played(
                card_played=card_played,
                current_trick_winner_bid_pos=self.current_trick_winner_bid_pos,
                current_trick_winner_card=self.current_trick_winner_card,
                follow_suit=self.follow_suit,
                lead_suit_played=self.lead_suit_played,
                current_pos_trick=current_pos_trick,
                current_pos_bidding=current_pos_bidding,
                current_pos_history=self.num_players * current_trick + current_pos_trick,
                history=history,
                ground_truth=self.ground_truth)

            # Inform all players about the played card and the player who played it
            for p in self.players:
                p.inform_about_played_card(card=card_played,
                                           pos_bidding=current_pos_bidding,
                                           is_last=history[-1, 1] != -1)

            current_pos_trick += 1
            current_player = current_player.get_next_player

        trick_winner = self.players[self.pos2id_bidding[self.current_trick_winner_bid_pos]]

        return trick_winner, history

    """
    Architecture B: always return control to MASTER player
    """

    def prepare_trick(self):
        self.current_player = self.start_player_trick
        self.current_player_position = 0
        self.current_trick_winner_card = None
        self.current_trick_winner_bid_pos = None
        self.follow_suit = self.num_suit
        self.lead_suit_played = False

    def bidding_master(self, bid):
        self.bids[self.current_player_position] = bid
        self.bids_gathered += 1
        self.current_player_position += 1
        self.current_player = self.current_player.get_next_player

    def bidding_others(self):
        bid = self.current_player.bidding(
            pos_in_bidding=self.current_player_position,
            trump_suit_index=self.trump_suit_index,
            bids=self.bids)
        self.bidding_master(bid)

    def playing_master(self, card_played):
        current_pos_bidding = self.current_player.get_pos_bidding
        assert card_played >= 0
        assert card_played in self.playable_cards
        self.current_player.set_hand(self.current_player.get_hand()[self.current_player.get_hand() != card_played])

        if self.lead_suit_played and self.suit_from_card[card_played] != self.follow_suit and \
                self.follow_suit < self.num_suit and self.suit_from_card[card_played] < self.num_suit:
            self.unable_to_follow[self.follow_suit, current_pos_bidding] = 1

        self.current_trick_winner_bid_pos, self.current_trick_winner_card, self.follow_suit, \
        self.lead_suit_played, self.history, self.ground_truth = self.process_card_played(
            card_played=card_played,
            current_trick_winner_bid_pos=self.current_trick_winner_bid_pos,
            current_trick_winner_card=self.current_trick_winner_card,
            follow_suit=self.follow_suit,
            lead_suit_played=self.lead_suit_played,
            current_pos_trick=self.current_player_position,
            current_pos_bidding=current_pos_bidding,
            current_pos_history=self.num_players * self.current_trick + self.current_player_position,
            history=self.history,
            ground_truth=self.ground_truth)

        # Inform all players about the played card and the player who played it
        for p in self.players:
            p.inform_about_played_card(card=card_played,
                                       pos_bidding=current_pos_bidding,
                                       is_last=self.history[-1, 1] != -1)

        self.current_player_position += 1
        self.current_player = self.current_player.get_next_player

    def playing_others(self):
        current_pos_bidding = self.current_player.get_pos_bidding
        current_hand = self.current_player.get_hand()
        action_mask = self.get_action_mask(hand=current_hand, follow_suit=self.follow_suit)
        self.playable_cards = np.flatnonzero(action_mask)
        assert len(self.playable_cards) >= 1
        card_played = self.current_player.playing(current_trick=self.current_trick,
                                                  pos_in_trick=self.current_player_position,
                                                  pos_in_bidding=current_pos_bidding,
                                                  playable_cards=self.playable_cards,
                                                  action_mask=action_mask,
                                                  trump_suit_index=self.trump_suit_index,
                                                  follow_suit=self.follow_suit,
                                                  bids=self.bids[self.pos2id_bidding],
                                                  tricks=self.tricks[self.pos2id_bidding],
                                                  history=self.history,
                                                  current_best_card=self.current_trick_winner_card)
        self.playing_master(card_played)

    def from_initialization_to_first_bid(self, num_round, start_player_round):
        """
        Initialization of game round until MASTER bidding
        """
        """
        Step 1: Initialization
        """
        self.bids = np.full(shape=self.num_players, fill_value=-1, dtype=int)
        self.tricks = np.zeros(shape=self.num_players, dtype=int)
        self.num_round = num_round
        self.current_trick = 0
        self.start_player_round = self.start_player_trick = start_player_round

        self.ground_truth = np.zeros(shape=(self.num_cards, self.hps['env']['GT_SIZE']), dtype=int)
        self.ground_truth[:, self.hps['env']['GT_DECK']] = 1

        self.unable_to_follow = np.zeros(shape=(self.num_suit, self.num_players), dtype=int)

        self.id2pos_bidding = np.roll(np.arange(self.num_players), start_player_round.get_id)
        self.pos2id_bidding = np.roll(np.arange(self.num_players), -start_player_round.get_id)

        current_player = start_player_round
        current_player_position = 0
        while current_player_position < self.num_players:
            current_player.set_pos_bidding(current_player_position)
            current_player_position += 1
            current_player = current_player.get_next_player

        assert num_round <= self.num_cards // self.num_players
        self.history = np.full(shape=(num_round * self.num_players, 2), fill_value=-1, dtype=int)
        if self.print_statements:
            print('--------------------------------------------------------------------------------------------------')
            print("{} starts round {} of game <{}>".format(start_player_round, num_round, self))
        self.initialize_game_round()

        """
        Step 2: Bidding Part A: initialize bidding
        """
        self.bids_gathered = 0
        self.current_player = start_player_round
        self.current_player_position = 0

        """
        Step 3: Bidding Part B: get all bids previous to MASTER player
        """
        while self.current_player is not self.players[self.hps['agent']['MASTER_INDEX']]:
            self.bidding_others()

        """
        Step 4: Bidding Part C: ask master for bidding
        """
        # Ask a regular DQN player what to do in the current situation:
        dqn_player = self.players[self.hps['agent']['MASTER_INDEX']].get_next_player
        observation = dqn_player.create_input_for_bidding(
            hand=self.current_player.get_hand(),
            pos_in_bidding=self.current_player_position,
            trump_suit_index=self.trump_suit_index,
            bids=self.bids)
        q_values = dqn_player.bidding_actor.get_all_q_values(observation)
        # If rewards have been normalized, denormalize them
        if self.hps['env']['NORMALIZE_REWARD']:
            q_values = self.denormalize_reward(q_values, self.num_round)
        q_values = np.round(q_values, decimals=1)
        return self.current_player_position, self.SUIT_SYMBOLS[self.trump_suit_index], self.bids[self.bids != -1], \
               self.card_symbols[self.current_player.get_hand()], q_values

    def from_first_bid_to_first_trick(self, bid_from_player):
        """
        MASTER bidding until MASTER first playing
        """
        """
        Step 1: Bidding phase Part D: process the bid from MASTER
        """
        self.bidding_master(bid_from_player)

        """
        Step 2: Bidding phase Part C: get all remaining bids
        """
        while self.bids_gathered < self.num_players:
            self.bidding_others()

        self.bids = np.roll(self.bids, self.start_player_round.get_id)

        """
        Step 3: Playing phase part A: initialize new trick
        """
        self.prepare_trick()

        """
        Step 4: Playing phase part B: get all cards previous to MASTER player
        """
        while self.current_player is not self.players[self.hps['agent']['MASTER_INDEX']]:
            self.playing_others()

        """
        Step 4: Playing phase part C: ask the MASTER player for a card
        """
        current_hand = self.current_player.get_hand()
        action_mask = self.get_action_mask(hand=current_hand, follow_suit=self.follow_suit)
        self.playable_cards = np.flatnonzero(action_mask)
        assert len(self.playable_cards) >= 1

        trick_current = self.history[self.num_players * self.current_trick:self.num_players * self.current_trick + self.current_player_position,:]

        # Ask a regular DQN player what to do in the current situation:
        dqn_player = self.players[self.hps['agent']['MASTER_INDEX']].get_next_player

        observation = dqn_player.create_input_for_playing(
            hand=current_hand,
            current_trick=self.current_trick,
            pos_in_trick=self.current_player_position,
            pos_in_bidding=self.current_player.get_pos_bidding,
            trump_suit_index=self.trump_suit_index,
            follow_suit=self.follow_suit,
            bids=self.bids[self.pos2id_bidding],
            tricks=self.tricks[self.pos2id_bidding],
            best=self.current_trick_winner_card)

        q_values_masked = dqn_player.playing_actor.get_all_q_values_masked(observation, action_mask)
        if self.hps['env']['NORMALIZE_REWARD']:
            q_values_masked = self.denormalize_reward(q_values_masked, self.num_round)
        q_values_masked = np.round(q_values_masked, decimals=1)

        return self.bids[self.pos2id_bidding], self.current_player_position, trick_current, self.card_symbols[
            self.playable_cards], q_values_masked

    def from_trick_to_trick(self, card_from_player):
        """
        Step 1: Playing phase part D: process the card from MASTER player
        """
        self.playing_master(self.playable_cards[card_from_player])

        """
        Step 2: Playing phase part E: finish the trick
        """
        while self.current_player_position < self.num_players:
            self.playing_others()

        """
        Step 3: Playing phase part F: Evaluate the trick
        """
        self.start_player_trick = self.players[self.pos2id_bidding[self.current_trick_winner_bid_pos]]
        self.tricks[self.start_player_trick.get_id] += 1
        self.last_trick_winner = self.start_player_trick
        self.current_trick += 1
        """
        Step 4: Playing phase part A: Prepare next trick
        """
        self.prepare_trick()
        """
        Step 4: Playing phase part B: get all cards previous to MASTER player
        """
        while self.current_player is not self.players[self.hps['agent']['MASTER_INDEX']]:
            self.playing_others()

        """
        Step 5: Playing phase part C: ask MASTER player for a card
        """
        current_hand = self.current_player.get_hand()
        action_mask = self.get_action_mask(hand=current_hand, follow_suit=self.follow_suit)
        self.playable_cards = np.flatnonzero(action_mask)
        assert len(self.playable_cards) >= 1

        # Ask a regular DQN player what to do in the current situation:
        dqn_player = self.players[self.hps['agent']['MASTER_INDEX']].get_next_player

        observation = dqn_player.create_input_for_playing(
            hand=current_hand,
            current_trick=self.current_trick,
            pos_in_trick=self.current_player_position,
            pos_in_bidding=self.current_player.get_pos_bidding,
            trump_suit_index=self.trump_suit_index,
            follow_suit=self.follow_suit,
            bids=self.bids[self.pos2id_bidding],
            tricks=self.tricks[self.pos2id_bidding],
            best=self.current_trick_winner_card)

        q_values_masked = dqn_player.playing_actor.get_all_q_values_masked(observation, action_mask)
        if self.hps['env']['NORMALIZE_REWARD']:
            q_values_masked = self.denormalize_reward(q_values_masked, self.num_round)
        q_values_masked = np.round(q_values_masked, decimals=1)
        trick_last = self.history[self.num_players * (self.current_trick - 1):self.num_players * self.current_trick, :]
        trick_current = self.history[
                        self.num_players * self.current_trick:self.num_players * self.current_trick + self.current_player_position,
                        :]
        return trick_last, self.last_trick_winner, trick_current, self.current_player_position, self.card_symbols[
            self.playable_cards], self.tricks[self.pos2id_bidding], q_values_masked

    def from_last_trick_to_evaluation(self, card_from_player):
        assert self.current_trick == self.num_round - 1

        """
        Step 1: Playing phase part D: process the card from MASTER player
        """
        self.playing_master(self.playable_cards[card_from_player])

        """
        Step 2: Playing phase part E: finish the trick
        """
        while self.current_player_position < self.num_players:
            self.playing_others()

        """
        Step 3: Playing phase part F: Evaluate the trick
        """
        self.start_player_trick = self.players[self.pos2id_bidding[self.current_trick_winner_bid_pos]]
        self.tricks[self.start_player_trick.get_id] += 1
        self.last_trick_winner = self.start_player_trick
        self.current_trick += 1

        """
        Step 4: Evaluate the whole round
        """
        reward = self.evaluate_round(self.bids, self.tricks)
        reward_normalized = self.normalize_reward(reward=reward, num_round=self.num_round)
        if self.print_statements:
            print("Bids / Tricks (in player order): {} / {}".format(self.bids, self.tricks))
            print("Score Absolute / Normalized (in player order): {} / {}".format(np.round(reward, 2), np.round(reward_normalized, 2)))

        # Inform players about the final reward
        # for i, p in enumerate(self.players):
        #     p.finish_round(reward=reward_normalized[i], history=self.history)

        # Return the results the MASTER player
        trick_last = self.history[self.num_players * (self.current_trick - 1):self.num_players * self.current_trick, :]
        return trick_last, self.last_trick_winner, self.bids[self.pos2id_bidding], self.tricks[self.pos2id_bidding], reward[self.pos2id_bidding]


class Spades(TrickTakingGame):
    """
    This class inherits from TrickTakingGame and implements special methods for the Spades environment
    """

    def __init__(self, hps):
        self.REWARD_PER_BID = 10
        self.REWARD_PER_ADDITIONAL_TRICK = 1

        super(Spades, self).__init__(hps)

        self.num_cards = self.num_suit * self.num_cards_per_suit
        self.card_names, self.card_symbols = self.create_cards()
        self.card_order = self.calculate_card_order()
        self.suit_from_card = self.calculate_suit_from_card()
        self.cards_to_be_followed_from_suit = self.calculate_cards_to_be_followed_from_suit()

        self.full_deck = 52
        # Random rewards for round 1 to 13 for the 4-player game with a full deck
        if self.hps['env']['CARDS'] == self.full_deck and self.hps['env']['PLAYERS'] == 4:
            self.rule_based_acc[1:6] = [.85, .69, .59, .52, .49]
            self.rule_based_acc[6:11] = [.47, .45, .43, .42, .41]
            self.rule_based_acc[11:14] = [.41, .41, .41]

            if self.hps['env']['NORMALIZE_REWARD']:
                self.random_reward[1:6] = [-.24, -.29, -.33, -.34, -.36]
                self.random_reward[6:11] = [-.37, -.38, -.39, -.39, -.39]
                self.random_reward[11:14] = [-.40, -.40, -.40]

                self.rule_based_reward[1:6] = [.11, .08, .08, .09, .09]
                self.rule_based_reward[6:11] = [.09, .09, .09, .09, .09]
                self.rule_based_reward[11:14] = [.09, .09, .09]

        if self.hps['env']['CARDS'] == 4:
            if self.hps['env']['NORMALIZE_REWARD']:
                self.random_reward[2] = -.02
            else:
                self.random_reward[2] = -.30

    def __repr__(self):
        return "Spades"

    def initialize_game_round(self):
        # In Spades, the trump_suit_index is always Spades with index 0
        self.trump_suit_index = 0

        shuffled_cards = self.rng.choice(a=self.num_cards, size=self.num_players * self.num_round, replace=False)
        self.distribute_cards(shuffled_cards)

    def evaluate_round(self, bids, tricks):
        diff = tricks - bids
        results = np.zeros(shape=len(bids))
        # Set score if enough tricks were taken:
        results[diff >= 0] = self.REWARD_PER_BID * bids[diff >= 0] + self.REWARD_PER_ADDITIONAL_TRICK * diff[diff >= 0]
        # Set score if not enough tricks were taken
        results[diff < 0] = - self.REWARD_PER_BID * bids[diff < 0]
        return results

    def normalize_reward(self, reward, num_round):
        low = -num_round * self.REWARD_PER_BID
        high = num_round * self.REWARD_PER_BID
        # return (reward - low) / (high - low)  # [0,1]
        return -1 + 2 * (reward - low) / (high - low)   # [-1,1]

    def denormalize_reward(self, reward, num_round):
        low = -num_round * self.REWARD_PER_BID
        high = num_round * self.REWARD_PER_BID
        return low + (reward + 1) * (high - low) / 2


class OhHell(Spades):
    """
    Implementation of the Oh Hell environment (also known as Nomination Whist)
    """

    def __init__(self, hps):
        super(OhHell, self).__init__(hps)

        self.REWARD_FOR_CORRECT_PREDICTION = 10
        self.REWARD_PER_CORRECT_TRICK = 1
        self.trump_card = None
        self.full_deck = 52
        # Random rewards for round 1 to 12 for the 4-player game with a full deck
        if self.hps['env']['CARDS'] == self.full_deck and self.hps['env']['PLAYERS'] == 4:
            self.rule_based_acc[1:6] = [.85, .69, .60, .53, .50]
            self.rule_based_acc[6:11] = [.48, .46, .44, .43, .43]
            self.rule_based_acc[11:13] = [.43, .43]

            if self.hps['env']['NORMALIZE_REWARD']:
                self.random_reward[1:6] = [-.07, -.42, -.59, -.68, -.75]
                self.random_reward[6:11] = [-.79, -.83, -.85, -.87, -.89]
                self.random_reward[11:13] = [-.90, -.91]

                self.rule_based_reward[1:6] = [.58, .19, -.04, -.19, -.27]
                self.rule_based_reward[6:11] = [-.33, -.38, -.42, -.45, -.47]
                self.rule_based_reward[11:13] = [-.49, -.50]

    def __repr__(self):
        return "Oh Hell"

    def initialize_game_round(self):
        # Make sure there are enough cards to be able to draw a trump card
        assert self.num_players * self.num_round < self.num_cards
        shuffled_cards = self.rng.choice(a=self.num_cards, size=self.num_players * self.num_round + 1,
                                         replace=False)
        trump_card = shuffled_cards[-1]
        self.ground_truth[trump_card, self.hps['env']['GT_DECK']] = 0  # Remove the trump card from the deck
        self.ground_truth[trump_card, self.hps['env']['GT_TRUMP']] = 1  # Add the trump card the the cards played by nature
        self.trump_suit_index = self.suit_from_card[trump_card]
        if self.print_statements:
            print("Trump card {} [{}] with suit index {}".format(self.card_symbols[trump_card],
                                                                 trump_card,
                                                                 self.trump_suit_index
                                                                 ))
        self.trump_card = trump_card
        self.distribute_cards(shuffled_cards)

    def evaluate_round(self, bids, tricks):
        diff = tricks - bids
        results = np.zeros(shape=len(bids))
        # Overwrite score if bidding was correct
        results[diff == 0] = self.REWARD_FOR_CORRECT_PREDICTION + self.REWARD_PER_CORRECT_TRICK * bids[diff == 0]
        return results

    def normalize_reward(self, reward, num_round):
        low = 0
        high = self.REWARD_FOR_CORRECT_PREDICTION + num_round * self.REWARD_PER_CORRECT_TRICK
        # return (reward - low) / (high - low)  # [0,1]
        return -1 + 2 * (reward - low) / (high - low)   # [-1,1]

    def denormalize_reward(self, reward, num_round):
        low = 0
        high = self.REWARD_FOR_CORRECT_PREDICTION + num_round * self.REWARD_PER_CORRECT_TRICK
        return low + (reward + 1) * (high - low) / 2


class Wizard(TrickTakingGame):
    """
    Implementation for the Wizard environment
    """

    def __init__(self, hps):

        super(Wizard, self).__init__(hps)

        self.REWARD_PER_CORRECT_TRICK = 10
        self.REWARD_PER_WRONG_TRICK = -10
        self.REWARD_FOR_CORRECT_PREDICTION = 20

        self.num_jesters = self.hps['env']['JESTERS']
        self.num_wizards = self.hps['env']['WIZARDS']
        self.trump_card = None

        self.index_jester = self.num_suit
        self.index_wizard = self.num_suit + 1
        self.index_jester_in_cards = self.num_suit * self.num_cards_per_suit
        self.index_wizard_in_cards = self.index_jester_in_cards + self.num_jesters

        """
        Overwrite Wizard-specific variables:
        """
        # self.SUIT_NAMES = ['Red', 'Yellow', 'Green', 'Blue']
        # self.SUIT_SYMBOLS = ['R', 'Y', 'G', 'B', '❄']
        self.SUIT_SYMBOLS = ['♠', '♥', '♣', '♦', '❄']  # Use same symbols as for Spades (easier to distinguish)
        self.RANK_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']

        self.num_cards = self.num_suit * self.num_cards_per_suit + self.num_jesters + self.num_wizards
        self.card_names, self.card_symbols = self.create_cards()
        self.card_order = self.calculate_card_order()
        self.suit_from_card = self.calculate_suit_from_card()
        self.cards_to_be_followed_from_suit = self.calculate_cards_to_be_followed_from_suit()

        self.full_deck = 60

        # Random rewards for round 1 to 15 for the 4-player game with a full deck
        if self.hps['env']['CARDS'] == self.full_deck and self.hps['env']['PLAYERS'] == 4:
            self.rule_based_acc[1:6] = [.84, .71, .59, .51, .48]
            self.rule_based_acc[6:11] = [.46, .44, .43, .42, .41]
            self.rule_based_acc[11:16] = [.41, .41, .41, .41, .40]

            if self.hps['env']['NORMALIZE_REWARD']:
                self.random_reward[1:6] = [-.18, -.35, -.39, -.40, -.40]
                self.random_reward[6:11] = [-.39, -.38, -.38, -.38, -.37]
                self.random_reward[11:16] = [-.37, -.37, -.36, -.36, -.36]

                self.rule_based_reward[1:6] = [.34, .11, .01, -.02, -.03]
                self.rule_based_reward[6:11] = [-.03, -.02, -.01, -.01, .0]
                self.rule_based_reward[11:16] = [.01, .01, .02, .02, .02]

            else:
                self.random_reward[1:6] = [6.5, -0.5, -5.5, -10.0, -14.0]
                self.random_reward[6:11] = [-17.5, -21.0, -24.0, -27.5, -31.0]
                self.random_reward[11:16] = [-34.5, -37.5, -41.0, -44.5, -47.5]

                self.rule_based_reward[1:6] = [17.0, 13.5, 10.5, 8.5, 8.0]
                self.rule_based_reward[6:11] = [8.0, 8.5, 9.0, 9.5, 10.0]
                self.rule_based_reward[11:16] = [11.0, 12.0, 13.0, 14.0, 13.0]

        if self.hps['env']['CARDS'] == 5:
            if self.hps['env']['NORMALIZE_REWARD']:
                self.random_reward[2] = -.31
            else:
                self.random_reward[2] = .60

    def __repr__(self):
        return "Wizard"

    def create_cards(self):
        card_names, card_symbols = super().create_cards()
        card_names[self.index_jester_in_cards:self.index_wizard_in_cards] = "Jester"
        card_symbols[self.index_jester_in_cards:self.index_wizard_in_cards] = "❄"
        card_names[self.index_wizard_in_cards:] = "Wizard"
        card_symbols[self.index_wizard_in_cards:] = "☀"
        return card_names, card_symbols

    def calculate_card_order(self):
        # Get order from base class:
        order = super().calculate_card_order()

        # Set special case for wizard and jesters:
        for suit_index in range(self.num_suit + 1):
            compare = order[suit_index]
            # Overwrite Wizard row: Wizards BEAT all other cards (even other Wizards) when played first
            compare[self.index_wizard_in_cards:, :] = 0
            # Overwrite Wizard column: All cards (except Wizards) are BEATEN by Wizards
            compare[:self.index_wizard_in_cards, self.index_wizard_in_cards:] = 1
            # Overwrite Jester column: No card is BEATEN by Jesters - even Jesters themselves (the others win)
            compare[:self.index_wizard_in_cards, self.index_jester_in_cards:self.index_wizard_in_cards] = 0
            # Overwrite Jester row: Jesters don't BEAT ordinary cards
            compare[self.index_jester_in_cards:self.index_wizard_in_cards, :self.index_jester_in_cards] = 1
            order[suit_index] = compare

        return order

    def calculate_suit_from_card(self):
        suit_from_card = np.empty(shape=self.num_cards, dtype=int)
        for suit_index in range(self.num_suit):
            suit_from_card[suit_index * self.num_cards_per_suit:(suit_index + 1) * self.num_cards_per_suit] = suit_index

        index_jester = self.num_suit * self.num_cards_per_suit
        index_wizard = index_jester + self.num_jesters

        suit_from_card[index_jester:index_wizard] = self.index_jester
        suit_from_card[index_wizard:] = self.index_wizard
        return suit_from_card

    def calculate_cards_to_be_followed_from_suit(self):
        cards_to_be_followed_from_suit = {}

        for suit_index in range(self.num_suit):
            mask = np.zeros(shape=self.num_cards, dtype=int)
            mask[suit_index * self.num_cards_per_suit:(suit_index + 1) * self.num_cards_per_suit] = 1
            cards_to_be_followed_from_suit[suit_index] = mask

        # For Jesters and Wizard, nothing needs to be followed
        cards_to_be_followed_from_suit[self.index_jester] = np.zeros(shape=self.num_cards, dtype=int)
        cards_to_be_followed_from_suit[self.index_wizard] = np.zeros(shape=self.num_cards, dtype=int)

        return cards_to_be_followed_from_suit

    def initialize_game_round(self):
        # If there are enough cards, also draw a trump card
        if self.num_players * self.num_round < self.num_cards:
            shuffled_cards = self.rng.choice(a=self.num_cards, size=self.num_players * self.num_round + 1,
                                             replace=False)
            trump_card = shuffled_cards[-1]
            self.ground_truth[trump_card, self.hps['env']['GT_DECK']] = 0
            self.ground_truth[trump_card, self.hps['env']['GT_TRUMP']] = 1
            self.trump_suit_index = self.suit_from_card[trump_card]
            # The special case Wizard is handled as the special case Jester: no suit is trump
            if self.trump_suit_index == self.index_wizard:
                self.trump_suit_index = self.index_jester
            if self.print_statements:
                print("Trump card {} [{}] with suit index {}".format(self.card_symbols[trump_card],
                                                                     trump_card,
                                                                     self.trump_suit_index
                                                                     ))
        # If there are no cards left
        else:
            shuffled_cards = self.rng.choice(a=self.num_cards, size=self.num_players * self.num_round, replace=False)
            if self.print_statements:
                print("No trump card!")
            trump_card = None
            self.trump_suit_index = self.index_jester

        self.trump_card = trump_card
        self.distribute_cards(shuffled_cards)

    def get_action_mask(self, hand, follow_suit):
        action_mask = super().get_action_mask(hand=hand, follow_suit=follow_suit)
        """
        In Wizard, one can always choose to play Jesters or Wizards:
        """
        hand_one_hot = np.zeros(shape=self.num_cards, dtype=int)
        hand_one_hot[hand] = 1
        special_cards_one_hot = np.zeros_like(action_mask)
        special_cards_one_hot[self.index_jester_in_cards:] = hand_one_hot[self.index_jester_in_cards:]

        action_mask = np.logical_or(action_mask, special_cards_one_hot)
        return action_mask

    def evaluate_round(self, bids, tricks):
        diff = np.abs(tricks - bids)
        results = np.zeros(shape=len(bids))
        # Set score if bidding was correct:
        results[diff == 0] = self.REWARD_PER_CORRECT_TRICK * bids[diff == 0] + self.REWARD_FOR_CORRECT_PREDICTION
        # Set score if bidding was incorrect
        results[diff != 0] = self.REWARD_PER_WRONG_TRICK * diff[diff != 0]
        return results

    def normalize_reward(self, reward, num_round):
        low = num_round * self.REWARD_PER_WRONG_TRICK
        high = num_round * self.REWARD_PER_CORRECT_TRICK + self.REWARD_FOR_CORRECT_PREDICTION
        # return (reward - low) / (high - low)  # [0,1]
        return -1 + 2 * (reward - low) / (high - low)   # [-1,1]

    def denormalize_reward(self, reward, num_round):
        low = num_round * self.REWARD_PER_WRONG_TRICK
        high = num_round * self.REWARD_PER_CORRECT_TRICK + self.REWARD_FOR_CORRECT_PREDICTION
        return low + (reward + 1) * (high - low) / 2
