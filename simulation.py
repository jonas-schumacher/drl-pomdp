import numpy as np


class Simulation:
    def __init__(self, env, current_trick, position_in_trick, position_in_bidding, history, hps):
        self.rng = np.random.default_rng(hps['env']['SEED'])
        self.env = env
        self.current_trick = current_trick
        self.current_pos_trick = position_in_trick
        self.current_pos_bidding = position_in_bidding
        self.current_pos_history = self.env.num_players * current_trick + position_in_trick
        self.history_static = history
        self.history = np.zeros_like(history)
        self.ground_truth = np.zeros_like(self.env.ground_truth)
        self.tricks = np.zeros_like(self.env.tricks)

        self.current_trick_winner_bid_pos = None
        self.current_trick_winner_card = None

    def run_simulation(self, sampled_state, card_played):
        """
        Run the rest of the game with fictitious players
        """

        # Step 0: Recover the current state of the environment
        self.history[:, :] = self.history_static[:, :]
        self.ground_truth[:, :] = sampled_state[:, :]
        self.tricks[:] = self.env.tricks[:]

        self.current_trick_winner_bid_pos = self.env.current_trick_winner_bid_pos
        self.current_trick_winner_card = self.env.current_trick_winner_card
        follow_suit = self.env.follow_suit
        lead_suit_played = self.env.lead_suit_played

        current_trick = self.current_trick

        # Step 1: Play the fixed card of the optimized player
        self.current_trick_winner_bid_pos, self.current_trick_winner_card, follow_suit, \
        lead_suit_played, self.history, self.ground_truth = self.env.process_card_played(
            card_played=card_played,
            current_trick_winner_bid_pos=self.current_trick_winner_bid_pos,
            current_trick_winner_card=self.current_trick_winner_card,
            follow_suit=follow_suit,
            lead_suit_played=lead_suit_played,
            current_pos_trick=self.current_pos_trick,
            current_pos_bidding=self.current_pos_bidding,
            current_pos_history=self.current_pos_history,
            history=self.history,
            ground_truth=self.ground_truth)

        # Step 2: Finish the current trick
        start_bid_pos = self.history[self.current_pos_history - self.current_pos_trick, 1]
        start_player_trick = self.env.players[self.env.pos2id_bidding[start_bid_pos]]
        start_player_trick, self.history, self.ground_truth = self.game_trick(
            start_player_trick=start_player_trick,
            current_trick=current_trick,
            history=self.history,
            current_pos_trick=self.current_pos_trick + 1,
            lead_suit_played=lead_suit_played,
            follow_suit=follow_suit,
            ground_truth=self.ground_truth)
        current_trick += 1
        self.tricks[start_player_trick.get_id] += 1

        # Step 3: Finish the game by playing all remaining tricks
        while current_trick < self.env.num_round:
            self.current_trick_winner_card = None
            start_player_trick, self.history, self.ground_truth = self.game_trick(
                start_player_trick=start_player_trick,
                current_trick=current_trick,
                history=self.history,
                current_pos_trick=0,
                lead_suit_played=False,
                follow_suit=self.env.num_suit,
                ground_truth=self.ground_truth)
            current_trick += 1
            self.tricks[start_player_trick.get_id] += 1

        # Calculate rewards (in player ID order)
        reward = self.env.evaluate_round(self.env.bids, self.tricks)
        reward_normalized = self.env.normalize_reward(reward=reward, num_round=self.env.num_round)
        return reward_normalized[self.env.pos2id_bidding[self.current_pos_bidding]]

    def game_trick(self, start_player_trick, current_trick, history, current_pos_trick, lead_suit_played, follow_suit,
                   ground_truth):
        current_player = self.env.players[(start_player_trick.get_id + current_pos_trick) % self.env.num_players]
        while current_pos_trick < self.env.num_players:
            current_pos_bidding = current_player.get_pos_bidding
            current_hand = np.flatnonzero(ground_truth[:, self.env.hps['env']['GT_HAND'] + current_pos_bidding])
            action_mask = self.env.get_action_mask(hand=current_hand, follow_suit=follow_suit)
            playable_cards = np.flatnonzero(action_mask)
            assert len(playable_cards) >= 1
            # Option A: chose action randomly
            if self.env.hps['model']['RANDOM_POLICY']:
                card_played = self.rng.choice(a=playable_cards)
            # Option B: chose action with memoryless DQN from master player
            else:
                observation = self.env.players[self.env.hps['agent']['MASTER_INDEX']].create_input_for_playing(
                    hand=current_hand,
                    current_trick=current_trick,
                    pos_in_trick=current_pos_trick,
                    pos_in_bidding=current_pos_bidding,
                    trump_suit_index=self.env.trump_suit_index,
                    follow_suit=follow_suit,
                    bids=self.env.bids[self.env.pos2id_bidding],
                    tricks=self.tricks[self.env.pos2id_bidding],
                    best=self.current_trick_winner_card)
                card_played = self.env.players[self.env.hps['agent']['MASTER_INDEX']].playing_rollout.sample_masked_action(observation, action_mask)
            assert card_played >= 0
            assert card_played in playable_cards

            self.current_trick_winner_bid_pos, self.current_trick_winner_card, follow_suit, \
            lead_suit_played, history, ground_truth = self.env.process_card_played(
                card_played=card_played,
                current_trick_winner_bid_pos=self.current_trick_winner_bid_pos,
                current_trick_winner_card=self.current_trick_winner_card,
                follow_suit=follow_suit,
                lead_suit_played=lead_suit_played,
                current_pos_trick=current_pos_trick,
                current_pos_bidding=current_pos_bidding,
                current_pos_history=self.env.num_players * current_trick + current_pos_trick,
                history=history,
                ground_truth=ground_truth)

            current_pos_trick += 1
            current_player = current_player.get_next_player

        trick_winner = self.env.players[self.env.pos2id_bidding[self.current_trick_winner_bid_pos]]

        return trick_winner, history, ground_truth
