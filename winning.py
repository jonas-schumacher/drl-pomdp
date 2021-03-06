import logging
import time

import numpy as np
import yaml
from matplotlib import pyplot as plt

from agent import AgentMode, RuleBasedPlayer, RandomPlayer
from agent_dqn import DQNPlayer
from environment import Wizard
from training import manipulate_hps


def create_players(hps):
    if PLAYER_TYPE == "RANDOM":
        players = [RandomPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(RandomPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "RULE":
        players = [RuleBasedPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(RuleBasedPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN":
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN3_RANDOM1":
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS - 1):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        for i in range(NUM_PLAYERS - 1, NUM_PLAYERS):
            players.append(RandomPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN2_RANDOM2":
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, 2):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        for i in range(2, NUM_PLAYERS):
            players.append(RandomPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN1_RANDOM3":
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(RandomPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN3_RULE1":
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS - 1):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        for i in range(NUM_PLAYERS - 1, NUM_PLAYERS):
            players.append(RuleBasedPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN2_RULE2":
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, 2):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        for i in range(2, NUM_PLAYERS):
            players.append(RuleBasedPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN1_RULE3":
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(RuleBasedPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    else:
        raise ValueError('This player type is unknown.')

    for i in range(0, NUM_PLAYERS - 1):
        players[i].set_next_player(players[i + 1])
    players[0].set_previous_player(players[NUM_PLAYERS - 1])
    players[NUM_PLAYERS - 1].set_next_player(players[0])

    return players


def play_whole_game(hps, start_player_index):
    overall_reward = np.zeros(NUM_PLAYERS, dtype=float)
    for wizard_round in range(1, 16):
        hps_modified = hps
        hps_modified["env"]["TRICKS"] = wizard_round
        # Each round should start with a new (and hopefully unique) seed
        # Only relevant for the environment (card shuffling) and RandomPlayers (bidding & playing)
        hps_modified["env"]["SEED"] = rng.integers(0, 1000000000)
        hps_modified = manipulate_hps(h=hps_modified)

        players = create_players(hps=hps_modified)

        for p in players:
            p.set_agent_mode(AgentMode.EVAL)

        env = Wizard(hps_modified)
        env.set_players(players)

        # logger.info(f"Player {players[start_player_index]} starts round {wizard_round}")

        _, _, rewards = env.game_round(num_round=wizard_round,
                                       start_player_round=players[start_player_index])
        overall_reward += rewards

        start_player_index = (start_player_index + 1) % NUM_PLAYERS

    # logger.info(f"Rewards: {overall_reward}")
    winner_indices = np.zeros(NUM_PLAYERS, dtype=int)
    winners = np.where(overall_reward == max(overall_reward))
    winner_indices[winners] = 1

    # if winner_indices.sum() > 1:
    #     print(overall_reward)
    #     print(winner_indices)

    return winner_indices


if __name__ == '__main__':

    start_time = time.time()

    with open('hps_win.yaml') as file:
        hps = yaml.load(file, Loader=yaml.FullLoader)

    PLAYER_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']
    PLOT_COLORS = ['black', 'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']

    NUM_PLAYERS = hps['env']['PLAYERS']
    MASTER = hps['agent']['MASTER_INDEX']
    PLAYER_TYPE = hps['agent']['PLAYER_TYPE']

    rng = np.random.default_rng(hps['env']['SEED'])

    num_games_to_play = hps['agent']['ITERATIONS_PER_BATCH_TOURNAMENT']

    winning_count = np.zeros(NUM_PLAYERS, dtype=float)
    winning_probs = np.zeros((num_games_to_play, NUM_PLAYERS), dtype=float)

    logger = logging.getLogger("trick")
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)
    logger.info("Start calculation:")

    win_count = 0

    for game in range(num_games_to_play):
        start_player_index = game % NUM_PLAYERS
        winner_indices = play_whole_game(hps=hps,
                                         start_player_index=start_player_index)
        win_count += winner_indices.sum()
        winning_count += winner_indices
        current_winning_probs = winning_count / win_count
        winning_probs[game, :] = current_winning_probs
        if (game + 1) % (num_games_to_play / 100) == 0:
            logger.info(f"{game + 1} / {win_count}: {np.round(100 * current_winning_probs, decimals=2)}")
            logger.info(f"{game + 1} / {win_count}: {winning_count}")
    logger.info(f"Final winning probs: {np.round(100 * winning_count / win_count, decimals=2)}")
    logger.info(f"Final winning count: {winning_count}")
    fig, ax = plt.subplots()
    for i in range(NUM_PLAYERS):
        ax.plot(winning_probs[:, i],
                label=f"Player {PLAYER_NAMES[i]}",
                color=PLOT_COLORS[i])
    plt.xlabel(f"Number of Wizard games played")
    plt.ylabel("Average winning probability")
    plt.legend()
    plt.show()
    fig.savefig(f"results-update-cog/{PLAYER_TYPE}.png")

    duration = time.time() - start_time
    logger.info("Overall time: {} minutes".format(np.round(duration / 60.0, decimals=2)))
