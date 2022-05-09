import numpy as np
import yaml
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

from agent import AgentMode, RuleBasedPlayer
from environment import Wizard
from agent_dqn import DQNPlayer
from training import manipulate_hps


def play_whole_game(hps, start_player_index):
    overall_reward = np.zeros(NUM_PLAYERS, dtype=float)
    for wizard_round in range(1, 15):
        hps_modified = hps
        hps_modified["env"]["TRICKS"] = wizard_round
        hps_modified["env"]["SEED"] = np.random.randint(0,1000000,1)
        hps_modified = manipulate_hps(h=hps_modified)

        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps_modified, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(RuleBasedPlayer(i, PLAYER_NAMES[i], hps_modified, None))
            players[i].set_previous_player(players[i - 1])

        for i in range(0, NUM_PLAYERS - 1):
            players[i].set_next_player(players[i + 1])
        players[0].set_previous_player(players[NUM_PLAYERS - 1])
        players[NUM_PLAYERS - 1].set_next_player(players[0])

        for p in players:
            p.set_agent_mode(AgentMode.EVAL)

        env = Wizard(hps_modified)
        env.set_players(players)

        _, _, rewards = env.game_round(num_round=wizard_round,
                                       start_player_round=players[start_player_index])
        overall_reward += rewards

        start_player_index = (start_player_index + 1) % NUM_PLAYERS

    # print(f"Rewards: {overall_reward}")
    winner_player_index = np.argmax(overall_reward)

    return winner_player_index


if __name__ == '__main__':

    start_time = time.time()

    with open('hps_win.yaml') as file:
        hps = yaml.load(file, Loader=yaml.FullLoader)

    PLAYER_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']
    PLOT_COLORS = ['black', 'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']

    NUM_PLAYERS = hps['env']['PLAYERS']
    MASTER = hps['agent']['MASTER_INDEX']

    rng = np.random.default_rng(hps['env']['SEED'])

    num_games_to_play = 1000

    winning_count = np.zeros(NUM_PLAYERS, dtype=float)
    winning_probs = np.zeros((num_games_to_play, NUM_PLAYERS), dtype=float)

    for game in tqdm(range(num_games_to_play)):
        index_winner = play_whole_game(hps=hps,
                                       start_player_index=game % NUM_PLAYERS)
        winning_count[index_winner] += 1
        winning_probs[game, :] = winning_count / (game+1)
    print(f"Winning probs: {np.round(100 * winning_count / num_games_to_play, decimals=1)}")
    plt.plot(winning_probs)
    plt.show()
    duration = time.time() - start_time
    print("Overall time: {} minutes".format(np.round(duration / 60.0, decimals=2)))

