import logging
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from agent import AgentMode, RandomPlayer, RuleBasedPlayer, HumanPlayer
from agent_dqn import DQNPlayer, ModelPlayer
from agent_history import HistoryPlayer
from environment import Wizard, Spades, OhHell


def manipulate_hps(h):
    """
    Calculate and add variables which depend on other variables
    :param h: hyperparameters
    :return: modified hyperparameters
    """
    S = h['env']['SUIT']
    C = h['env']['CARDS_PER_SUIT']
    P = h['env']['PLAYERS']
    T = h['env']['TRICKS']

    if h['env']['GAME_TO_PLAY'] == "WIZARD":
        CARDS = h['env']['CARDS'] = S * C + h['env']['JESTERS'] + h['env']['WIZARDS']
    else:
        CARDS = h['env']['CARDS'] = S * C
    CHECKPOINT_NAME = h['env']['CHECKPOINT'] = \
        str(h['env']['GAME_TO_PLAY']) + "-" + str(S) + "-" + str(C) + "-" + str(P) + "-" + str(T)
    h['hist']['CHECKPOINT'] = "checkpoints/hist-" + CHECKPOINT_NAME
    h['dqn']['CHECKPOINT'] = "checkpoints/dqn-" + CHECKPOINT_NAME
    h['dqn']['CHECKPOINT_HIST'] = "checkpoints/dqn-hist-" + CHECKPOINT_NAME
    h['model']['CHECKPOINT'] = "checkpoints/model-" + CHECKPOINT_NAME
    h['model']['CHECKPOINT_HIST'] = "checkpoints/model-hist-" + CHECKPOINT_NAME

    h['env']['GT_TRUMP'] = h['env']['GT_HAND'] + P
    h['env']['GT_PLAYED'] = h['env']['GT_TRUMP'] + 1
    h['env']['GT_SIZE'] = 2 * P + 2

    # Player type specific hyperparameters:
    h['agent']['HISTORY_PREPROCESSING'] = True if h['agent']['PLAYER_TYPE'] in ['DQN_HIST', 'MODEL_HIST'] else False
    h['agent']['PERFORM_TOURNAMENT'] = False if h['agent']['PLAYER_TYPE'] in ['HISTORY', 'HUMAN'] else True
    if h['agent']['PLAYER_TYPE'] in ['RANDOM', 'RULE']:
        h['agent']['TRAINING_MODE'] = False
    if h['agent']['PLAYER_TYPE'] in ['HUMAN']:
        h['agent']['TRAINING_MODE'] = False
        h['agent']['READ_CHECKPOINTS'] = True

    # Start training after 25% of replay buffer filling (instead of 10%), if checkpoints are read
    if h['agent']['READ_CHECKPOINTS']:
        h['agent']['REPLAY_START'] = 0.25

    GAMES = h['agent']['GAMES'] = h['agent']['BATCHES'] * h['agent']['ITERATIONS_PER_BATCH']

    h['hist']['INPUT'] = CARDS + P + (S + 1)
    h['hist']['OUTPUT'] = CARDS * P + S * P + P
    h['hist']['REPLAY_SIZE'] = min(h['hist']['MAX_REPLAY_SIZE'],
                                   int(h['agent']['REPLAY_FULL'] * GAMES))  # there is one sample per game
    h['hist']['REPLAY_START_SIZE'] = max(2 * h['hist']['BATCH_SIZE'],
                                         int(h['agent']['REPLAY_START'] * h['hist']['REPLAY_SIZE']))
    h['dqn']['INPUT_BIDDING'] = CARDS + P + S + (P - 1) * (T + 1)
    h['dqn']['OUTPUT_BIDDING'] = T + 1
    h['dqn']['REPLAY_SIZE_BIDDING'] = min(h['dqn']['MAX_REPLAY_SIZE_BIDDING'],
                                          int(h['agent']['REPLAY_FULL'] * GAMES * P))
    h['dqn']['REPLAY_START_SIZE_BIDDING'] = max(2 * h['dqn']['BATCH_SIZE_BIDDING'],
                                                int(h['agent']['REPLAY_START'] * h['dqn'][
                                                    'REPLAY_SIZE_BIDDING']))
    h['dqn']['INPUT_PLAYING'] = 2 * CARDS + 2 * P + 2 * S + 2 * P * (T + 1)
    h['dqn']['OUTPUT_PLAYING'] = CARDS
    h['dqn']['REPLAY_SIZE_PLAYING'] = min(h['dqn']['MAX_REPLAY_SIZE_PLAYING'],
                                          int(h['agent']['REPLAY_FULL'] * GAMES * P * T))
    h['dqn']['REPLAY_START_SIZE_PLAYING'] = max(2 * h['dqn']['BATCH_SIZE_PLAYING'],
                                                int(h['agent']['REPLAY_START'] * h['dqn'][
                                                    'REPLAY_SIZE_PLAYING']))

    h['model']['INPUT'] = h['dqn']['INPUT_PLAYING'] + h['hist']['HIDDEN_SIZE']
    h['model']['OUTPUT'] = CARDS * (2 * P + 2)
    h['model']['REPLAY_SIZE'] = min(h['model']['MAX_REPLAY_SIZE'],
                                    int(h['agent']['REPLAY_FULL'] * GAMES * P * T))  # same as for playing
    h['model']['REPLAY_START_SIZE'] = max(2 * h['model']['BATCH_SIZE'],
                                          int(h['agent']['REPLAY_START'] * h['model']['REPLAY_SIZE']))
    if h['agent']['TRAINING_MODE']:
        h['model']['SEARCH'] = False

    return h


def play(num_batches, num_games_per_batch, fix_start_player, train):
    """
    Play num_batches*num_games_per_batch rounds of the given game
    :param num_batches: these are only fictional groups for aggregation purposes
    :param num_games_per_batch: the results of these games will be aggregated
    :param fix_start_player: if None, start player is chosen randomly
    :param train: True = training mode, False = evaluation mode
    :return: array of scores per batch
    """
    score_outer = np.zeros((num_batches, NUM_PLAYERS, 4), dtype=float)
    score_inner = np.zeros((num_games_per_batch, NUM_PLAYERS, 4), dtype=float)
    iter_games = 0
    for batch_index in range(num_batches):
        start_time_batch = time.time()
        for iter_index in range(num_games_per_batch):
            if fix_start_player is None:
                start_player_round = players[rng.integers(low=0, high=NUM_PLAYERS)]
            else:
                start_player_round = fix_start_player
            bids, tricks, rewards = env.game_round(num_round=NUM_TRICKS,
                                                   start_player_round=start_player_round)

            score_inner[iter_index, :, 0] = rewards
            score_inner[iter_index, :, 1] = bids == tricks
            score_inner[iter_index, :, 2] = bids
            score_inner[iter_index, :, 3] = tricks

            iter_games += 1

        score_outer[batch_index, :, :] = np.mean(score_inner, axis=0)

        logger.info(
            "Batch: {} \N{tab} Rewards: {} \N{tab} = {} \N{tab} Accuracy: {} \N{tab} = {}% \N{tab} Bids-Tricks {}".format(
                batch_index,
                np.round(score_outer[batch_index, :, 0], 2),
                np.round(np.mean(score_outer[batch_index, :, 0]), 2),
                np.round(score_outer[batch_index, :, 1], 2),
                np.round(100 * np.mean(score_outer[batch_index, :, 1]), 0),
                np.round(np.mean(score_outer[batch_index, :, 2] - score_outer[batch_index, :, 3]), 2)))

        if train:
            writer.add_scalar("duration", time.time() - start_time_batch, iter_games)
            for p_index in range(NUM_PLAYERS):
                writer.add_scalar("results_score/" + str(players[p_index]), score_outer[batch_index, p_index, 0],
                                  iter_games)
                writer.add_scalar("results_accuracy/" + str(players[p_index]), score_outer[batch_index, p_index, 1],
                                  iter_games)

    return score_outer


def create_players():
    """
    Instantiate players based on parameters from hps_train.yaml file
    :return:
    """
    if PLAYER_TYPE == "RANDOM":  # all players play randomly
        algo = "random"
        players = [RandomPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(RandomPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "RULE":  # all players use rule-based playing
        algo = "rule"
        players = [RuleBasedPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(RuleBasedPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "HISTORY":  # all players learn the history and play randomly
        algo = "hist"
        players = [HistoryPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(HistoryPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN":  # all players are regular DQN agents
        algo = "dqn"
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN_HIST":  # all players DQN agents with additional historic input
        algo = "dqn-hist"
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "MODEL":  # all players learn a model of their environment
        algo = "model"
        players = [ModelPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(ModelPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "MODEL_HIST":  # players learn a model of their environment
        algo = "model-hist"
        players = [ModelPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(ModelPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "HUMAN":  # N-1 regular DQN agents play against 1 human player
        algo = "human"
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS - 1):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        players.append(HumanPlayer(NUM_PLAYERS - 1, PLAYER_NAMES[NUM_PLAYERS - 1], hps, None))
        players[NUM_PLAYERS - 1].set_previous_player(players[NUM_PLAYERS - 2])
    elif PLAYER_TYPE == "DQN3_RANDOM1":  # 3 DQN agents play agents 1 random agent
        algo = "custom"
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS - 1):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        for i in range(NUM_PLAYERS - 1, NUM_PLAYERS):
            players.append(RandomPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN2_RANDOM2":
        algo = "custom"
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, 2):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        for i in range(2, NUM_PLAYERS):
            players.append(RandomPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN1_RANDOM3":
        algo = "custom"
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS):
            players.append(RandomPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN3_RULE1":
        algo = "custom"
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, NUM_PLAYERS - 1):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        for i in range(NUM_PLAYERS - 1, NUM_PLAYERS):
            players.append(RuleBasedPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN2_RULE2":
        algo = "custom"
        players = [DQNPlayer(MASTER, PLAYER_NAMES[MASTER], hps, None)]
        for i in range(1, 2):
            players.append(DQNPlayer(i, PLAYER_NAMES[i], hps, players[MASTER]))
            players[i].set_previous_player(players[i - 1])
        for i in range(2, NUM_PLAYERS):
            players.append(RuleBasedPlayer(i, PLAYER_NAMES[i], hps, None))
            players[i].set_previous_player(players[i - 1])
    elif PLAYER_TYPE == "DQN1_RULE3":
        algo = "custom"
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

    return players, algo


if __name__ == '__main__':

    start_time = time.time()

    # Load variables from file and calculate additional dependent variables
    with open('hps_train.yaml') as file:
        hps = yaml.load(file, Loader=yaml.FullLoader)
    hps = manipulate_hps(h=hps)

    PLAYER_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']
    PLOT_COLORS = ['black', 'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    SMOOTHING_WINDOW_LENGTH = 10

    # Extract relevant hps:
    NUM_TRICKS = hps['env']['TRICKS']
    NUM_PLAYERS = hps['env']['PLAYERS']
    NUM_CARDS = hps['env']['CARDS']
    GAME_TO_PLAY = hps['env']['GAME_TO_PLAY']
    PLAYER_TYPE = hps['agent']['PLAYER_TYPE']
    MASTER = hps['agent']['MASTER_INDEX']

    rng = np.random.default_rng(hps['env']['SEED'])
    if GAME_TO_PLAY == "SPADES":
        env = Spades(hps)
    elif GAME_TO_PLAY == "HELL":
        env = OhHell(hps)
    else:
        env = Wizard(hps)
    assert env.num_cards == NUM_CARDS

    players, algo = create_players()

    env.set_players(players)
    experiment_name = algo + "-" + hps['env']['CHECKPOINT']
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

    if not os.path.exists("runs"):
        os.makedirs("runs")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("results"):
        os.makedirs("results")

    logger = logging.getLogger("trick")
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)
    logfile = logging.FileHandler("logs/" + experiment_name + "-" + current_time + ".log", mode="w")
    logfile.setFormatter(logging.Formatter("%(asctime)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(logfile)

    logger.info("Hyperparameters: ")
    for entry in hps:
        logger.info("{}: {}".format(entry, hps[entry]))

    labels = ["Reward", "Accuracy"]
    column_labels_df = [f"{l} {str(p)}" for l in reversed(labels) for p in np.arange(1, NUM_PLAYERS + 1)]

    if hps['agent']['TRAINING_MODE']:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join('runs', current_time + "-" + experiment_name))
        players[MASTER].give_writing_access(writer)
        score_batch = play(num_batches=hps['agent']['BATCHES'],
                           num_games_per_batch=hps['agent']['ITERATIONS_PER_BATCH'],
                           fix_start_player=None,
                           train=True)
        duration = time.time() - start_time
        logger.info("Training time: {} minutes".format(np.round(duration / 60.0, decimals=2)))

        # Optionally write checkpoints
        players[MASTER].save_checkpoint(current_time=current_time) if hps['agent']['WRITE_CHECKPOINTS'] else None

        # Analyze training results
        random_reference = [env.random_reward[NUM_TRICKS], 1 / (NUM_TRICKS + 1)]
        rule_reference = [env.rule_based_reward[NUM_TRICKS], env.rule_based_acc[NUM_TRICKS]]
        upper_reference = [env.upper_bound_reward(NUM_TRICKS)[0], 1.0]
        for data_index in range(2):
            fig, ax = plt.subplots()  # create a new figure
            for i in range(NUM_PLAYERS):
                # Smoothing is only necessary if few games are played
                if hps['agent']['ITERATIONS_PER_BATCH'] >= 10:
                    ax.plot(score_batch[:, i, data_index],
                            label="Player " + PLAYER_NAMES[i],
                            color=PLOT_COLORS[i],
                            alpha=1.0)
                else:
                    ax.plot(score_batch[:, i, data_index], color=PLOT_COLORS[i], alpha=0.3)
                    score_smoothed = np.convolve(score_batch[:, i, data_index],
                                                 np.ones(SMOOTHING_WINDOW_LENGTH), 'valid') / SMOOTHING_WINDOW_LENGTH
                    score_smoothed = np.concatenate(
                        (score_batch[:SMOOTHING_WINDOW_LENGTH - 1, i, data_index], score_smoothed))
                    ax.plot(score_smoothed,
                            label="Player " + PLAYER_NAMES[i],
                            color=PLOT_COLORS[i])
            if env.full_deck == NUM_CARDS and NUM_PLAYERS == 4:
                ax.plot(np.full_like(a=score_batch[:, 0, 0], fill_value=random_reference[data_index]),
                        label="Random",
                        color="Grey",
                        linestyle='dashed')
                ax.plot(np.full_like(a=score_batch[:, 0, 0], fill_value=rule_reference[data_index]),
                        label="Rule-based",
                        color="Grey",
                        linestyle='dashdot')
            elif NUM_CARDS <= 5:
                ax.plot(np.full_like(a=score_batch[:, 0, 0], fill_value=random_reference[data_index]),
                        label="Random",
                        color="Grey",
                        linestyle='dashed')

            ax.plot(np.full_like(a=score_batch[:, 0, 0], fill_value=upper_reference[data_index]),
                    label="Upper bound",
                    color="Grey",
                    linestyle='dotted')

            plt.xlabel("Iterations of {} (x{} game rounds)".format(
                str(GAME_TO_PLAY),
                hps['agent']['ITERATIONS_PER_BATCH']))
            plt.ylabel(labels[data_index])
            plt.legend()
            plt.show()
            fig.savefig("results/" + experiment_name + "-" + current_time + "-" + labels[data_index])

        train_df = pd.DataFrame(index=np.arange(hps['agent']['BATCHES']),
                                columns=column_labels_df,
                                dtype=float)
        train_df.iloc[:, :NUM_PLAYERS] = score_batch[:, :, 1]
        train_df.iloc[:, NUM_PLAYERS:] = score_batch[:, :, 0]
        train_df.to_csv(f"results/training-en-{experiment_name}-{current_time}.csv")
        train_df.to_csv(f"results/training-de-{experiment_name}-{current_time}.csv", sep=";", decimal=",")

    # Run 3 game per player in play mode with print_statements
    env.set_print_statements(True)
    for p in players:
        p.set_agent_mode(AgentMode.PLAY)
    for _ in range(hps['agent']['EVALUATION_GAMES']):
        for p in players:
            env.game_round(num_round=NUM_TRICKS, start_player_round=p)
    env.set_print_statements(False)

    # Perform a tournament between the trained players using fixed starting positions
    if hps['agent']['PERFORM_TOURNAMENT']:
        for p in players:
            p.set_agent_mode(AgentMode.EVAL)

        eval_df = pd.DataFrame(index=np.arange(NUM_PLAYERS),
                               columns=column_labels_df,
                               dtype=float)
        for index in range(0, -NUM_PLAYERS, -1):
            logger.info('----------------------------------------------------------------------')
            player_index = (index + NUM_PLAYERS) % NUM_PLAYERS
            logger.info("Position: {}".format((NUM_PLAYERS - index) % NUM_PLAYERS))
            score_batch = play(num_batches=hps['agent']['BATCHES_TOURNAMENT'],
                               num_games_per_batch=hps['agent']['ITERATIONS_PER_BATCH_TOURNAMENT'],
                               fix_start_player=players[player_index],
                               train=False)
            eval_df.iloc[-index, :NUM_PLAYERS] = score_batch[0, :, 1]
            eval_df.iloc[-index, NUM_PLAYERS:] = score_batch[0, :, 0]
        eval_df.to_csv(f"results/evaluation-en-{experiment_name}-{current_time}.csv")
        eval_df.to_csv(f"results/evaluation-de-{experiment_name}-{current_time}.csv", sep=";", decimal=",")

    for p in players:
        p.finish_interaction()

    duration = time.time() - start_time
    logger.info("Overall time: {} minutes".format(np.round(duration / 60.0, decimals=2)))
