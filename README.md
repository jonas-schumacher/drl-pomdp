# DRL for POMDPs
In this project I'm using **Deep Reinforcement Learning (DRL)** to optimize **Partially Observable Markov Decision Processes (POMDP)**. 
The environment constitutes a flexible and scalable trick-taking game, which allows card games like *Spades*, *Oh Hell* and *Wizard* to be analyzed and optimized.

## Learning agents
- The core algorithm is a **Deep Q-Network (DQN)**. 
- For historic preprocessing, a **Long Short-Term Memory (LSTM)** network has been implemented
- To optimize strategies at runtime, a simple **tree search** algorithm is provided

## Implemented opponents
- Self-play against DQN agents
- Random agents
- Rule-based agents

## Getting started
- Simply run the *training.py* script which will train a DQN agent in a simplified Wizard environment (2 players, 5 cards, perfect information).
- For more complicated games, you can change the parameters in the *hps_train.yaml* file. 
- Configs used in the IEEE CoG paper can be found in the *config-files* directory.
