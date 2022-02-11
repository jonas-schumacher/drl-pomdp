import numpy as np
import torch
import torch.nn.functional as F

from agent import OptimizedPlayer, calc_grad, AgentMode, GamePhase
from policy_gradient.networks_pg import A2CActor, A2CCritic
from replay_buffer import BiddingExperience, PlayingExperience


class A2CPlayer(OptimizedPlayer):
    """
    This player uses Advantage Actor-Critic (A2C, policy gradient) for both bidding and playing
    The higher eps (=1e-3) results in a better training stability (see M. Lapan - Deep Reinforcement Learning Hands-On)
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps)

        torch.manual_seed(hps['env']['SEED'])
        self.master = master
        self.checkpoint_name = self.hps['a2c']['CHECKPOINT'] + ".pt"

        if master is None:
            self.trajectory_bidding = []
            self.trajectory_playing = []

            self.bidding_actor = A2CActor(num_inputs=self.input_size_bidding,
                                          num_actions=self.output_size_bidding,
                                          hidden=self.hps['dqn']['HIDDEN_BIDDING'],
                                          hps=hps)
            self.bidding_actor_opt = torch.optim.Adam(self.bidding_actor.parameters(),
                                                      lr=self.hps['a2c']['LR_ACTOR_BIDDING'],
                                                      eps=1e-3)
            self.bidding_critic = A2CCritic(num_inputs=self.input_size_bidding,
                                            hidden=self.hps['dqn']['HIDDEN_BIDDING'],
                                            hps=hps)
            self.bidding_critic_opt = torch.optim.Adam(self.bidding_critic.parameters(),
                                                       lr=self.hps['a2c']['LR_CRITIC_BIDDING'])
            self.playing_actor = A2CActor(num_inputs=self.input_size_playing,
                                          num_actions=self.output_size_playing,
                                          hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                          hps=hps)
            self.playing_actor_opt = torch.optim.Adam(self.playing_actor.parameters(),
                                                      lr=self.hps['a2c']['LR_ACTOR_PLAYING'],
                                                      eps=1e-3)
            self.playing_critic = A2CCritic(num_inputs=self.input_size_playing,
                                            hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                            hps=hps)
            self.playing_critic_opt = torch.optim.Adam(self.playing_critic.parameters(),
                                                       lr=self.hps['a2c']['LR_CRITIC_PLAYING'])

            self.load_checkpoint() if self.hps['agent']['READ_CHECKPOINTS'] else None

        else:
            self.bidding_actor = self.master.bidding_actor
            self.bidding_critic = self.master.bidding_critic
            self.playing_actor = self.master.playing_actor
            self.playing_critic = self.master.playing_critic
            self.trajectory_bidding = self.master.trajectory_bidding
            self.trajectory_playing = self.master.trajectory_playing

    def bidding(self, pos_in_bidding, trump_suit_index, bids):
        observation = self.create_input_for_bidding(
            hand=self.hand,
            pos_in_bidding=pos_in_bidding,
            trump_suit_index=trump_suit_index,
            bids=bids)
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

        action = self.playing_actor.sample_masked_action(observation, action_mask)

        if self.agent_mode == AgentMode.TRAIN:
            self.save_transition(obs=observation, action=action, mask=action_mask, reward=None)
        self.transition()

        return action

    def finish_round(self, reward, history):
        self.game_phase = GamePhase.FINISH
        if self.agent_mode == AgentMode.TRAIN:
            self.save_transition(obs=None, action=None, mask=None, reward=reward)
            self.train_networks()
        self.transition()

    def save_transition(self, obs, action, mask, reward):
        if self.game_phase == GamePhase.BIDDING:
            self.obs_bidding = obs
            self.action_bidding = action
        elif self.game_phase == GamePhase.FIRST_PLAYING:
            self.previous_obs = obs
            self.previous_action = action
            self.previous_mask = mask
        elif self.game_phase == GamePhase.PLAYING:
            """
            In A2C we are on-policy and therefore have a simple list which is cleared after each training
            """
            exp = PlayingExperience(obs=self.previous_obs,
                                    action=self.previous_action,
                                    reward=0.0,
                                    done=False,
                                    next_obs=obs,
                                    mask=self.previous_mask,
                                    next_mask=mask)
            self.trajectory_playing.append(exp)
            self.previous_obs = obs
            self.previous_action = action
            self.previous_mask = mask
        elif self.game_phase == GamePhase.FINISH:
            exp = BiddingExperience(self.obs_bidding, self.action_bidding, reward)
            self.trajectory_bidding.append(exp)
            exp = PlayingExperience(obs=self.previous_obs,
                                    action=self.previous_action,
                                    reward=reward,
                                    done=True,
                                    next_obs=np.ones_like(a=self.previous_obs),
                                    mask=self.previous_mask,
                                    next_mask=np.ones_like(a=self.previous_mask))
            self.trajectory_playing.append(exp)

    def train_networks(self):
        if self.master is None:
            if (self.game_count + 1) % self.hps['dqn']['TRAIN_INTERVAL'] == 0:
                if len(self.trajectory_bidding) >= self.hps['a2c']['TRAJECTORY_SIZE_BIDDING']:
                    self.train_bidding()
                    self.train_count_bidding += 1
                if len(self.trajectory_playing) >= self.hps['a2c']['TRAJECTORY_SIZE_PLAYING']:
                    self.train_playing()
                    self.train_count_playing += 1

    def train_a2c_generic(self, actor, actor_opt, critic, critic_opt,
                          state_batch, action_batch, target_value_batch, batch_size,
                          phase="bidding", action_masks_batch=None):
        """
        Generic method for training both bidding and playing phase
        """
        actor_opt.zero_grad()

        # Value loss
        critic_opt.zero_grad()
        value_batch = critic(state_batch)
        loss_value = torch.nn.MSELoss()(value_batch.squeeze(-1), target_value_batch)
        loss_value.backward()
        critic_opt.step()
        logits = actor(state_batch)

        # Policy loss
        advantage = target_value_batch - value_batch.detach().squeeze(-1)
        if action_masks_batch is not None:
            logits[~action_masks_batch] = -float("inf")
        prob = F.softmax(logits, dim=1)
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_action = log_prob[torch.arange(batch_size), action_batch]
        weighted_advantage = advantage * log_prob_action
        loss_policy = -weighted_advantage.mean()

        # Entropy calculation
        if action_masks_batch is not None:
            entropy_sum = torch.FloatTensor([0.0])
            for i in range(batch_size):
                entropy_sum += (prob[i][action_masks_batch[i]] * log_prob[i][action_masks_batch[i]]).sum()
            mean_entropy = entropy_sum / batch_size
        else:
            mean_entropy = (prob * log_prob).sum(dim=1).mean()

        if phase == "bidding":
            loss_entropy = self.hps['a2c']['ENTROPY_BIDDING'] * mean_entropy
        else:
            loss_entropy = self.hps['a2c']['ENTROPY_PLAYING'] * mean_entropy

        # In separate networks the entropy is added to the policy loss
        loss_total = loss_policy + loss_entropy
        loss_total.backward()
        actor_opt.step()

        self.writer.add_scalar("loss/critic_" + phase, loss_value.item(), self.game_count)
        self.writer.add_scalar("loss/actor_" + phase, loss_policy.item(), self.game_count)
        self.writer.add_scalar("grad/critic_" + phase, calc_grad(critic)[0], self.game_count)
        self.writer.add_scalar("grad/actor_" + phase, calc_grad(actor)[0], self.game_count)
        self.writer.add_scalar("entropy/actor_" + phase, mean_entropy.item(), self.game_count)
        self.writer.add_scalar("refvals/actor_" + phase, torch.mean(target_value_batch).item(), self.game_count)
        self.writer.add_scalar("refadv/actor_" + phase, torch.mean(advantage).item(), self.game_count)

    def train_bidding(self):

        batch_size = len(self.trajectory_bidding)
        obs_batch = []
        action_batch = []
        reward_batch = []
        for idx, exp in enumerate(self.trajectory_bidding):
            obs_batch.append(np.array(exp.obs, copy=False))
            action_batch.append(int(exp.action))
            reward_batch.append(exp.reward)
        obs_batch = torch.FloatTensor(np.array(obs_batch, copy=False))
        action_batch = torch.LongTensor(action_batch)
        reward_batch = np.array(reward_batch, dtype=np.float32)
        target_value_batch = torch.FloatTensor(reward_batch)

        self.trajectory_bidding.clear()

        self.train_a2c_generic(
            actor=self.bidding_actor,
            actor_opt=self.bidding_actor_opt,
            critic=self.bidding_critic,
            critic_opt=self.bidding_critic_opt,
            state_batch=obs_batch,
            action_batch=action_batch,
            target_value_batch=target_value_batch,
            batch_size=batch_size,
            phase="bidding",
            action_masks_batch=None)

    def train_playing(self):

        batch_size = len(self.trajectory_playing)

        obs_batch = []
        action_mask_batch = []
        action_batch = []
        reward_batch = []
        not_done_idx = []
        next_obs_batch = []
        for idx, exp in enumerate(self.trajectory_playing):
            current_obs = np.array(exp.obs, copy=False)
            obs_batch.append(current_obs)
            action_mask = np.array(exp.mask, copy=False)
            action_mask_batch.append(action_mask)
            action_batch.append(int(exp.action))
            reward_batch.append(exp.reward)
            # Only save next_states if trajectory is not done
            if not exp.done:
                not_done_idx.append(idx)
                next_obs_batch.append(np.array(exp.next_obs, copy=False))

        obs_batch = torch.FloatTensor(np.array(obs_batch, copy=False))
        action_mask_batch = torch.BoolTensor(action_mask_batch)
        action_batch = torch.LongTensor(action_batch)
        rewards_batch = np.array(reward_batch, dtype=np.float32)
        if not_done_idx:
            next_obs_batch = torch.FloatTensor(np.array(next_obs_batch, copy=False))
            next_vals_batch = self.playing_critic(next_obs_batch)
            next_vals_batch_numpy = next_vals_batch.data.numpy()[:, 0]
            next_vals_batch_numpy *= self.hps['agent']['GAMMA']
            rewards_batch[not_done_idx] += next_vals_batch_numpy
        target_value_batch = torch.FloatTensor(rewards_batch)

        self.trajectory_playing.clear()

        self.train_a2c_generic(
            actor=self.playing_actor,
            actor_opt=self.playing_actor_opt,
            critic=self.playing_critic,
            critic_opt=self.playing_critic_opt,
            state_batch=obs_batch,
            action_batch=action_batch,
            target_value_batch=target_value_batch,
            batch_size=batch_size,
            phase="playing",
            action_masks_batch=action_mask_batch)

    def load_checkpoint(self):
        if self.master is None:
            checkpoint = torch.load(self.checkpoint_name)
            self.bidding_actor.load_state_dict(checkpoint['actor_bidding'])
            self.bidding_actor_opt.load_state_dict(checkpoint['actor_bidding_opt'])
            self.bidding_critic.load_state_dict(checkpoint['critic_bidding'])
            self.bidding_critic_opt.load_state_dict(checkpoint['critic_bidding_opt'])
            self.playing_actor.load_state_dict(checkpoint['actor_playing'])
            self.playing_actor_opt.load_state_dict(checkpoint['actor_playing_opt'])
            self.playing_critic.load_state_dict(checkpoint['critic_playing'])
            self.playing_critic_opt.load_state_dict(checkpoint['critic_playing_opt'])

    def save_checkpoint(self, current_time):
        assert self.master is None
        checkpoint = {'actor_bidding': self.bidding_actor.state_dict(),
                      'actor_bidding_opt': self.bidding_actor_opt.state_dict(),
                      'actor_playing': self.playing_actor.state_dict(),
                      'actor_playing_opt': self.playing_actor_opt.state_dict(),
                      'critic_bidding': self.bidding_critic.state_dict(),
                      'critic_bidding_opt': self.bidding_critic_opt.state_dict(),
                      'critic_playing': self.playing_critic.state_dict(),
                      'critic_playing_opt': self.playing_critic_opt.state_dict()}
        torch.save(checkpoint, self.checkpoint_name + "-" + current_time + ".pt")


def calc_adv_ref(trajectory, critic, states_v, gamma, gae_lambda):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param critic: critic network
    :param states_v: states tensor
    :param gamma:
    :param gae_lambda:
    :return: tuple with advantage numpy array and reference values
    """
    values_v = critic(states_v)
    values = values_v.squeeze().data.numpy()
    # GAE = generalized advantage estimator = smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    # Iterate in reverse order through value, next value and reward
    for val, next_val, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
        # GAE = (Reward) - (StateValue) [If current step is terminal, there is no next value]
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        # GAE = (Reward + gamma*nextVal) - (StateValue)
        else:
            delta = exp.reward + gamma * next_val - val
            # discount all future GAEs and add current GAE
            last_gae = delta + gamma * gae_lambda * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    # adv = advantage = return of future trajectory - state value >> needed for actor
    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    # ref = return of future trajectory   >> needed for critic
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v, ref_v


class PPOPlayer(A2CPlayer):
    """
    This player uses PPO = Proximal Policy Optimization (policy gradient) for playing
    """

    def __init__(self, identification, player_name, hps, master):
        super().__init__(identification, player_name, hps, master)

        self.checkpoint_name = self.hps['ppo']['CHECKPOINT'] + ".pt"

        if master is None:
            self.playing_actor = A2CActor(num_inputs=self.input_size_playing,
                                          num_actions=self.output_size_playing,
                                          hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                          hps=hps)
            self.playing_critic = A2CCritic(num_inputs=self.input_size_playing,
                                            hidden=self.hps['dqn']['HIDDEN_PLAYING'],
                                            hps=hps)
            self.playing_actor_opt = torch.optim.Adam(self.playing_actor.parameters(), lr=self.hps['ppo']['LR_ACTOR'])
            self.playing_critic_opt = torch.optim.Adam(self.playing_critic.parameters(), lr=self.hps['ppo']['LR_CRITIC'])
            self.load_checkpoint() if self.hps['agent']['READ_CHECKPOINTS'] else None

        else:
            self.playing_actor = self.master.playing_actor
            self.playing_critic = self.master.playing_critic

    def train_playing(self):
        """
        Train playing with PPO
        """
        traj_obs = [t.obs for t in self.trajectory_playing]
        traj_action_mask = [t.mask for t in self.trajectory_playing]
        traj_action = [t.action for t in self.trajectory_playing]
        traj_obs = torch.FloatTensor(traj_obs)
        traj_action_mask = torch.BoolTensor(traj_action_mask)
        traj_action = torch.LongTensor(traj_action)
        # Calculate advantages and reference values: no gradients so far!
        traj_adv, traj_ref = calc_adv_ref(trajectory=self.trajectory_playing,
                                          critic=self.playing_critic,
                                          states_v=traj_obs,
                                          gamma=self.hps['agent']['GAMMA'],
                                          gae_lambda=self.hps['ppo']['GAE_LAMBDA'])

        logits = self.playing_actor(traj_obs)
        logits[~traj_action_mask] = -float("inf")
        old_logprob_dist = F.log_softmax(logits, dim=1)
        old_logprob = old_logprob_dist[torch.arange(len(self.trajectory_playing)), traj_action]

        # normalize advantages: (x-mu)/std
        traj_adv = traj_adv - torch.mean(traj_adv)
        traj_adv /= torch.std(traj_adv)

        # drop last entry from the trajectory, as adv and ref value have been calculated without it
        self.trajectory_playing = self.trajectory_playing[:-1]
        # call .detach, because we don't want gradients to be calculated for the old log_prob
        old_logprob = old_logprob[:-1].detach()

        sum_loss_value = 0.0
        sum_loss_policy = 0.0
        count_steps = 0

        """
        Actual training loop: iterate x times over data from trajectory
        """
        for epoch in range(self.hps['ppo']['EPOCHS']):
            for batch_ofs in range(0, len(self.trajectory_playing), self.hps['ppo']['BATCH_SIZE']):
                batch_l = min(batch_ofs + self.hps['ppo']['BATCH_SIZE'], len(self.trajectory_playing))
                batch_size = batch_l - batch_ofs
                obs_batch = traj_obs[batch_ofs:batch_l]
                action_mask_batch = traj_action_mask[batch_ofs:batch_l]
                action_batch = traj_action[batch_ofs:batch_l]
                adv_batch = traj_adv[batch_ofs:batch_l]
                ref_batch = traj_ref[batch_ofs:batch_l]
                old_logprob_batch = old_logprob[batch_ofs:batch_l]

                """
                Critic Training
                - critic needs reference values
                """
                self.playing_critic_opt.zero_grad()
                value = self.playing_critic(obs_batch)
                loss_value = torch.nn.MSELoss()(value.squeeze(-1), ref_batch)
                loss_value.backward()
                self.playing_critic_opt.step()

                """
                Actor Training
                - actor needs advantages
                """
                self.playing_actor_opt.zero_grad()

                logits = self.playing_actor(obs_batch)
                logits[~action_mask_batch] = -float("inf")
                logprob_dist = F.log_softmax(logits, dim=1)
                logprob = logprob_dist[torch.arange(batch_size), action_batch]

                ratio = torch.exp(logprob - old_logprob_batch)
                surr_obj = adv_batch * ratio
                c_ratio = torch.clamp(ratio, 1.0 - self.hps['ppo']['CLAMP'], 1.0 + self.hps['ppo']['CLAMP'])
                clipped_surr = adv_batch * c_ratio
                loss_policy = -torch.min(surr_obj, clipped_surr).mean()
                loss_policy.backward()
                self.playing_actor_opt.step()

                sum_loss_value += loss_value.item()
                sum_loss_policy += loss_policy.item()
                count_steps += 1

        self.trajectory_playing.clear()

        self.writer.add_scalar("loss/actor_playing", sum_loss_policy / count_steps, self.game_count)
        self.writer.add_scalar("loss/critic_playing", sum_loss_value / count_steps, self.game_count)
        self.writer.add_scalar("grad/actor_playing", calc_grad(self.playing_actor)[0], self.game_count)
        self.writer.add_scalar("grad/critic_playing", calc_grad(self.playing_critic)[0], self.game_count)
        self.writer.add_scalar("refvals/actor_playing", traj_ref.mean().item(), self.game_count)
        self.writer.add_scalar("refadv/actor_playing", traj_adv.mean().item(), self.game_count)
