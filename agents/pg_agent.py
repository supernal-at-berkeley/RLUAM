from typing import Optional, Sequence
import numpy as np
import torch

from networks.policies import MLPPolicyPG
# from networks.critics import ValueCritic
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        # if use_baseline:
        #     self.critic = ValueCritic(
        #         ob_dim, n_layers, layer_size, baseline_learning_rate
        #     )
        #     self.baseline_gradient_steps = baseline_gradient_steps
        # else:
        self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        rewards = np.concatenate(rewards)
        actions = np.concatenate(actions)
        q_values = np.concatenate(q_values)
        terminals = np.concatenate(terminals)
        obs_final = obs[0]

        for idx in range(1,len(obs)):
            # obs_final = np.array(obs_final, dtype=np.float32)
            # print(obs_final.shape, obs[idx].shape)
            obs_final = np.append(obs_final, obs[idx], axis=0)
        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(obs_final, rewards, q_values, terminals)



        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        info: dict = self.actor.update(obs_final,actions,advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            for _ in range(self.baseline_gradient_steps):
                critic_info: dict = self.critic.update(obs_final, q_values)
                info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values=[]
            for reward in rewards:
                q_values.append(np.array(self._discounted_return(reward),dtype='float32'))


        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = []
            for reward in rewards:
                q_values.append(np.array(self._discounted_reward_to_go(reward),dtype='float32'))
        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            advantages = q_values.copy()
        else:
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs))).reshape(q_values.shape)
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                advantages = q_values - values
            else:
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    delta = rewards[i] + self.gamma * values[i + 1] * (1 - terminals[i]) - values[i]
                    advantages[i] = delta + self.gamma * self.gae_lambda * (1 - terminals[i]) * advantages[i + 1]

                # remove dummy advantage
                advantages = advantages[:-1]

        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        sum = 0.0
        for i in range(len(rewards)):
            sum += rewards[i] * (self.gamma ** i)
        q_list = [sum for _ in range(len(rewards))]
        return q_list


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        q_list = []
        for i in range(len(rewards)):
            _list = rewards[i:-1]
            _sum = 0
            for j in range(len(_list)):
                _sum += _list[j] * (self.gamma ** j)
            q_list.append(_sum)
        return q_list
