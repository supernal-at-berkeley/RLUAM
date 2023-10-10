import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()
        self.num_bins = 8
        if discrete:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        action = self(ptu.from_numpy(obs)).sample().numpy()
        # action = np.array([int(char) for char in str(action)])
        # if len(action) < 4:
        #     action = np.concatenate([action, np.repeat(0, 4-len(action))])
        # elif len(action) > 4:
        #     action = action[:4]
        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            mean = self.mean_net(obs)
            mean = torch.relu(mean)
            # Ensure the covariance matrix is positive definite by creating a diagonal matrix
            covariance_matrix = torch.diag(torch.exp(self.logstd))
            action_distribution = torch.distributions.MultivariateNormal(mean, covariance_matrix)

            # Discretize each dimension of the distribution
            bin_edges = [torch.linspace(0, 8, self.num_bins + 1) for _ in range(4)]  # Adjust the range as needed
            bin_probs = torch.zeros([self.num_bins] * 4)

            # Sample from the multivariate normal distribution
            samples = action_distribution.sample([10])  # Adjust the number of samples as needed

            # Count samples in each bin for each dimension
            # for dim in range(4):
            #     for i in range(self.num_bins):
            #         lower_bound = bin_edges[dim][i]
            #         upper_bound = bin_edges[dim][i + 1]
            #         bin_probs[i] = ((samples[:, dim] > lower_bound) & (
            #                     samples[:, dim] <= upper_bound)).sum().item() / samples.size(0)
            #
            # # Create a categorical distribution with the computed probabilities
            # discrete_distribution = distributions.Categorical(bin_probs.view(-1))
            # return discrete_distribution

            return action_distribution

        else:
            action_distribution = torch.distributions.Normal(self.mean_net(obs), torch.exp(self.logstd))
            return action_distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        if self.discrete:
            forward = self.forward(obs).log_prob(actions) * advantages
            forward = forward.mean()
            loss = -forward
        else:
            forward = self.forward(obs).log_prob(actions).sum(dim=-1) * advantages
            forward = forward.mean()
            loss = -forward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
