import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


def get_activation_fn(name: str):
    """ Get activation function by name from the flax.linen module."""
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class FullyConnectedNet(nn.Module):

    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        # add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x

class LatticeLatentNet(nn.Module):

    hidden_layer_dims: Sequence[int]
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):

        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # build network
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation_fn(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        actor_mean = FullyConnectedNet(self.hidden_layer_dims, self.action_dim, self.activation,
                                       "tanh", False, False)(actor_x)
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(critic_x)

        return pi, jnp.squeeze(critic, axis=-1)

class LatticeActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    output_activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        # seperate latent from output layer
        actor_latent = LatticeLatentNet(self.hidden_layer_dims, self.activation,
                                       "tanh", False)(actor_x)
        final_layer = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="W")
        actor_mean = final_layer(actor_latent)
        actor_mean = self.output_activation_fn(actor_mean)

        # add lattice noise
        # get log(S_a)
        actor_mean_logtstd = self.param("mean_log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim, self.hidden_layer_dims[-1]))
        if not self.learnable_std:
            actor_mean_logtstd = jax.lax.stop_gradient(actor_mean_logtstd)
        # get log(S_x)
        actor_latent_logtstd = self.param("latent_log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.hidden_layer_dims[-1], self.hidden_layer_dims[-1]))
        if not self.learnable_std:
            actor_latent_logtstd = jax.lax.stop_gradient(actor_latent_logtstd)
        # compute S_a^2 * x^2
        actor_mean_var = jnp.einsum("ah,...h->...a", jnp.exp(2.0 * actor_mean_logtstd), jnp.square(actor_latent))
        # compute S_x^2 * x^2
        actor_latent_var = jnp.einsum("ah,...h->...a", jnp.exp(2.0 * actor_latent_logtstd), jnp.square(actor_latent))
        # get W
        final_layer_weights = self.get_variable("params", "W")["kernel"].mT
        # compute total covariance (W * Diag(S_x^2 * x^2) * W^T) + Diag(S_a^2 * x^2)
        covx = jnp.einsum("ah,...h,jh->...aj", final_layer_weights, actor_latent_var, final_layer_weights)
        cova = jnp.einsum("...i,ij->...ij", actor_mean_var, jnp.eye(self.action_dim, dtype=actor_mean_var.dtype))
        actor_covar = covx + cova

        # create policy using the mean W * x and the covariance
        pi = distrax.MultivariateNormalFullCovariance(actor_mean, actor_covar)

        # build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(critic_x)
        return pi, jnp.squeeze(critic, axis=-1)


class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and variance for input normalization."""

    @nn.compact
    def __call__(self, x):

        x = jnp.atleast_2d(x)

        # Initialize running mean, variance, and count
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        var = self.variable('run_stats', 'var', lambda: jnp.ones(x.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

        # Compute batch mean and variance
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0) + 1e-6  # Add epsilon for numerical stability
        batch_count = x.shape[0]

        # Update counts
        updated_count = count.value + batch_count

        # Numerically stable mean and variance update
        delta = batch_mean - mean.value
        new_mean = mean.value + delta * batch_count / updated_count

        # Compute the new variance using Welford's method
        m_a = var.value * count.value
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
        new_var = M2 / updated_count

        # Normalize input
        normalized_x = (x - new_mean) / jnp.sqrt(new_var + 1e-8)

        # Update state variables
        mean.value = new_mean
        var.value = new_var
        count.value = updated_count

        return jnp.squeeze(normalized_x)
