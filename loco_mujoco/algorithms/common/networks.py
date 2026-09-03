import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax
from jax import random

from loco_mujoco.core.utils.math import (
    calculate_relative_site_quatities,
    quaternion_relative_rotation,
)


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
    activation: str = "silu"
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
    activation: str = "silu"
    output_activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    full_latent_matrix: bool = False
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    def scale_std(self, log_std):
        # scale the log stds to remove dependency of the noise on the size of the latent state
        log_std = log_std - 0.5 * jnp.log(self.hidden_layer_dims[-1])
        scaled_std = jnp.exp(log_std)
        return scaled_std

    @nn.compact
    def __call__(self, x):

        x = RunningMeanStd()(x)

        # build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        # seperate latent from output layer
        actor_latent = LatticeLatentNet(self.hidden_layer_dims, self.activation,
                                       self.activation, False)(actor_x)
        actor_latent_detached = jax.lax.stop_gradient(actor_latent)
        final_layer = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="W")
        actor_mean = final_layer(actor_latent)
        actor_mean = self.output_activation_fn(actor_mean)

        # add lattice noise
        # get log(S_a)
        actor_mean_logtstd = self.param("mean_log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim, self.hidden_layer_dims[-1]))
        if not self.learnable_std:
            actor_mean_logtstd = jax.lax.stop_gradient(actor_mean_logtstd)
        else:
            actor_mean_logtstd = jnp.clip(actor_mean_logtstd, a_min=jnp.log(1.0e-3), a_max=jnp.log(1.0))
        # get log(S_x)
        if self.full_latent_matrix:
            actor_latent_logtstd = self.param("latent_log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                       (self.hidden_layer_dims[-1], self.hidden_layer_dims[-1]))
        else:
            actor_latent_logtstd = self.param("latent_log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                    (self.hidden_layer_dims[-1],))
        if not self.learnable_std:
            actor_latent_logtstd = jax.lax.stop_gradient(actor_latent_logtstd)
        else:
            actor_latent_logtstd = jnp.clip(actor_latent_logtstd, a_min=jnp.log(1.0e-3), a_max=jnp.log(1.0))
        # compute S_a^2 * x^2
        actor_latent_detached_norm = actor_latent_detached
        actor_latent_detached_norm = actor_latent_detached_norm / (jnp.sqrt(jnp.mean(actor_latent_detached_norm**2, axis=-1, keepdims=True)) + 1e-6)
        actor_mean_var = jnp.einsum("ah,...h->...a", jnp.square(self.scale_std(actor_mean_logtstd)), jnp.square(actor_latent_detached_norm))
        # compute S_x^2 * x^2
        if self.full_latent_matrix:
            actor_latent_var = jnp.einsum("ah,...h->...a", jnp.square(self.scale_std(actor_latent_logtstd)), jnp.square(actor_latent_detached_norm))
        else:
            actor_latent_var = jnp.square(self.scale_std(actor_latent_logtstd)) * jnp.square(actor_latent_detached_norm)
        # get W
        final_layer_weights_T = self.get_variable("params", "W")["kernel"]
        # compute total covariance (W * Diag(S_x^2 * x^2) * W^T) + Diag(S_a^2 * x^2) + Diag(epsilon)
        def cov_x(latent_var):
            return (final_layer_weights_T.mT * latent_var[None, :]) @ final_layer_weights_T
        covx = jax.vmap(cov_x)(jnp.atleast_2d(actor_latent_var))
        def add_diag(cov, diagonal):
            return cov + jnp.diag(diagonal + 1e-6)
        actor_covar = jax.vmap(add_diag)(covx, jnp.atleast_2d(actor_mean_var))

        # create policy using the mean W * x and the covariance
        pi = distrax.MultivariateNormalFullCovariance(actor_mean, actor_covar)

        # build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(critic_x)
        return pi, jnp.squeeze(critic, axis=-1)
    
class Encoder(nn.Module):
    """VAE Encoder.
    Code adopted from J. Heek et al., Flax: A neural network library and ecosystem for JAX. 2024."""

    latent_dim: int = 20
    hidden_layer_dims: Sequence[int] = (1024, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, x):
        x = LatticeLatentNet(self.hidden_layer_dims, self.activation, self.activation, False)(x)
        mean_x = nn.Dense(self.latent_dim, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latent_dim, name='fc2_logvar')(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    """VAE Decoder.
    Code adopted from J. Heek et al., Flax: A neural network library and ecosystem for JAX. 2024."""

    output_dim: Sequence[int]
    hidden_layer_dims: Sequence[int] = (512, 1024)
    activation: str = "silu"

    @nn.compact
    def __call__(self, z):
        z = LatticeLatentNet(self.hidden_layer_dims, self.activation, self.activation, False)(z)
        z = nn.Dense(self.output_dim, name='fc2')(z)
        return z


class VAE(nn.Module):
    """Full VAE model.
    Code adopted from J. Heek et al., Flax: A neural network library and ecosystem for JAX. 2024."""
    output_dim: Sequence[int]
    latent_dim: int = 20
    hidden_layer_dims_enc: Sequence[int] = (1024, 512)
    hidden_layer_dims_dec: Sequence[int] = (512, 1024)
    activation: str = "silu"

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, hidden_layer_dims=self.hidden_layer_dims_enc, activation=self.activation)
        self.decoder = Decoder(output_dim=self.output_dim, hidden_layer_dims=self.hidden_layer_dims_dec, activation=self.activation)

    def reparameterize(self, rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def encode(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(z_rng, mean, logvar)
        return z

    def decode(self, z):
        return self.decoder(z)
    
class CVAE(nn.Module):
    """Conditional VAE model.
    Code inspired by J. Heek et al., Flax: A neural network library and ecosystem for JAX. 2024."""

    output_dim: Sequence[int]
    latent_dim: int = 20
    hidden_layer_dims_enc: Sequence[int] = (1024, 512)
    hidden_layer_dims_dec: Sequence[int] = (512, 1024)
    activation: str = "silu"
    
class CVAE(nn.Module):
    """Conditional VAE model.
    Code inspired by J. Heek et al., Flax: A neural network library and ecosystem for JAX. 2024."""

    output_dim: Sequence[int]
    latent_dim: int = 20
    hidden_layer_dims_enc: Sequence[int] = (1024, 512)
    hidden_layer_dims_dec: Sequence[int] = (512, 1024)
    activation: str = "silu"
    n_step_lookahead: int = 4

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, hidden_layer_dims=self.hidden_layer_dims_enc, activation=self.activation)
        self.decoder = Decoder(output_dim=self.output_dim, hidden_layer_dims=self.hidden_layer_dims_dec, activation=self.activation)

    def reparameterize(self, rng, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std

    def __call__(self, x, condition, z_rng):
        mean, logvar = self.encoder(jnp.concatenate([x, condition], axis=-1))
        z = self.reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(jnp.concatenate([z, condition], axis=-1))
        return z, mean, logvar, recon_x

    def get_cvae_obs(self, traj_data, traj_state, current_qpos, current_qvel, qpos_ind, qvel_ind, quat_in_qpos, backend):
        # traj_data = env.th.traj.data
        # jax.debug.print("traj_state.subtraj_step_no: {x}", x=traj_state.subtraj_step_no)
        
        traj_data_single = traj_data.get(traj_state.traj_no, traj_state.subtraj_step_no, backend)
        qpos_traj = jnp.atleast_2d(traj_data_single.qpos)
        qvel_traj = jnp.atleast_2d(traj_data_single.qvel)
        
        qpos, qvel = current_qpos[:, qpos_ind], current_qvel[:, qvel_ind]
        qpos_quat = qpos[:, quat_in_qpos]
        qpos_quat_traj = qpos_traj[:, qpos_ind][:, quat_in_qpos]
        qpos_quat_rel = jax.vmap(
            lambda q1, q2: quaternion_relative_rotation(q1, q2, backend)
        )(qpos_quat, qpos_quat_traj)

        if self.n_step_lookahead > 0:
            future_qpos_traj = []
            future_qvel_traj = []
            future_site_rpos = []
            future_site_rangles = []
            future_site_rvel = []
            
            # for power of two steps after the current state
            for i in [2**k for k in range(0, self.n_step_lookahead)]:
                
                future_traj_data_single = traj_data.get(traj_state.traj_no, traj_state.subtraj_step_no + i, backend)

                qpos_quat_future_traj = jnp.atleast_2d(future_traj_data_single.qpos)[:, qpos_ind][:, quat_in_qpos]
                future_qpos_quat_rel = jax.vmap(
                    lambda q1, q2: quaternion_relative_rotation(q1, q2, backend)
                )(qpos_quat, qpos_quat_future_traj)
                future_qpos_traj.append(backend.concatenate([future_qpos_quat_rel, jnp.atleast_2d(future_traj_data_single.qpos)[:, qpos_ind][:, ~quat_in_qpos] - qpos[:, ~quat_in_qpos]], axis=-1))
                future_qvel_traj.append(jnp.atleast_2d(future_traj_data_single.qvel)[:, qvel_ind] - qvel)

            future_qpos_traj = backend.stack(future_qpos_traj, axis=1)
            future_qvel_traj = backend.stack(future_qvel_traj, axis=1)

            traj_goal_obs = backend.concatenate([
                # Trajectory joint positions and velocities for the current state
                qpos_quat_rel,
                qpos_traj[:, qpos_ind][:, ~quat_in_qpos] - qpos[:, ~quat_in_qpos],
                qvel_traj[:, qvel_ind] - qvel,
                # # Trajectory site positions, angles and velocities relative to the root site for the current state
                # backend.ravel(site_rpos),
                # backend.ravel(site_rangles),
                # backend.ravel(site_rvel),
                # Trajectory joint positions and velocities for the future states
                future_qpos_traj.reshape(future_qpos_traj.shape[0], -1),
                future_qvel_traj.reshape(future_qvel_traj.shape[0], -1),
            ], axis=1)
        else:
            traj_goal_obs = backend.concatenate([
                # Trajectory joint positions and velocities of the current simulation state
                qpos_quat_rel,
                qpos_traj[:, qpos_ind][:, ~quat_in_qpos] - qpos[:, ~quat_in_qpos],
                qvel_traj[:, qvel_ind] - qvel,
                # Trajectory site positions, angles and velocities relative to the root site of the current simulation state
                # backend.ravel(site_rpos),
                # backend.ravel(site_rangles),
                # backend.ravel(site_rvel),
            ], axis=1)

        return traj_goal_obs

    # def encode(self, x, condition, z_rng):
    #     mean, logvar = self.encoder(jnp.concatenate([x, condition], axis=-1))
    #     z = self.reparameterize(z_rng, mean, logvar)
    #     return z

    # def decode(self, z, condition):
    #     return self.decoder(jnp.concatenate([z, condition], axis=-1))

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
