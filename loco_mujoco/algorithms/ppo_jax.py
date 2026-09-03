import ast
import os
from omegaconf import open_dict
import warnings
from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig, OmegaConf, ListConfig
import csv

import numpy as np
import jax
from jax import config, lax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
from flax import struct
from flax import linen as nn
import flax
import optax
import wandb
import mujoco
import pandas as pd

from loco_mujoco.algorithms import (JaxRLAlgorithmBase, AgentConfBase, AgentStateBase, ActorCritic, LatticeActorCritic,
                                    CVAE, Transition, TrainState, TrainStateBuffer, MetricHandlerTransition)
from loco_mujoco.core.wrappers import LogWrapper, NStepWrapper, LogEnvState, VecEnv, NormalizeVecReward, SummaryMetrics
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from loco_mujoco.core.utils.mujoco import (
    mj_jntid2qposid, mj_jntid2qvelid,
)


@dataclass(frozen=True)
class PPOAgentConf(AgentConfBase):
    config: DictConfig
    network: ActorCritic
    tx: Any

    def serialize(self):
        """
        Serialize the agent configuration and network configuration.

        Returns:
            Serialized agent configuration as a dictionary.

        """
        conf_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True)
        serialized_network = flax.serialization.to_state_dict(self.network)
        return {"config": conf_dict, "network": serialized_network}

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx = PPOJax._get_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                   tx=tx)


@struct.dataclass
class PPOAgentState(AgentStateBase):
    train_state: TrainState

    def serialize(self):
        serialized_train_state = flax.serialization.to_state_dict(self.train_state)
        return {"train_state": serialized_train_state}

    @classmethod
    def from_dict(cls, d, agent_conf):
        train_state = TrainState(apply_fn=agent_conf.network, tx=agent_conf.tx, **d["train_state"])
        return cls(train_state)


class PPOJax(JaxRLAlgorithmBase):

    _agent_conf = PPOAgentConf
    _agent_state = PPOAgentState

    @classmethod
    def init_agent_conf(cls, env, config):

        with (open_dict(config.experiment)):
            config.experiment.num_updates = (
                    config.experiment.total_timesteps // config.experiment.num_steps // config.experiment.num_envs)
            config.experiment.minibatch_size = (
                    config.experiment.num_envs * config.experiment.num_steps // config.experiment.num_minibatches)
            config.experiment.validation_interval = config.experiment.num_updates // config.experiment.validation.num
            config.experiment.validation.num = int(
                config.experiment.num_updates // config.experiment.validation_interval)

        # INIT NETWORK
        hidden_layers = config.experiment.hidden_layers \
            if isinstance(config.experiment.hidden_layers, (list, ListConfig)) \
            else ast.literal_eval(config.experiment.hidden_layers)
        if hasattr(config.experiment, "actor_obs_group") and config.experiment.actor_obs_group is not None:
            actor_obs_ind = env.obs_container.get_obs_ind_by_group(config.experiment.actor_obs_group)
        else:
            actor_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config.experiment, "critic_obs_group") and config.experiment.critic_obs_group is not None:
            critic_obs_ind = env.obs_container.get_obs_ind_by_group(config.experiment.critic_obs_group)
        else:
            critic_obs_ind = jnp.arange(env.mdp_info.observation_space.shape[0])
        if hasattr(config.experiment, "len_obs_history") and config.experiment.len_obs_history > 1:
            obs_len = env.info.observation_space.shape[0]
            actor_obs_ind = jnp.concatenate([actor_obs_ind + i*obs_len
                                             for i in range(config.experiment.len_obs_history)])
            critic_obs_ind = jnp.concatenate([critic_obs_ind + i*obs_len
                                              for i in range(config.experiment.len_obs_history)])
        if hasattr(config.experiment, "cvae") and config.experiment.cvae.latent_dim > 0:
            actor_obs_ind = jnp.concatenate([actor_obs_ind, jnp.arange(actor_obs_ind.max() + 1, actor_obs_ind.max() + 1 + config.experiment.cvae.latent_dim)])
            critic_obs_ind = jnp.concatenate([critic_obs_ind, jnp.arange(critic_obs_ind.max() + 1, critic_obs_ind.max() + 1 + config.experiment.cvae.latent_dim)])
            
        actorcritic_class = LatticeActorCritic if "use_lattice" in config.experiment and config.experiment.use_lattice else ActorCritic
        network = actorcritic_class(
            env.info.action_space.shape[0],
            activation=config.experiment.activation,
            init_std=config.experiment.init_std,
            learnable_std=config.experiment.learnable_std,
            hidden_layer_dims=hidden_layers,
            actor_obs_ind=actor_obs_ind,
            critic_obs_ind=critic_obs_ind,
            **({"full_latent_matrix": config.experiment.full_latent_matrix} if actorcritic_class == LatticeActorCritic else {})
        )
        if hasattr(config.experiment, "cvae") and config.experiment.cvae.use_cvae:
            cvae = CVAE(output_dim=jnp.size(env.obs_container["GoalTrajMimic"].obs_ind) * (config.experiment.cvae.n_step_lookahead + 1), 
                        latent_dim=config.experiment.cvae.latent_dim,
                        hidden_layer_dims_enc=config.experiment.cvae.hidden_layers_enc, 
                        hidden_layer_dims_dec=config.experiment.cvae.hidden_layers_dec,
                        activation=config.experiment.cvae.activation,
                        n_step_lookahead=config.experiment.cvae.n_step_lookahead)
            network = (network, cvae)

        # set up optimizers
        tx = cls._get_optimizer(config)

        return cls._agent_conf(config, network, tx)

    @classmethod
    def _get_optimizer(cls, config):
        if config.experiment.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(config.experiment.max_grad_norm),
                optax.adamw(weight_decay=config.experiment.weight_decay, eps=1e-5,
                            learning_rate=lambda count: cls._linear_lr_schedule(count, config.experiment.num_minibatches,
                                                                                config.experiment.update_epochs, config.lr,
                                                                                config.experiment.num_updates))
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.experiment.max_grad_norm),
                optax.adamw(config.experiment.lr, weight_decay=config.experiment.weight_decay, eps=1e-5),
            )

        tx = optax.apply_if_finite(tx, max_consecutive_errors=10000000)

        return tx
    
    @classmethod
    def log_wandb(cls, metrics, step):
        wandb_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), metrics)

        wandb.log({
            "Training Info/Mean Episode Return": wandb_metrics.mean_episode_return,
            "Training Info/Mean Episode Length": wandb_metrics.mean_episode_length,
            "Training Info/Max Timestep": wandb_metrics.max_timestep,
            "Training Info/Var Episode Return": wandb_metrics.var_episode_return,
            "Training Info/Var Episode Length": wandb_metrics.var_episode_length,
            "Training Info/Min Timestep": wandb_metrics.min_timestep,
            "Training Info/Max Episode Return": wandb_metrics.max_episode_return,
            "Training Info/Max Episode Length": wandb_metrics.max_episode_length,
            "Training Info/Min Episode Return": wandb_metrics.min_episode_return,
            "Training Info/Min Episode Length": wandb_metrics.min_episode_length,
            "Training Info/Num Episodes": wandb_metrics.num_episodes,
            "Training Info/Absorbing Episodes": wandb_metrics.absorbing_episodes,
            "Training Info/Success Rate": wandb_metrics.success_rate,
            "Training Info/Total Loss": wandb_metrics.total_loss,
            "Training Info/Loss Critic": wandb_metrics.value_loss,
            "Training Info/Loss Actor": wandb_metrics.loss_actor,
            "Training Info/CVAE Reconstruction Loss": wandb_metrics.recon_loss,
            "Training Info/CVAE KL-Divergence Loss": wandb_metrics.kld_loss,
        }, step=step)

    @classmethod
    def log_val_wandb(cls, metrics, step):
        wandb_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), metrics)

        wandb.log({
            "Validation Info/Mean Episode Return": wandb_metrics.mean_episode_return,
            "Validation Info/Mean Episode Length": wandb_metrics.mean_episode_length,
            "Validation Info/Max Timestep": wandb_metrics.max_timestep,
            "Validation Info/Var Episode Return": wandb_metrics.var_episode_return,
            "Validation Info/Var Episode Length": wandb_metrics.var_episode_length,
            "Validation Info/Min Timestep": wandb_metrics.min_timestep,
            "Validation Info/Max Episode Return": wandb_metrics.max_episode_return,
            "Validation Info/Max Episode Length": wandb_metrics.max_episode_length,
            "Validation Info/Min Episode Return": wandb_metrics.min_episode_return,
            "Validation Info/Min Episode Length": wandb_metrics.min_episode_length,
            "Validation Info/Num Episodes": wandb_metrics.num_episodes,
            "Validation Info/Absorbing Episodes": wandb_metrics.absorbing_episodes,
            "Validation Info/Success Rate": wandb_metrics.success_rate,
        }, step=step)

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: PPOAgentConf,
                  agent_state: PPOAgentState = None,
                  mh: MetricsHandler = None):

        # extract static agent info
        config, network, tx =\
            (agent_conf.config.experiment, agent_conf.network, agent_conf.tx)

        env = cls._wrap_env(env, config)

        # # derive useful observation index masks from env.obs_container
        # # (current qpos, current qvel, future/reference, last_action)
        # obs_items = list(env.obs_container.items())
        # current_qpos_idx = np.concatenate([obs.obs_ind for _, obs in obs_items
        #                                    if obs.__class__.data_type() == "qpos"]) \
        #     if any(obs.__class__.data_type() == "qpos" for _, obs in obs_items) else np.array([], dtype=int)
        # current_qvel_idx = np.concatenate([obs.obs_ind for _, obs in obs_items
        #                                    if obs.__class__.data_type() == "qvel"]) \
        #     if any(obs.__class__.data_type() == "qvel" for _, obs in obs_items) else np.array([], dtype=int)

        # if "last_action" in env.obs_container:
        #     last_action_idx = np.array(env.obs_container["last_action"].obs_ind, dtype=int)
        # else:
        #     last_action_idx = np.concatenate([
        #         obs.obs_ind for _, obs in obs_items
        #         if obs.__class__.__name__ == "LastAction" or (obs.name and "last_action" in obs.name.lower())
        #     ]) if any(obs.__class__.__name__ == "LastAction" or (obs.name and "last_action" in obs.name.lower()) for _, obs in obs_items) else np.array([], dtype=int)

        # # convert to jax arrays for indexing later
        # current_qpos_idx = jnp.array(current_qpos_idx, dtype=int)
        # current_qvel_idx = jnp.array(current_qvel_idx, dtype=int)
        # future_idx = jnp.array(future_idx, dtype=int)
        # last_action_idx = jnp.array(last_action_idx, dtype=int)

        # Get observation indices to give as input and condition to CVAE
        current_state_idx = env._obs_indices.concatenated_indices
        reference_state_idx = env.obs_container["GoalTrajMimic"].obs_ind
        root_free_joint_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_JOINT, env.root_free_joint_xml_name)
        qpos_ind = np.concatenate(
            [mj_jntid2qposid(i, env._model)[2:] for i in range(env._model.njnt) if i == root_free_joint_id] +
            [mj_jntid2qposid(i, env._model) for i in range(env._model.njnt) if i != root_free_joint_id]
        )
        qvel_ind = np.concatenate([mj_jntid2qvelid(i, env._model) for i in range(env._model.njnt)])
        quat_in_qpos_ind = np.concatenate([mj_jntid2qposid(i, env._model)[3:] for i in range(env._model.njnt) if i == root_free_joint_id])
        quat_in_qpos = np.array([True if q in quat_in_qpos_ind else False for q in qpos_ind])

        # extract current agent state
        if agent_state is not None:
            train_state = agent_state.train_state
            if hasattr(config, "cvae") and config.cvae.use_cvae:
                train_state_cvae = train_state[1]
                train_state = train_state[0]
        else:
            train_state = None
            if hasattr(config, "cvae") and config.cvae.use_cvae:
                train_state_cvae = None

        if hasattr(config, "cvae") and config.cvae.use_cvae:
            network, cvae = network
            if train_state_cvae is None:
                rng, _rng1, _rng2 = jax.random.split(rng, 3)
                init_x = jnp.zeros(jnp.size(current_state_idx) * (config.cvae.n_step_lookahead + 1))
                init_condition = jnp.zeros(jnp.size(reference_state_idx))
                cvae_params = cvae.init(_rng1, init_x, init_condition, _rng2)

        if train_state is None:
            obs_dim = env.info.observation_space.shape[0]
            if hasattr(config, "cvae") and config.cvae.use_cvae:
                obs_dim = obs_dim + config.experiment.cvae.latent_dim

            rng, _rng1, _rng2 = jax.random.split(rng, 3)
            init_x = jnp.zeros((obs_dim,))
            network_params = network.init(_rng1, init_x)

        else:
            raise NotImplementedError("Loading of train state not implemented yet.")

        # init new train states from old params
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"] if train_state is None else train_state.params,
            run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
            tx=tx,
        )

        if hasattr(config, "cvae") and config.cvae.use_cvae:
            train_state_cvae = TrainState.create(
                apply_fn=cvae.apply,
                params=cvae_params["params"] if train_state_cvae is None else train_state_cvae.params,
                run_stats=network_params["run_stats"] if train_state_cvae is None else train_state_cvae.run_stats,
                tx=tx,
            )
            train_state = (train_state, train_state_cvae)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        # TRAIN LOOP
        @scan_tqdm(config.num_updates, print_rate=1, desc='PPO JAX Training')
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, step):
                train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                if hasattr(config, "cvae") and config.cvae.use_cvae:
                    train_state_cvae = train_state[1]
                    train_state = train_state[0]
                    cvae_state = {'params': train_state_cvae.params, 'run_stats': train_state_cvae.run_stats}
                    rng, z_rng = jax.random.split(rng)
                    current_state = last_obs[..., current_state_idx]
                    state_difference = cvae.get_cvae_obs(traj_data=env.th.traj.data, 
                                                         traj_state=env_state.additional_carry.traj_state, 
                                                        #  goal_obs=env.obs_container["GoalTrajMimic"],  
                                                         current_qpos=env_state.data.qpos,
                                                         current_qvel=env_state.data.qvel,
                                                         qpos_ind=qpos_ind,
                                                         qvel_ind=qvel_ind,
                                                         quat_in_qpos=quat_in_qpos,
                                                         backend=jnp,)
                    z, mean, logvar, recon_x = cvae.apply(cvae_state, state_difference, current_state, z_rng)
                    policy_obs = jnp.concatenate([last_obs, z], axis=-1)
                    cvae_obs_x = state_difference
                    cvae_obs_cond = current_state
                else:
                    policy_obs = last_obs
                    cvae_obs_x = None
                    cvae_obs_cond = None
                state = {'params': train_state.params, 'run_stats': train_state.run_stats}
                y, updates = network.apply(state, policy_obs, mutable=["run_stats", "noise"] if config.use_lattice else ["run_stats"])
                pi, value = y
                action = pi.sample(seed=_rng)
                train_state = train_state.replace(run_stats=updates['run_stats'])   # update stats
                if config.debug:
                    jax.debug.print("action: {s} {x}", s=action.shape, x=action)
                log_prob = pi.log_prob(action)

                # STEP ENV
                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                # GET METRICS
                log_env_state = env_state.find(LogEnvState)
                logged_metrics = log_env_state.metrics

                transition = Transition(
                    done, absorbing, action, value, reward, log_prob, last_obs, info, env_state.additional_carry.traj_state,
                    logged_metrics, cvae_obs_x, cvae_obs_cond
                )
                if hasattr(config, "cvae") and config.cvae.use_cvae:
                    train_state = (train_state, train_state_cvae)
                runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config.num_steps), config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, train_state_buffer, rng = runner_state
            if hasattr(config, "cvae") and config.cvae.use_cvae:
                train_state_cvae = train_state[1]
                train_state = train_state[0]
                cvae_state = {'params': train_state_cvae.params, 'run_stats': train_state_cvae.run_stats}
                rng, z_rng = jax.random.split(rng)
                current_state = last_obs[..., current_state_idx]
                state_difference = cvae.get_cvae_obs(traj_data=env.th.traj.data, 
                                                    traj_state=env_state.additional_carry.traj_state, 
                                                    # goal_obs=env.obs_container["GoalTrajMimic"], 
                                                    current_qpos=env_state.data.qpos,
                                                    current_qvel=env_state.data.qvel,
                                                    qpos_ind=qpos_ind,
                                                    qvel_ind=qvel_ind,
                                                    quat_in_qpos=quat_in_qpos,
                                                    backend=jnp,)
                z, mean, logvar, recon_x = cvae.apply(cvae_state, state_difference, current_state, z_rng)
                policy_obs = jnp.concatenate([last_obs, z], axis=-1)
            else:
                policy_obs = last_obs
            state = {'params': train_state.params, 'run_stats': train_state.run_stats}
            if config.use_lattice:
                state['noise'] = train_state.lattice_noise
            y, _ = network.apply(state, policy_obs, mutable=["run_stats", "noise"] if config.use_lattice else ["run_stats"])
            *pi, last_val = y

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward, obs = (
                        transition.done,
                        transition.absorbing,
                        transition.value,
                        transition.reward,
                        transition.obs
                    )

                    delta = reward + config.gamma * next_value * (1 - absorbing) - value
                    gae = (
                        delta
                        + config.gamma * config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    if hasattr(config, "cvae") and config.cvae.use_cvae:
                        train_state_cvae = train_state[1]
                        train_state = train_state[0]

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        obs = traj_batch.obs
                        if hasattr(config, "cvae") and config.cvae.use_cvae:
                            params, cvae_params = params
                            cvae_state = {'params': cvae_params, 'run_stats': train_state_cvae.run_stats}
                            rng_, z_rng = jax.random.split(rng)
                            cvae_obs_x, cvae_obs_cond = traj_batch.cvae_obs_x, traj_batch.cvae_obs_cond
                            z, mean, logvar, recon_x = cvae.apply(cvae_state, cvae_obs_x, cvae_obs_cond, z_rng)
                            obs = jnp.concatenate([traj_batch.obs, z], axis=-1)
                        state = {'params': params, 'run_stats': train_state.run_stats}
                        if config.use_lattice:
                            state['noise'] = train_state.lattice_noise
                        y, _ = network.apply(state, obs, mutable=["run_stats", "noise"] if config.use_lattice else ["run_stats"])
                        *pi, value = y
                        pi = pi[0]
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE PPO ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config.clip_eps,
                                    1.0 + config.clip_eps,
                                )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi[0].entropy().mean()

                        #CALCULATE CVAE LOSS
                        if hasattr(config, "cvae") and config.cvae.use_cvae:
                            @jax.vmap
                            def kl_divergence(mean, logvar):
                                return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
                            
                            recon_loss = jnp.mean(jnp.sum((recon_x - cvae_obs_x)**2, axis=1))
                            kld_loss = kl_divergence(mean, logvar).mean()

                        if hasattr(config, "cvae") and config.cvae.use_cvae:
                            total_loss = (
                                loss_actor
                                + config.vf_coef * value_loss
                                - config.ent_coef * entropy
                                + config.cvae.recon_coef * recon_loss 
                                + config.cvae.kl_coef * kld_loss
                            )
                            return total_loss, (value_loss, loss_actor, entropy, recon_loss, kld_loss)
                        else:
                            total_loss = (
                                loss_actor
                                + config.vf_coef * value_loss
                                - config.ent_coef * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                    if hasattr(config, "cvae") and config.cvae.use_cvae:
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        (total_loss, aux), (grads_policy, grads_cvae) = grad_fn(
                            (train_state.params, train_state_cvae.params), traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads_policy)
                        train_state_cvae = train_state_cvae.apply_gradients(grads=grads_cvae)
                        train_state = (train_state, train_state_cvae)
                        total_loss = (total_loss, aux)
                    else:
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            if hasattr(config, "cvae") and config.cvae.use_cvae:
                train_state = (train_state, train_state_cvae)
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            if hasattr(config, "cvae") and config.cvae.use_cvae:
                train_state_cvae = train_state[1]
                train_state = train_state[0]
            rng = update_state[-1]

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            if hasattr(config, "cvae") and config.cvae.use_cvae:
                train_state = (train_state, train_state_cvae)

            logged_metrics = traj_batch.metrics

            mean_episode_return = jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done)
            mean_episode_length = jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done)

            metric = SummaryMetrics(
                mean_episode_return=mean_episode_return,
                mean_episode_length=mean_episode_length,
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
                var_episode_return=jnp.sum(jnp.square(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns - mean_episode_return, 0.0))) / jnp.sum(logged_metrics.done),
                var_episode_length=jnp.sum(jnp.square(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths - mean_episode_length, 0.0))) / jnp.sum(logged_metrics.done),
                min_timestep=jnp.min(logged_metrics.timestep * config.num_envs),
                max_episode_return=jnp.max(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)),
                max_episode_length=jnp.max(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)),
                min_episode_return=jnp.min(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 9999.9)),
                min_episode_length=jnp.min(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 9999.9)),
                num_episodes=jnp.sum(logged_metrics.done),
                absorbing_episodes=jnp.sum(logged_metrics.absorbing),
                success_rate = lax.cond(
                    jnp.sum(logged_metrics.done) > 0,
                    lambda _: 1.0 - (jnp.sum(logged_metrics.absorbing) / jnp.sum(logged_metrics.done)),
                    lambda _: 0.0,
                    operand=None
                ),
                total_loss = loss_info[0],
                value_loss = loss_info[1][0], 
                loss_actor = loss_info[1][1], 
                recon_loss = jnp.mean(loss_info[1][3]) if hasattr(config, "cvae") and config.cvae.use_cvae else 0.0,#jnp.clip(jnp.mean(loss_info[1][3]), -100000, 100000),
                kld_loss = jnp.mean(loss_info[1][4]) if hasattr(config, "cvae") and config.cvae.use_cvae else 0.0,#jnp.clip(jnp.mean(loss_info[1][4]), -100000, 100000),
            )

            jax.debug.callback(cls.log_wandb, metric, config.num_steps * config.num_envs * counter)

            def _evaluation_step():

                def _eval_env(runner_state, unused):
                    train_state, env_state, last_obs, train_state_buffer, rng = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    if hasattr(config, "cvae") and config.cvae.use_cvae:
                        train_state_cvae = train_state[1]
                        train_state = train_state[0]
                        cvae_state = {'params': train_state_cvae.params, 'run_stats': train_state_cvae.run_stats}
                        rng, z_rng = jax.random.split(rng)
                        current_state = last_obs[..., current_state_idx]
                        state_difference = cvae.get_cvae_obs(traj_data=env.th.traj.data, 
                                                            traj_state=env_state.additional_carry.traj_state, 
                                                            # goal_obs=env.obs_container["GoalTrajMimic"], 
                                                            current_qpos=env_state.data.qpos,
                                                            current_qvel=env_state.data.qvel,
                                                            qpos_ind=qpos_ind,
                                                            qvel_ind=qvel_ind,
                                                            quat_in_qpos=quat_in_qpos,
                                                            backend=jnp,)
                        z, mean, logvar, recon_x = cvae.apply(cvae_state, state_difference, current_state, z_rng)
                        policy_obs = jnp.concatenate([last_obs, z], axis=-1)
                    else:
                        policy_obs = last_obs
                    state = {'params': train_state.params, 'run_stats': train_state.run_stats}
                    if config.use_lattice:
                        state['noise'] = train_state.lattice_noise
                    y, updates = train_state.apply_fn(state, policy_obs, 
                                                      mutable=["run_stats", "noise"] if config.use_lattice else ["run_stats"])
                    *pi, value = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])  # update stats
                    action = pi[0].mean()

                    # STEP ENV
                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    # GET METRICS
                    log_env_state = env_state.find(LogEnvState)
                    logged_metrics = log_env_state.metrics

                    transition = MetricHandlerTransition(env_state, logged_metrics)

                    if hasattr(config, "cvae") and config.cvae.use_cvae:
                        train_state = (train_state, train_state_cvae)

                    runner_state = (train_state, env_state, obsv, train_state_buffer, rng)
                    return runner_state, transition

                rng = runner_state[-1]
                reset_rng = jax.random.split(rng, config.validation.num_envs)
                obsv, env_state = env.reset(reset_rng)
                runner_state_eval = (train_state, env_state, obsv, train_state_buffer, rng)

                # do evaluation runs
                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, runner_state_eval, None, config.validation.num_steps
                )

                env_states = traj_batch_eval.env_state

                validation_metrics = mh(env_states)

                jax.debug.callback(cls.log_val_wandb, validation_metrics, config.num_steps * config.num_envs * counter)

                return validation_metrics

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                   mh.get_zero_container)

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, env_state.metrics)

            # add train state to buffer if needed
            train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                              lambda x, y: TrainStateBuffer.add(x, y),
                                              lambda x, y: x, train_state_buffer, train_state)

            runner_state = (train_state, env_state, last_obs, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config.num_updates)
        )

        agent_state = cls._agent_state(train_state=runner_state[0])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}

    @classmethod
    @classmethod
    def play_policy(cls, env,
                    agent_conf: PPOAgentConf,
                    agent_state: PPOAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, save_kinematics=False,
                    save_kinematics_folder="./LocoMuJoCo_recordings"):

        if use_mujoco and wrap_env:
            if hasattr(agent_conf.config.experiment, "len_obs_history"):
                assert agent_conf.config.experiment.len_obs_history == 1, "len_obs_history must be 1 for mujoco envs."
        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."
        # if save_kinematics:
        #     assert use_mujoco, "Kinematics can only be saved for mujoco envs."

        config = agent_conf.config.experiment
        train_state = agent_state.train_state

        # print("env.obs_container.names()", env.obs_container.names())
        # print("env.obs_container.entries()", env.obs_container.entries())
        # jax.debug.print("env.obs_container: {x}", x=env.obs_container)

        # Indices from observation, in order of MuJoCo observations
        current_state_idx = np.argsort(env._obs_indices.concatenated_indices)
        if "GoalTrajMimic" in env.obs_container.names():
            reference_state_idx = env.obs_container["GoalTrajMimic"].obs_ind[current_state_idx]
        elif "GoalTrajMimicv2" in env.obs_container.names():
            reference_state_idx = env.obs_container["GoalTrajMimicv2"].obs_ind[current_state_idx]
        else:
            if hasattr(config, "cvae") and config.cvae.use_cvae:
                raise NotImplementedError("Using CVAE without GoalTrajMimic not implemented yet.")

        root_free_joint_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_JOINT, env.root_free_joint_xml_name)
        qpos_ind = np.concatenate(
            [mj_jntid2qposid(i, env._model)[2:] for i in range(env._model.njnt) if i == root_free_joint_id] +
            [mj_jntid2qposid(i, env._model) for i in range(env._model.njnt) if i != root_free_joint_id]
        )
        qvel_ind = np.concatenate([mj_jntid2qvelid(i, env._model) for i in range(env._model.njnt)])
        quat_in_qpos_ind = np.concatenate([mj_jntid2qposid(i, env._model)[3:] for i in range(env._model.njnt) if i == root_free_joint_id])
        print("qpos_ind:", qpos_ind)
        print("quat_in_qpos_ind:", quat_in_qpos_ind)
        quat_in_qpos = np.array([True if q in quat_in_qpos_ind else False for q in qpos_ind])

        for o in env.obs_container.values():
            if getattr(o, "name", None) == "q_root":
                print(f"{o}.obs_ind {o.obs_ind}")

        def sample_actions(ts, obs, _rng):
            # if config.use_lattice:
            #     y, updates = agent_conf.network.apply({'params': ts.params,
            #                                         'run_stats': ts.run_stats,
            #                                         # 'noise': ts.lattice_noise
            #                                         },
            #                                             obs, mutable=["run_stats"])#, "noise"])
            #     ts = ts.replace(run_stats=updates['run_stats'])  # update stats
            #     *pi, _ = y
            #     a = pi[0].mean() if deterministic else (pi[1] if config.use_lattice else pi[0].sample(seed=_rng))
            # else: 2
            if hasattr(config, "cvae") and config.cvae.use_cvae:
                train_state_cvae = ts[1]
                ts = ts[0]
                network, cvae = agent_conf.network
                cvae_state = {'params': train_state_cvae.params, 'run_stats': train_state_cvae.run_stats}
                _rng, z_rng = jax.random.split(_rng)
                current_state = obs[..., current_state_idx]
                state_difference = cvae.get_cvae_obs(traj_data=env.th.traj.data, 
                                                    traj_state=env_state.additional_carry.traj_state, 
                                                    # goal_obs=env.obs_container["GoalTrajMimic"], 
                                                    current_qpos=env_state.data.qpos,
                                                    current_qvel=env_state.data.qvel,
                                                    qpos_ind=qpos_ind,
                                                    qvel_ind=qvel_ind,
                                                    quat_in_qpos=quat_in_qpos,
                                                    backend=jnp,)
                z, mean, logvar, recon_x = cvae.apply(cvae_state, state_difference, current_state, z_rng)
                jax.debug.print("state_difference: {x}", x=state_difference)
                jax.debug.print("recon_x: {x}", x=recon_x)
                jax.debug.print("current_state: {x}", x=current_state)
                policy_obs = jnp.concatenate([obs, z], axis=-1)
            else:
                network = agent_conf.network
                policy_obs = obs

            y, updates = network.apply({'params': ts.params, 'run_stats': ts.run_stats}, policy_obs, mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])  # update stats
            pi, _ = y
            a = pi.mean() if deterministic else pi.sample(seed=_rng)

            if hasattr(config, "cvae") and config.cvae.use_cvae:
                ts = (ts, train_state_cvae)
                
            return a, ts
        
        def process_kinematics(env, kinematics_history, evaluation_stats_csv_file_path, motion_csv_file_path, 
                               joint_types_to_fetch=["JointPos", "JointVel"]):
            # Calculate kinematics statistics for each joint and add to CSV rows
            kinematics_history = np.degrees(kinematics_history)
            csv_stats = []
            joint_indices = []
            joint_names = []
            for o in env.obs_container.values():
                print(f"idx {o.obs_ind}, name {o.name}, type {o.__class__.__name__}")
            for observation in env.obs_container.values():
                if observation.__class__.__name__ in joint_types_to_fetch:
                    j = env._obs_indices.concatenated_indices[observation.obs_ind]
                    joint_name = observation.name
                    joint_indices.append(j)
                    joint_names.append(joint_name)

                    angles = kinematics_history[:, j]
                    mean_angle = np.mean(np.abs(angles))
                    min_angle = np.min(angles)
                    max_angle = np.max(angles)
                    std_angle = np.std(angles)
                    rms_angle = np.sqrt(np.mean(angles ** 2))
                    
                    csv_stats.append({
                        "Actuator": joint_name,
                        "Mean_abs_angle": f"{mean_angle:.4f}",
                        "Min_angle": f"{min_angle:.4f}",
                        "Max_angle": f"{max_angle:.4f}",
                        "Std_Dev_angle": f"{std_angle:.4f}",
                        "RMS_angle": f"{rms_angle:.4f}"
                    })
            
            # Calculate overall kinematics statistics
            all_angles = kinematics_history.flatten()
            overall_mean_angle = np.mean(np.abs(all_angles))
            overall_min_angle = np.min(all_angles)
            overall_max_angle = np.max(all_angles)
            overall_rms_angle = np.sqrt(np.mean(all_angles ** 2))
            
            csv_stats.append({
                "Actuator": "OVERALL",
                "Mean_abs_angle": f"{overall_mean_angle:.4f}",
                "Min_angle": f"{overall_min_angle:.4f}",
                "Max_angle": f"{overall_max_angle:.4f}",
                "Std_Dev_angle": "N/A",
                "RMS_angle": f"{overall_rms_angle:.4f}"
            })

            with open(evaluation_stats_csv_file_path, mode='w', newline='') as f:
                fieldnames = ["Actuator", #"Mean_abs_Nm", "Max_abs_Nm", "Std_Dev_Nm", "RMS_Nm", 
                        "Mean_abs_angle", "Min_angle", "Max_angle", "Std_Dev_angle", "RMS_angle"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_stats)

            # print(kinematics_history.shape)
            # print(joint_indices)
            joint_indices = np.asarray(joint_indices).flatten()
            # print(joint_indices)
            # kinematics_history = np.squeeze(kinematics_history)
            # print(kinematics_history.shape)
                
            motion_df = pd.DataFrame(
                kinematics_history[:, joint_indices],
                columns=joint_names
            )

            motion_df.index.name = "Timestep"
            motion_df.to_csv(motion_csv_file_path)
        
        def process_kinetics(env, kinetics_history, evaluation_stats_csv_file_path, motion_csv_file_path):
            # Calculate kinetics statistics for each actuator and add to CSV rows
            # print("mujoco:", [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(env.model.nu)])
            # print("loco:", env._get_action_specification(None))

            torques_history = {}
            muscle_forces_history = {}

            for i in range(env.model.nu):
                name = mujoco.mj_id2name(
                    env.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    i
                )
                actions = kinetics_history[:, i]
                if name.startswith("mot_"):
                    torques_history[name] = actions * np.array(env.model.actuator_gear[i, 0])
                else:
                    muscle_forces_history[name] = actions

            for history in [torques_history, muscle_forces_history]:
                if len(history) != 0:
                    history_df = pd.DataFrame(history)
                    history_df.index.name = "Timestep"
                    history_df.to_csv(motion_csv_file_path.replace("motion", "torques" if history is torques_history else "muscle_forces"))
                
                    csv_stats = pd.DataFrame({
                        "Mean_abs_activation": history_df.abs().mean(),
                        "Min_activation": history_df.min(),
                        "Max_activation": history_df.max(),
                        "Std_Dev_activation": history_df.std(),
                        "RMS_activation": np.sqrt((history_df ** 2).mean())
                    })
                
                    # Calculate overall kinetics statistics
                    all_activations = history_df.to_numpy()
                    overall_mean_activation = np.mean(np.abs(all_activations))
                    overall_min_activation = np.min(all_activations)
                    overall_max_activation = np.max(all_activations)
                    overall_rms_activation = np.sqrt(np.mean(all_activations ** 2))
                    
                    csv_stats.loc["OVERALL"] = {
                        "Mean_abs_activation": f"{overall_mean_activation:.4f}",
                        "Min_activation": f"{overall_min_activation:.4f}",
                        "Max_activation": f"{overall_max_activation:.4f}",
                        "Std_Dev_activation": "N/A",
                        "RMS_activation": f"{overall_rms_activation:.4f}"
                    }
                    csv_stats.to_csv(evaluation_stats_csv_file_path.replace("motion", "torques" if history is torques_history else "muscle_forces"))

        def process_joint_kinetics(env, joint_kinetics_history, evaluation_stats_csv_file_path, motion_csv_file_path):
            # Calculate kinetics statistics for each actuator and add to CSV rows
            joint_columns = []
            for joint_id in range(env.model.njnt):
                joint_name = mujoco.mj_id2name(
                    env.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_id
                )
                joint_type = env.model.jnt_type[joint_id]
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    for k in range(6):
                        joint_columns.append(f"{joint_name}_{k}")
                else:
                    joint_columns.append(joint_name)

            motion_df = pd.DataFrame(
                joint_kinetics_history,
                columns=joint_columns
            )
            motion_df.index.name = "Timestep"
            motion_df.to_csv(motion_csv_file_path)
                
            csv_stats = pd.DataFrame({
                "Mean_abs_activation": motion_df.abs().mean(),
                "Min_activation": motion_df.min(),
                "Max_activation": motion_df.max(),
                "Std_Dev_activation": motion_df.std(),
                "RMS_activation": np.sqrt((motion_df ** 2).mean())
            })
            
            # Calculate overall kinetics statistics
            all_activations = joint_kinetics_history.flatten()
            overall_mean_activation = np.mean(np.abs(all_activations))
            overall_min_activation = np.min(all_activations)
            overall_max_activation = np.max(all_activations)
            overall_rms_activation = np.sqrt(np.mean(all_activations ** 2))
            
            csv_stats.loc["OVERALL"] = {
                "Mean_abs_activation": f"{overall_mean_activation:.4f}",
                "Min_activation": f"{overall_min_activation:.4f}",
                "Max_activation": f"{overall_max_activation:.4f}",
                "Std_Dev_activation": "N/A",
                "RMS_activation": f"{overall_rms_activation:.4f}"
            }
            csv_stats.to_csv(evaluation_stats_csv_file_path)

        # if deterministic:
        #     if "use_lattice" in config and config.use_lattice:
        #         train_state.params["mean_log_std"] = np.ones_like(train_state.params["mean_log_std"]) * -np.inf
        #         train_state.params["latent_log_std"] = np.ones_like(train_state.params["latent_log_std"]) * -np.inf
        #     else:
        #         train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                                  "train_state_seed for replay.")

            # take the seed queried for evaluation
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        # create env
        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        plcy_call = jax.jit(sample_actions)

        # reset env
        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys)

        # print("obs[..., current_state_idx][:15]", obs[..., current_state_idx][:15])
        # print("obs[..., reference_state_idx][:15]", obs[..., reference_state_idx][:15])
        # print("obs[..., reference_state_idx[current_state_idx]][:15]", obs[..., reference_state_idx[current_state_idx]][:15])

        if save_kinematics:
            kinematics_history_agent = np.zeros((n_steps + 1, len(env._obs_indices.concatenated_indices)), dtype=np.float32)
            kinematics_history_reference = np.zeros((n_steps + 1, len(env._obs_indices.concatenated_indices)), dtype=np.float32)
            actuators = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(env.model.nu)]
            kinetics_history_agent = np.zeros((n_steps, len(actuators)), dtype=np.float32)
            # kinetics_history_reference = np.zeros((n_steps, len(actuators)), dtype=np.float32)
            joint_kinetics_history_agent = np.zeros((n_steps, env.model.nv), dtype=np.float32)

            kinematics_history_agent[0] = obs[..., current_state_idx]
            # kinematics_history_reference[0] = obs[..., current_state_idx]
            kinematics_history_reference[0] = obs[..., reference_state_idx]

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        
        # @scan_tqdm(n_steps, print_rate=1, desc='Playing Motion')
        for i in range(n_steps):

            # SAMPLE ACTION
            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, _rng)
            action = jnp.atleast_2d(action)

            # STEP ENV
            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action)

            if save_kinematics:
                # Collect kinematics data
                kinematics_history_agent[i+1] = np.asarray(obs[..., current_state_idx])
                # if i < n_steps - 1:
                kinematics_history_reference[i+1] = np.asarray(obs[..., reference_state_idx])

                # Collect kinetics data
                kinetics_history_agent[i] = np.asarray(env_state.data.actuator_force)
                # kinetics_history_reference[i] = np.asarray(action)
                joint_kinetics_history_agent[i] = np.asarray(env_state.data.qfrc_actuator)

            # RENDER
            if use_mujoco:
                env.render(record=True)
            else:
                env.mjx_render(env_state, record=record)

            # RESET MUJOCO ENV (MJX resets by itself)
            if use_mujoco:
                if done:
                    obs = env.reset()

        env.stop()

        if save_kinematics:
            if not os.path.exists(save_kinematics_folder + "kinematics"):
                os.makedirs(save_kinematics_folder + "kinematics")
            kinematics_dir = save_kinematics_folder + "kinematics"
            process_kinematics(env, kinematics_history_agent, kinematics_dir + "/agent_eval_kinematics.csv", 
                               kinematics_dir + "/agent_motion_kinematics.csv")
            process_kinematics(env, kinematics_history_reference, kinematics_dir + "/reference_kinematics.csv",
                               kinematics_dir + "/reference_motion_kinematics.csv")
            process_kinetics(env, kinetics_history_agent, kinematics_dir + "/agent_eval_motion_kinetics.csv", 
                               kinematics_dir + "/agent_motion_kinetics.csv")
            # process_kinetics(env, kinetics_history_reference, kinematics_dir + "/reference_kinetics.csv",
            #                    kinematics_dir + "/reference_motion_kinetics.csv")
            process_joint_kinetics(env, joint_kinetics_history_agent, kinematics_dir + "/agent_eval_joint_kinetics.csv", 
                               kinematics_dir + "/agent_motion_joint_kinetics.csv")
            
    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf: PPOAgentConf,
                           agent_state: PPOAgentState,
                           n_steps=None, render=True,
                           record=False, rng=None, deterministic=False,
                           train_state_seed=None):

        cls.play_policy(env, agent_conf, agent_state, 1, n_steps, render, record, rng, deterministic,
                        True, False, train_state_seed)

    @staticmethod
    def _wrap_env(env, config):

        if "len_obs_history" in config and config.len_obs_history > 1:
            env = NStepWrapper(env, config.len_obs_history)
        env = LogWrapper(env)
        env = VecEnv(env)
        if config.normalize_env:
            env = NormalizeVecReward(env, config.gamma)
        return env
