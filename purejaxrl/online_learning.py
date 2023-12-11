import jax
import time
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import os
import gymnax
from wrappers import LogWrapper, FlattenObservationWrapper
import flashbax as fbx
import pickle
from jax import config


# config.update("jax_disable_jit", True)
class Critic(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return jnp.squeeze(critic, axis=-1)


class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    last_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, ENT_COEF, ACTOR_LR, CRITIC_LR, GAE_LAMBDA):
        # INIT NETWORK
        actor_network = Actor(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        critic_network = Critic(activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        actor_network_params = actor_network.init(_rng, init_x)
        rng, _rng = jax.random.split(rng)
        critic_network_params = critic_network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(ACTOR_LR, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(CRITIC_LR, eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # initialize replay buffer
        d_obs, d_env_state = env.reset(rng, env_params)
        pi = actor_network.apply(actor_train_state.params, d_obs)
        d_action = pi.sample(seed=rng)
        d_log_prob = pi.log_prob(d_action)
        d_value = critic_network.apply(critic_train_state.params, d_obs)
        d_last_obs, d_env_state, d_reward, d_done, d_info = env.step(
            rng, d_env_state, d_action, env_params
        )
        dummy_transition = Transition(
            done=d_done,
            action=d_action,
            value=d_value,
            reward=d_reward,
            log_prob=d_log_prob,
            obs=d_obs,
            last_obs=d_last_obs,
            info=d_info,
        )
        replay_buffer = fbx.make_trajectory_buffer(
            min_length_time_axis=config["MIN_BUFFER_SIZE"] // config["NUM_ENVS"],
            max_size=config["BUFFER_SIZE"],
            sample_batch_size=config["BATCH_SIZE"] * config["NUM_MINIBATCHES"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=config["BATCH_LENGTH"],
            period=1,
        )
        buffer_state = replay_buffer.init(dummy_transition)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _env_step(runner_state, unused):
            (
                actor_train_state,
                critic_train_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi = actor_network.apply(actor_train_state.params, last_obs)
            value = critic_network.apply(critic_train_state.params, last_obs)

            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, action, env_params)
            transition = Transition(
                done, action, value, reward, log_prob, last_obs, obsv, info
            )
            runner_state = (
                actor_train_state,
                critic_train_state,
                env_state,
                obsv,
                rng,
            )
            return runner_state, transition

        runner_state = (actor_train_state, critic_train_state, env_state, obsv, rng)
        runner_state, traj_batch = jax.lax.scan(
            _env_step,
            runner_state,
            None,
            config["MIN_BUFFER_SIZE"] // config["NUM_ENVS"],
        )
        swap_axis = lambda arr: jnp.moveaxis(arr, 0, 1)
        traj_batch_buffer = jax.tree_util.tree_map(swap_axis, traj_batch)
        buffer_state = replay_buffer.add(buffer_state, traj_batch_buffer)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            env_runner_state = runner_state[:-1]
            buffer_state = runner_state[-1]
            env_runner_state, traj_batch = jax.lax.scan(
                _env_step, env_runner_state, None, config["NUM_STEPS"]
            )
            traj_batch_buffer = jax.tree_util.tree_map(swap_axis, traj_batch)
            buffer_state = replay_buffer.add(buffer_state, traj_batch_buffer)

            (
                actor_train_state,
                critic_train_state,
                env_state,
                last_obs,
                rng,
            ) = env_runner_state

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                (
                    actor_train_state,
                    critic_train_state,
                    buffer_state,
                    rng,
                ) = update_state

                # SAMPLE FROM BUFFER
                rng, _rng = jax.random.split(rng)
                sampled_batch = replay_buffer.sample(buffer_state, _rng)
                sampled_batch = jax.tree_util.tree_map(swap_axis, sampled_batch)

                last_obs_buffer = sampled_batch.experience.last_obs[-1, ...]
                last_val_buffer = critic_network.apply(
                    critic_train_state.params, last_obs_buffer
                )

                # CALCULATE ADVANTAGE
                def _calculate_gae(traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.done,
                            transition.value,
                            transition.reward,
                        )
                        delta = (
                            reward + config["GAMMA"] * next_value * (1 - done) - value
                        )
                        gae = delta + config["GAMMA"] * GAE_LAMBDA * (1 - done) * gae
                        return (gae, value), gae

                    _, advantages = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )
                    return advantages, advantages + traj_batch.value

                advantages, targets = _calculate_gae(
                    sampled_batch.experience, last_val_buffer
                )

                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    actor_train_state, critic_train_state = train_state

                    def _critic_loss_fn(critic_params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        value = critic_network.apply(critic_params, traj_batch.obs)

                        # CALCULATE VALUE LOSS
                        # value_pred_clipped = traj_batch.value + (
                        #     value - traj_batch.value
                        # ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(
                            value - jax.lax.stop_gradient(targets)
                        )
                        # value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        # value_loss = (
                        #     0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        # )
                        value_loss = value_losses.mean()

                        total_loss = value_loss

                        return total_loss, (value_loss,)

                    def _actor_loss_fn(actor_params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi = actor_network.apply(actor_params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = log_prob * jax.lax.stop_gradient(gae)
                        # loss_actor2 = (
                        #     jnp.clip(
                        #         ratio,
                        #         1.0 - config["CLIP_EPS"],
                        #         1.0 + config["CLIP_EPS"],
                        #     )
                        #     * gae
                        # )
                        # loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = -loss_actor1
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor - ENT_COEF * entropy

                        return total_loss, (loss_actor, entropy)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)

                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, traj_batch, advantages, targets
                    )
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, traj_batch, advantages, targets
                    )
                    total_loss = actor_loss + critic_loss

                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )

                    train_state = (actor_train_state, critic_train_state)
                    return train_state, total_loss

                rng, _rng = jax.random.split(rng)

                batch = (sampled_batch.experience, advantages, targets)

                batch = jax.tree_util.tree_map(swap_axis, batch)

                resize_minibatches = lambda x: jnp.reshape(
                    x,
                    [config["NUM_MINIBATCHES"], config["BATCH_SIZE"]]
                    + list(x.shape[1:]),
                )
                minibatches = jax.tree_util.tree_map(resize_minibatches, batch)

                train_state = (actor_train_state, critic_train_state)
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                actor_train_state, critic_train_state = train_state
                update_state = (
                    actor_train_state,
                    critic_train_state,
                    buffer_state,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                actor_train_state,
                critic_train_state,
                buffer_state,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            actor_train_state = update_state[0]
            critic_train_state = update_state[1]

            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (
                actor_train_state,
                critic_train_state,
                env_state,
                last_obs,
                rng,
                buffer_state,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            actor_train_state,
            critic_train_state,
            env_state,
            obsv,
            _rng,
            buffer_state,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = {
        # "ACTOR_LR": 2.5e-4,
        # "CRITIC_LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 1,  # T = num_steps*num_envs
        "TOTAL_TIMESTEPS": 5e5,  # Z in pseudocode
        "UPDATE_EPOCHS": 1,  # E in pseudocode
        "NUM_MINIBATCHES": 1,  # M in pseudocode
        "GAMMA": 0.99,
        # "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "Acrobot-v1",
        "ANNEAL_LR": False,
        "DEBUG": True,
        "BUFFER_SIZE": 100000,  # C in psuedocode
        "MIN_BUFFER_SIZE": 100,
        "BATCH_SIZE": 32,  # N = C/batch_size in psueudocode
        "BATCH_LENGTH": 15,
    }
    rng = jax.random.PRNGKey(42)
    num_seeds = 10
    # gae_lambdas = jnp.array([0.1, 0.5, 0.7, 0.9, 1.0])
    # critic_lrs = jnp.array([10.0e-5, 10.0e-4, 10.0e-3, 10.0e-2, 10.0e-1])
    # actor_lrs = jnp.array([10.0e-5, 10.0e-4, 10.0e-3, 10.0e-2, 10.0e-1])
    # ent_coefs = jnp.array([10.0e-3, 10.0e-2, 10.0e-1, 10.0e0, 10.0e1])
    gae_lambdas = jnp.array([0.95])

    critic_lrs = jnp.array([2.5e-4])
    actor_lrs = jnp.array([2.5e-4])
    ent_coefs = jnp.array([10.0])

    rngs = jax.random.split(rng, num_seeds)

    # train_vjit = jax.jit(jax.vmap(make_train(config)))

    train_vvjit = jax.jit(
        jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(
                        jax.vmap(
                            make_train(config),
                            in_axes=(0, None, None, None, None),
                        ),
                        in_axes=(None, 0, None, None, None),
                    ),
                    in_axes=(None, None, 0, None, None),
                ),
                in_axes=(None, None, None, 0, None),
            ),
            in_axes=(None, None, None, None, 0),
        ),
    )

    t0 = time.time()
    # outs is indexed in reverse  order that the args are placed
    # i.e. outs.shape = (gae_lambdas, critic_lrs, actor_lrs, ent_coefs, rngs, NUM_UPDATES, 1, NUM_ENVS)
    outs = jax.block_until_ready(
        train_vvjit(rngs, ent_coefs, actor_lrs, critic_lrs, gae_lambdas)
    )
    print(f"time: {time.time() - t0:.2f} s")

    with open("/home/jadkins/scratch/lambda_ac.pkl", "wb") as f:
        pickle.dump(outs["metrics"], f)

    # import matplotlib.pyplot as plt

    # plt.plot(outs["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
    # plt.xlabel("Update Step")
    # for gae_lambdas_idx in range(gae_lambdas.size):
    #     for critic_lrs_idx in range(critic_lrs.size):
    #         for actor_lrs_idx in range(actor_lrs.size):
    #             for ent_coefs_idx in range(ent_coefs.size):
    #                 for seed in range(num_seeds):
    #                     competed_episode_lengths = outs["metrics"][
    #                         "returned_episode_lengths"
    #                     ][
    #                         gae_lambdas_idx,
    #                         critic_lrs_idx,
    #                         actor_lrs_idx,
    #                         ent_coefs_idx,
    #                         seed,
    #                         ...,
    #                     ][
    #                         outs["metrics"]["returned_episode"][
    #                             gae_lambdas_idx,
    #                             critic_lrs_idx,
    #                             actor_lrs_idx,
    #                             ent_coefs_idx,
    #                             seed,
    #                             ...,
    #                         ]
    #                         == True
    #                     ]
    #                     experience = jnp.cumsum(competed_episode_lengths)
    #                     returns = outs["metrics"]["returned_episode_returns"][
    #                         gae_lambdas_idx,
    #                         critic_lrs_idx,
    #                         actor_lrs_idx,
    #                         ent_coefs_idx,
    #                         seed,
    #                         ...,
    #                     ][
    #                         outs["metrics"]["returned_episode"][
    #                             gae_lambdas_idx,
    #                             critic_lrs_idx,
    #                             actor_lrs_idx,
    #                             ent_coefs_idx,
    #                             seed,
    #                             ...,
    #                         ]
    #                         == True
    #                     ]

    #                     # plt.plot(outs["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1))
    #                     plt.plot(experience, returns)

    # plt.xlabel("Steps")
    # plt.ylabel("Return")
    # plt.show()
