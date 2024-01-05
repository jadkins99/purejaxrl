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
import flashbax as fbx
from jax.nn.initializers import variance_scaling
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
from rlax import transform_to_2hot
from pathlib import Path

from jax import config
import pickle

# config.update("jax_disable_jit", True)
# config.update("jax_debug_nans", True)


class ActorTrainState(TrainState):
    advn_stats: dict[float, float]


class Critic(nn.Module):
    activation: str = "tanh"
    num_bins: int = None
    bin_min_value: int = -20
    bin_max_value: int = 20

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu

        elif self.activation == "silu":
            activation = nn.silu

        else:
            activation = nn.tanh

        critic = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        if self.num_bins is None:
            value = nn.Dense(1, kernel_init=constant(0.0), bias_init=constant(0.0))(
                critic
            )
            return jnp.squeeze(value, axis=-1), 1.0

        else:
            bin_logits = nn.Dense(
                self.num_bins, kernel_init=constant(0.0), bias_init=constant(0.0)
            )(critic)
            probs = jax.nn.softmax(bin_logits, axis=-1)
            possible_outcomes = jnp.linspace(
                self.bin_min_value,
                self.bin_max_value,
                self.num_bins,
            )
            value = jnp.sum(probs * possible_outcomes, axis=-1)
            return value, bin_logits


class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "silu":
            activation = nn.silu
        else:
            activation = nn.tanh
        actor_rep = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_rep = activation(actor_rep)
        actor_rep = nn.Dense(
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_rep)
        actor_rep = activation(actor_rep)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_rep)
        actor_logtstd = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_rep)
        # actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

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
    # config["MINIBATCH_SIZE"] = (
    #     config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    # )
    if config["ENV_NAME"] == "Pendulum-v1":
        env, env_params = gymnax.make(config["ENV_NAME"])

    else:
        env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        # env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = Actor(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        critic_network = Critic(
            activation=config["ACTIVATION"],
            num_bins=config["NUM_BINS"],
            bin_min_value=config["BIN_MIN_VALUE"],
            bin_max_value=config["BIN_MAX_VALUE"],
        )

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
                optax.adam(config["ACTOR_LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["CRITIC_LR"], eps=1e-5),
            )
        actor_train_state = ActorTrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
            advn_stats={},
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        def symlog(x):
            return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

        def symexp(x):
            return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # COLLECT TRAJECTORIES
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
            value, logits = critic_network.apply(critic_train_state.params, last_obs)

            if config["SYMLOG_CRITIC_TARGETS"]:
                value = symexp(value)

            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )
            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_prob=log_prob,
                obs=obsv,
                last_obs=last_obs,
                info=info,
            )
            runner_state = (
                actor_train_state,
                critic_train_state,
                env_state,
                obsv,
                rng,
            )
            return runner_state, transition

        # initialize replay buffer
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, 1)
        d_obs, d_env_state = env.reset(reset_rng, env_params)
        pi = actor_network.apply(actor_train_state.params, d_obs)
        d_action = pi.sample(seed=rng)
        d_log_prob = pi.log_prob(d_action)
        d_value, logits = critic_network.apply(critic_train_state.params, d_obs)
        rng, _rng = jax.random.split(rng)
        step_rng = jax.random.split(_rng, 1)
        d_last_obs, d_env_state, d_reward, d_done, d_info = env.step(
            step_rng, d_env_state, d_action, env_params
        )
        # there are some weird shape issues going on with buffer. This is hacky but it works
        for key in d_info.keys():
            d_info[key] = d_info[key][0]
        dummy_transition = Transition(
            done=d_done[0],
            action=d_action[0],
            value=d_value[0],
            reward=d_reward[0],
            log_prob=d_log_prob[0],
            obs=d_obs[0],
            last_obs=d_last_obs[0],
            info=d_info,
        )
        replay_buffer = fbx.make_trajectory_buffer(
            min_length_time_axis=config["BUFFER_MIN_LENGTH_TIME_AXIS"],
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            sample_batch_size=config["BATCH_SIZE"] * config["NUM_MINIBATCHES"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=config["BATCH_LENGTH"],
            period=1,
        )
        buffer_state = replay_buffer.init(dummy_transition)
        runner_state = (actor_train_state, critic_train_state, env_state, obsv, rng)
        runner_state, traj_batch = jax.lax.scan(
            _env_step,
            runner_state,
            None,
            config["BUFFER_MIN_LENGTH_TIME_AXIS"],
        )
        swap_axis = lambda arr: jnp.moveaxis(arr, 0, 1)
        traj_batch_buffer = jax.tree_util.tree_map(swap_axis, traj_batch)
        buffer_state = replay_buffer.add(buffer_state, traj_batch_buffer)

        last_obs = traj_batch.last_obs[-1, ...]
        last_val, logits = critic_network.apply(critic_train_state.params, last_obs)
        if config["SYMLOG_CRITIC_TARGETS"]:
            last_val = symexp(last_val)

        advantages, _ = _calculate_gae(traj_batch, last_val)
        advn_per_95 = jnp.percentile(advantages, 95)
        advn_per_5 = jnp.percentile(advantages, 5)
        actor_train_state.advn_stats["advn_per_5"] = advn_per_5
        actor_train_state.advn_stats["advn_per_95"] = advn_per_95

        # TRAIN LOOP
        def _update_step(input_runner_state, unused):
            runner_state = input_runner_state[:-1]
            buffer_state = input_runner_state[-1]

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            traj_batch_buffer = jax.tree_util.tree_map(swap_axis, traj_batch)
            buffer_state = replay_buffer.add(buffer_state, traj_batch_buffer)

            (
                actor_train_state,
                critic_train_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state

            # SAMPLE FROM BUFFER
            rng, _rng = jax.random.split(rng)
            sampled_batch = replay_buffer.sample(buffer_state, _rng)
            sampled_batch = jax.tree_util.tree_map(swap_axis, sampled_batch)
            last_obs_buffer = sampled_batch.experience.last_obs[-1, ...]
            last_val_buffer, logits = critic_network.apply(
                critic_train_state.params, last_obs_buffer
            )

            last_val, logits = critic_network.apply(critic_train_state.params, last_obs)

            if config["SYMLOG_CRITIC_TARGETS"]:
                last_val_buffer = symexp(last_val_buffer)
                last_val = symexp(last_val)

            # CALCULATE ADVANTAGE

            advantages, targets = _calculate_gae(
                sampled_batch.experience, last_val_buffer
            )
            advantages_recent, _ = _calculate_gae(traj_batch, last_val)
            # COMPUTE ADVANTAGE STATISTICS
            (
                config["ADVANTAGE_EMA_RATE"] * jnp.percentile(advantages_recent, 5)
                + (1 - config["ADVANTAGE_EMA_RATE"])
                * actor_train_state.advn_stats["advn_per_5"]
            )
            (
                config["ADVANTAGE_EMA_RATE"] * jnp.percentile(advantages_recent, 95)
                + (1 - config["ADVANTAGE_EMA_RATE"])
                * actor_train_state.advn_stats["advn_per_95"]
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    actor_train_state, critic_train_state = train_state

                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        if config["SYMLOG_CRITIC_TARGETS"]:
                            targets = symlog(targets)
                        # RERUN NETWORK
                        value, logits = critic_network.apply(
                            critic_params, traj_batch.obs
                        )

                        # CALCULATE VALUE LOSS
                        # value_pred_clipped = traj_batch.value + (
                        #     value - traj_batch.value
                        # ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])

                        if config["NUM_BINS"] is None:
                            value_losses = jnp.square(
                                value - jax.lax.stop_gradient(targets)
                            )

                        else:
                            critic_target_2hot = transform_to_2hot(
                                targets,
                                min_value=config["BIN_MIN_VALUE"],
                                max_value=config["BIN_MAX_VALUE"],
                                num_bins=config["NUM_BINS"],
                            )
                            value_losses = -1 * jnp.sum(
                                jnp.multiply(logits, critic_target_2hot), axis=-1
                            )
                        # sum_value_loss_two_hot_B = jnp.sum(
                        #     value_loss_two_hot_B_H, axis=1
                        # )

                        # value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        # value_loss = (
                        #     0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        # )
                        value_loss = value_losses.mean()
                        total_loss = value_loss

                        return total_loss, (value_loss,)

                    def _actor_loss_fn(actor_params, traj_batch, gae, advn_stats):
                        # RERUN NETWORK
                        pi = actor_network.apply(actor_params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        # ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        if config["ADVN_NORM"] == "MEAN":
                            gae = (gae - gae.mean()) / (gae.std() + 1e-6)

                        elif config["ADVN_NORM"] == "EMA":
                            gae = gae / (
                                advn_stats["advn_per_95"]
                                - advn_stats["advn_per_5"]
                                + 1e-6
                            )

                        elif config["ADVN_NORM"] == "MAX_EMA":
                            gae = gae / (
                                jnp.maximum(
                                    1,
                                    advn_stats["advn_per_95"]
                                    - advn_stats["advn_per_5"],
                                )
                            )
                        # loss_actor1 = ratio * gae
                        # loss_actor2 = (
                        #     jnp.clip(
                        #         ratio,
                        #         1.0 - config["CLIP_EPS"],
                        #         1.0 + config["CLIP_EPS"],
                        #     )
                        #     * gae
                        # )
                        loss_actor = log_prob * jax.lax.stop_gradient(gae)
                        # loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor - config["ENT_COEF"] * entropy

                        return total_loss, (loss_actor, entropy)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)

                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, traj_batch, targets
                    )
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        traj_batch,
                        advantages,
                        actor_train_state.advn_stats,
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

                (
                    actor_train_state,
                    critic_train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                # batch_size = config["BATCH_SIZE"] * config["NUM_MINIBATCHES"]
                # assert (
                #     batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                # ), "batch size must be equal to number of steps * number of envs"
                # permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                # batch = jax.tree_util.tree_map(
                #     lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                # )
                # shuffled_batch = jax.tree_util.tree_map(
                #     lambda x: jnp.take(x, permutation, axis=0), batch
                # )

                # minibatches = jax.tree_util.tree_map(
                #     lambda x: jnp.reshape(
                #         x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                #     ),
                #     batch,
                # )
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
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                actor_train_state,
                critic_train_state,
                sampled_batch.experience,
                advantages,
                targets,
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
    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--alg_type",
        action="store",
        default="lambda_ac",
    )
    parser.add_argument(
        "--env_name",
        action="store",
        default="hopper",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    config = {
        "ACTOR_LR": 3e-4,
        "CRITIC_LR": 3e-4,
        "NUM_ENVS": 8,
        "NUM_STEPS": 1,  # T = num_steps*num_envs
        "TOTAL_TIMESTEPS": 5e5,  # Z in pseudocode
        "UPDATE_EPOCHS": 1,  # E in pseudocode
        "NUM_MINIBATCHES": 1,  # M in pseudocode
        "ADVANTAGE_EMA_RATE": 0.02,
        "ADVN_NORM": "OFF",
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "ENT_COEF": 0.1,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": args.env_name,
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": False,
        "DEBUG": True,
        "BUFFER_MIN_LENGTH_TIME_AXIS": 100,
        "BUFFER_SIZE": 1000000,
        "BATCH_SIZE": 32,  # N = C/batch_size in psueudocode
        "BATCH_LENGTH": 20,
        "SYMLOG_CRITIC_TARGETS": False,
        "NUM_BINS": None,
        "BIN_MIN_VALUE": -20,
        "BIN_MAX_VALUE": 20,
        "SWEEP": False,
    }

    if args.alg_type == "advn_norm_ema":
        config["ADVN_NORM"] = "EMA"
    elif args.alg_type == "advn_norm_max_ema":
        config["ADVN_NORM"] = "MAX_EMA"
    elif args.alg_type == "advn_norm_mean":
        config["ADVN_NORM"] = "MEAN"
    elif args.alg_type == "symlog_critic_targets":
        config["SYMLOG_CRITIC_TARGETS"] = True
    elif args.alg_type == "discrete_critic":
        config["SYMLOG_CRITIC_TARGETS"] = True
        config["NUM_BINS"] = 255

    rng = jax.random.PRNGKey(30)
    num_seeds = 10

    import sqlite3

    if config["SWEEP"]:
        dbase = sqlite3.connect("SweepDatabase.db")
        q = dbase.execute("SELECT * FROM sweep_runs WHERE mean_episode_return is NULL")
        qs = q.fetchall()

        # we only want to run jobs of varying hyperparam config with fixed env and alg type
        filtered_qs = list(
            filter(
                lambda setting: setting[0] == args.env_name
                and setting[1] == args.alg_type,
                qs,
            )
        )

        sweep_vals = list(map(lambda setting: setting[2:6], filtered_qs))
        sweep_vals = jnp.asarray(sweep_vals)

        rngs = jax.random.split(rng, num_seeds)

        train_vjit = jax.jit(
            jax.vmap(jax.vmap(make_train(config), in_axes=(0, None)), in_axes=(None, 0))
        )
        outs = train_vjit(rngs, sweep_vals)

        path_str = (
            f"/Users/jadkins/purejaxrl/jax_rl_logs/{args.alg_type}/{args.env_name}"
        )
        path_str = f"/home/jadkins/scratch/purejaxrl/jax_rl_logs/{args.alg_type}/{args.env_name}"

        Path(path_str).mkdir(parents=True, exist_ok=True)

        with open(
            f"{path_str}/GAE_LAMBDA={sweep_vals[:][0]}_CRITIC_LR={sweep_vals[:][1]}_ACTOR_LR={sweep_vals[:][2]}_ENT_COEF={sweep_vals[:][3]}RNG={start_seed},{num_seeds}.pkl",
            "wb",
        ) as f:
            pickle.dump(outs["metrics"], f)

        for sweep_params in range(sweep_vals.shape[0]):
            for seed_idx in range(0, num_seeds):
                completed_episodes = outs["metrics"]["returned_episode"][sweep_params][
                    seed_idx
                ]
                returns_completed_episodes = outs["metrics"][
                    "returned_episode_returns"
                ][sweep_params][seed_idx][completed_episodes == True]
                returned_episode_mean = returns_completed_episodes.mean()
                dbase.execute(
                    f"INSERT INTO sweep_runs(env_name,alg_type,gae_lambda,critic_lr,actor_lr,ent_coef, seed_idx, T, mean_episode_return) VALUES('{args.env_name}', '{args.alg_type}',{sweep_vals[sweep_params][0]},{sweep_vals[sweep_params][1]},{sweep_vals[sweep_params][2]},{sweep_vals[sweep_params][3]},{seed_idx},{config['NUM_STEPS']*config['NUM_ENVS']},{returned_episode_mean});"
                )
        dbase.commit()

        dbase.close()

    else:
        train_jit = jax.jit(make_train(config))

        out = train_jit(rng)
        import matplotlib.pyplot as plt

        plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
        plt.xlabel("Update Step")
        plt.ylabel("Return")
        plt.show()
