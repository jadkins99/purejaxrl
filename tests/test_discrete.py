import sys
import pytest

sys.path.insert(1, "../purejaxrl")
from purejaxrl.online_learning import make_train
import jax
import time


def test_cartpole():
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,  # T = num_steps*num_envs
        "TOTAL_TIMESTEPS": 5e5,  # Z in pseudocode
        "UPDATE_EPOCHS": 4,  # E in pseudocode
        "NUM_MINIBATCHES": 4,  # M in pseudocode
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": False,
        "DEBUG": True,
        "BUFFER_SIZE": 512,  # C in psuedocode
        "MIN_BUFFER_SIZE": 512,
        "BATCH_SIZE": 4,  # N = C/batch_size in psueudocode
        "BATCH_LENGTH": 128,
    }
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    t1 = time.time() - t0
    assert t1 < 22  # make sure it runs fast
    assert (
        out["metrics"]["returned_episode_returns"].mean() > 350.0
    )  # make sure it learns


def test_acrobot():
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,  # T = num_steps*num_envs
        "TOTAL_TIMESTEPS": 5e5,  # Z in pseudocode
        "UPDATE_EPOCHS": 4,  # E in pseudocode
        "NUM_MINIBATCHES": 4,  # M in pseudocode
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "Acrobot-v1",
        "ANNEAL_LR": False,
        "DEBUG": True,
        "BUFFER_SIZE": 512,  # C in psuedocode
        "MIN_BUFFER_SIZE": 512,
        "BATCH_SIZE": 4,  # N = C/batch_size in psueudocode
        "BATCH_LENGTH": 128,
    }
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    t1 = time.time() - t0
    assert t1 < 25  # make sure it runs fast
    assert (
        out["metrics"]["returned_episode_returns"].mean() > -150.0
    )  # make sure it learns
