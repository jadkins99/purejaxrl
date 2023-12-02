import sys
import pytest

sys.path.insert(1, "../purejaxrl")
from purejaxrl.ppo import make_train
import jax
import time


def test_cartpole():
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": False,
    }
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    t1 = time.time() - t0
    assert t1 < 10  # make sure it runs fast
    assert (
        out["metrics"]["returned_episode_returns"].mean() > 350.0
    )  # make sure it learns


def test_acrobot():
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "Acrobot-v1",
        "ANNEAL_LR": False,
    }
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    t1 = time.time() - t0
    assert t1 < 10  # make sure it runs fast
    assert (
        out["metrics"]["returned_episode_returns"].mean() > -150.0
    )  # make sure it learns
