import sys
import pytest

sys.path.insert(1, "../purejaxrl")
from purejaxrl.ppo_continuous_action import make_train
import jax
import time


def test_hopper():
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "hopper",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
    }
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    import time

    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    t1 = time.time() - t0
    assert t1 < 70  # make sure it runs fast
    print(out["metrics"]["returned_episode_returns"].mean())
    assert (
        out["metrics"]["returned_episode_returns"].mean() > 10.0
    )  # make sure it learns
