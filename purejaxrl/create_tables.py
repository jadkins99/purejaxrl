import sqlite3

dbase = sqlite3.connect("/home/jadkins/scratch/SweepDatabase.db")
import jax.numpy as jnp
import jax
import itertools


gae_lambdas = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
critic_lrs = jnp.array([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])

# actor lr is k x critic_lr
actor_lrs = jnp.array([1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1])
ent_coefs = jnp.array([1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1])
env_names = ["hopper","inverted_pendulum","reacher"]
alg_types = ["lambda_ac","advn_norm_ema","advn_norm_max_ema","advn_norm_mean","symlog_critic_targets","discrete_critic"]


dbase.execute("""CREATE TABLE IF NOT EXISTS sweep_runs ( env_name TEXT,
              alg_type TEXT, gae_lambda REAL,  critic_lr REAL, actor_lr REAL,
              ent_coef REAL, T INTEGER, seed_idx INTEGER, mean_episode_return
              REAL);"""
)
dbase.commit()

sweep_vals = jnp.asarray(
    list(itertools.product(gae_lambdas, critic_lrs, actor_lrs, ent_coefs))
)


sweep_vals = jnp.asarray(
    list(itertools.product(gae_lambdas, critic_lrs, actor_lrs, ent_coefs))
)


for alg in alg_types:
    for env_name in env_names:
        for sweep_params in range(sweep_vals.shape[0]):
            dbase.execute(
                f"""
                    
                INSERT INTO sweep_runs(env_name,alg_type,gae_lambda,critic_lr,actor_lr,ent_coef) VALUES('{env_name}', '{alg}',{sweep_vals[sweep_params][0]},{sweep_vals[sweep_params][1]},{sweep_vals[sweep_params][2]},{sweep_vals[sweep_params][3]});
                
                """
            )

dbase.commit()
dbase.close()
