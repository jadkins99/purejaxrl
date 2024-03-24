import matplotlib.pyplot as plt
import gc
import jax.numpy as jnp
import os
import pandas as pd
import seaborn as sns
from boostrap_code import normalize_scores
import numpy as np

change_labels = {
    "lambda_ac": "PPO",
    "symlog_obs": "PPO symlog($o_t$)",
    "norm_obs": "PPO $(o_t-\mu_{o_t})/\sigma_{o_t}$",
    "advn_norm_ema": "PPO $A_t/(perc(A_t,95) - perc(A_t,5))$",
    "advn_norm_max_ema": "PPO $A_t/(max(1,perc(A_t,95) - perc(A_t,5)))$",
    "advn_norm_mean": "PPO $(A_t - \mu_{A_t})/\sigma_{A_t}$",
    "symlog_critic_targets": "PPO $symlog(G_{t:k+L}^\lambda)$",
}


def generate_shaded_dummy_sensitivty_plot():
    ref_sens = 0.3
    ref_perf = 0.6757187532709882
    unit_line = lambda x: ref_perf + 1.0 * (x - ref_sens)
    xs = np.linspace(0, 0.6, 1000)
    alpha = 0.1
    colors = {
        "lambda_ac": "tab:blue",
        "symlog_critic_targets": "tab:orange",
        "advn_norm_ema": "tab:green",
        "advn_norm_max_ema": "tab:olive",
        "advn_norm_mean": "tab:purple",
        "symlog_obs": "tab:brown",
        "norm_obs": "tab:cyan",
        # "tab:red",
        # "tab:gray",
    }
    fig, ax = plt.subplots(layout="constrained")
    fig.supxlabel(
        "Sensitivity",
        # $\mathbb{E}_{e \in E}[max_{\eta \in H^a}[s(a,e,\eta)]] - max_{\eta \in H^a} [\mathbb{E}_{e \in E}[S(a,e,\eta)]]$"
    )
    fig.supylabel(
        "Performance",
        # \n $\mathbb{E}_{e \in E}[max_{\eta \in H^a}[s(a,e,\eta)]]$",
        rotation="horizontal",
    )
    ax.set_title("Cross Environment Sensitivity")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 0.6)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    # ax.axhline(ref_perf, color="blue", linestyle="--")
    # ax.axvline(ref_sens, color="blue", linestyle="--")
    # ax.plot(xs, unit_line(xs), color="blue", linestyle="--")

    ax.fill_between(
        xs[xs >= ref_sens],
        unit_line(xs[xs > ref_sens]),
        1.0,
        color="yellow",
        alpha=alpha,
    )
    ax.fill_between(
        xs[xs <= ref_sens],
        unit_line(xs[xs <= ref_sens]),
        ref_perf,
        color="blue",
        alpha=alpha,
    )

    ax.fill_between(
        xs[xs <= ref_sens],
        ref_perf,
        1.0,
        color="green",
        alpha=alpha,
    )
    ax.fill_between(
        xs,
        0.0,
        unit_line(xs),
        color="red",
        alpha=alpha,
    )
    # plt.plot(ref_sens, ref_perf, "bo")

    plt.savefig("cartoon_plot.svg")
    plt.show()


def generate_lcs():

    alg_list = [
        "lambda_ac",
        "advn_norm_ema",
        "advn_norm_max_ema",
        "advn_norm_mean",
        "symlog_critic_targets",
        "symlog_obs",
        "norm_obs",
    ]
    # env_list = ["ant", "swimmer", "walker2d", "hopper", "halfcheetah"]
    env_list = ["swimmer", "halfcheetah"]

    fig, axs = plt.subplots(7, 5)
    fig.supxlabel("Steps")
    fig.supylabel("Returns")
    t_95_100 = 1.984
    for i, alg in enumerate(alg_list):
        for j, env in enumerate(env_list):
            path = os.path.join("ppo_brax", alg, env, "returns.npy")
            print(i, j)
            returns = jnp.load(path)
            std = returns.mean(-1).std(0).reshape(-1)
            mean = returns.mean(-1).mean(0).reshape(-1)
            sqrt_n = jnp.sqrt(returns.shape[0])
            num_updates = 3000000 // len(mean)
            step_arr = num_updates * jnp.arange(len(mean))
            axs[i, j].plot(step_arr, mean, label=change_labels[alg])
            axs[i, j].fill_between(
                x=step_arr,
                y1=mean - t_95_100 * std / sqrt_n,
                y2=mean + t_95_100 * std / sqrt_n,
                alpha=0.2,
                lw=0,
            )
            axs[i, j].grid(False)
            axs[i, j].set_xticks([0.0, 3000000])
            if j == 0:
                axs[i, j].set_ylabel(
                    change_labels[alg], rotation="horizontal"
                )  # ,fontdict={'size':6})
            if i == 0:
                axs[i, j].set_title(env, rotation="horizontal")
            # axs[i,j].legend()
            del path
            del returns
            del mean
            del std
            del num_updates
            del step_arr
            del sqrt_n

            plt.savefig("learning_curves.svg")
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(18.5, 10.5)
            gc.collect()

    fig.tight_layout()


def generate_performance_bar_plot():
    env_list = ["ant", "swimmer", "walker2d", "hopper", "halfcheetah"]
    # env_list = ["halfcheetah", "swimmer"]
    csv_path = "/Users/jadkins/ppo_brax_logged_means.csv"

    df = pd.read_csv(csv_path)
    # normalized_df = normalize_scores(df, env_list)

    df_normed = normalize_scores(df, env_list)
    alg_list = [
        "lambda_ac",
        "advn_norm_ema",
        "advn_norm_max_ema",
        "advn_norm_mean",
        "symlog_critic_targets",
        "symlog_obs",
        "norm_obs",
    ]
    fig, ax = plt.subplots(layout="constrained")

    score_algs = []
    for alg in alg_list:
        score = sum(
            [
                df_normed.loc[df_normed["alg_type"] == alg][
                    f"{env} normalized score"
                ].max()
                for env in env_list
            ]
        ) / len(env_list)
        score_algs.append((change_labels[alg], score))

    score_algs.sort(key=lambda x: x[1])

    for score_alg in score_algs:

        ax.bar(score_alg[0], score_alg[1], color="blue")

    ax.set_ylabel("Max Normalized \n Score", rotation="horizontal")

    fig.set_size_inches(14, 5)
    plt.savefig("performance_bar.svg")
    plt.show()


def generate_return_distributions():

    env_list = ["ant", "swimmer", "walker2d", "hopper", "halfcheetah"]
    # env_list = ["halfcheetah", "swimmer"]
    csv_path = "/Users/jadkins/ppo_brax_logged_means.csv"

    df = pd.read_csv(csv_path)
    # normalized_df = normalize_scores(df, env_list)
    df = df.groupby(
        ["alg_type", "env_name", "actor_lr", "critic_lr", "ent_coef", "gae_lambda"]
    ).mean()
    df = df.reset_index()
    df = df.drop(
        ["seed_idx", "start_seed"],
        axis=1,
    )
    # fig, axs = plt.subplots(1, len(env_list))
    fig, ax = plt.subplots()

    # fig.supxlabel("Return")
    fig.suptitle("Return Distributions")

    for env in env_list:
        p_bar_a_e_theta_env = df.loc[df["env_name"] == env].copy()
        _max = p_bar_a_e_theta_env["mean_episode_return"].max()
        _min = p_bar_a_e_theta_env["mean_episode_return"].min()

        p_bar_a_e_theta_env[f"{env} normalized score"] = (
            p_bar_a_e_theta_env["mean_episode_return"] - _min
        ) / (_max - _min)

        # params good at hopper and bad at ant

        # (p_bar_a_e_theta_env["actor_lr"] == 0.0001)
        # & (p_bar_a_e_theta_env["critic_lr"] ==  0.001)
        # & (p_bar_a_e_theta_env["ent_coef"] == 0.1)
        # & (p_bar_a_e_theta_env["gae_lambda"] == 0.9)
        # & (p_bar_a_e_theta_env["alg_type"] == "norm_obs")

        # good param set over all envs
        # (p_bar_a_e_theta_env["actor_lr"] == 0.0001)
        # & (p_bar_a_e_theta_env["critic_lr"] == 0.0001)
        # & (p_bar_a_e_theta_env["ent_coef"] == 0.001)
        # & (p_bar_a_e_theta_env["gae_lambda"] == 0.9)

        theta_a_e_star = p_bar_a_e_theta_env.loc[
            (p_bar_a_e_theta_env["alg_type"] == "norm_obs")
        ]
        hyper_point = theta_a_e_star.loc[
            (p_bar_a_e_theta_env["actor_lr"] == 0.0001)
            & (p_bar_a_e_theta_env["critic_lr"] == 0.001)
            & (p_bar_a_e_theta_env["ent_coef"] == 0.1)
            & (p_bar_a_e_theta_env["gae_lambda"] == 0.9)
        ][f"mean_episode_return"]

        if env in ["swimmer", "halfcheetah"]:

            sns.kdeplot(
                ax=ax,
                data=theta_a_e_star,
                # x=f"{env} normalized score",
                x=f"mean_episode_return",
                label=env,
            )
            # ax.plot(
            #     float(hyper_point),
            #     plt.gca()
            #     .get_lines()[-1]
            #     .get_xydata()[
            #         abs(
            #             plt.gca().get_lines()[-1].get_xydata()[:, 0]
            #             - float(hyper_point)
            #         ).argmin()
            #     ][1],
            #     "o",
            #     color="red",
            # )

        # ax.set_ylabel("")
        ax.set_xlabel("Mean Episode Return")
        # ax.set_ylabel("Density", rotation="horizontal")
        # ax.set_xticks([0.0, 1.0])
        # ax.set_ylim(0, 0.4)
        # ax.set_xlim(0, 1.0)
        ax.set_xscale("log")

        ax.legend()

        # ax.set_xticks([theta_a_e_star[f"mean_episode_return"].mean()])
    fig.set_size_inches(3, 3)
    plt.savefig("return_dist_normalized.svg")

    plt.show()

    # ax.set_xlim(0, 5000)
    # sns.displot(
    #     data=df.loc[
    #         (df["env_name"] == "walker2d")
    #         & (df["actor_lr"] == 0.01)
    #         & (df["critic_lr"] == 0.01)
    #         & (df["ent_coef"] == 0.1)
    #         & (df["gae_lambda"] == 0.9)
    #     ],
    #     x="mean_episode_return",
    #     kind="kde",
    # )


generate_return_distributions()

# generate_shaded_dummy_sensitivty_plot()
# generate_performance_bar_plot()
