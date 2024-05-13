import matplotlib.pyplot as plt
import gc
import jax.numpy as jnp
import os
import pandas as pd
import seaborn as sns
from boostrap_code import normalize_scores, boostrap_scores, compute_scores
import numpy as np
import pickle as pkl


change_labels = {
    "lambda_ac": "PPO",
    "symlog_obs": "Symlog observation",
    "norm_obs": "Observation $\mu -\sigma$ normalization",
    "advn_norm_ema": "Percentile scaling",
    "advn_norm_max_ema": "Lower bounded percentile scaling",
    "advn_norm_mean": "Per-minibatch $\mu -\sigma$ normalization",
    "symlog_critic_targets": "Symlog value target",
}
CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def generate_shaded_dummy_sensitivty_plot():
    ref_sens = 0.3
    ref_perf = 0.6
    unit_line = lambda x: ref_perf + 1.0 * (x - ref_sens)
    xs = np.linspace(0, 0.6, 1000)
    alpha = 0.1

    # colors = {
    #     "lambda_ac": "tab:blue",
    #     "symlog_critic_targets": "tab:orange",
    #     "advn_norm_ema": "tab:green",
    #     "advn_norm_max_ema": "tab:olive",
    #     "advn_norm_mean": "tab:purple",
    #     "symlog_obs": "tab:brown",
    #     "norm_obs": "tab:cyan",
    #     # "tab:red",
    #     # "tab:gray",
    # }
    fig, ax = plt.subplots(layout="constrained")
    fig.supxlabel(
        "Hyperparameter Sensitivity",
        horizontalalignment="center",
        # $\mathbb{E}_{e \in E}[max_{\eta \in H^a}[s(a,e,\eta)]] - max_{\eta \in H^a} [\mathbb{E}_{e \in E}[S(a,e,\eta)]]$"
    )
    fig.supylabel(
        "Environment Tuned \n Performance",
        # \n $\mathbb{E}_{e \in E}[max_{\eta \in H^a}[s(a,e,\eta)]]$",
        rotation="horizontal",
    )
    # ax.set_title("An illustration ")
    ax.set_ylim(ref_perf - 0.2, ref_perf + 0.2)
    ax.set_xlim(ref_sens - 0.1, ref_sens + 0.1)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xticks(
        [ref_sens - 0.2, ref_sens - 0.1, ref_sens, ref_sens + 0.1, ref_sens + 0.2]
    )

    ax.set_yticks(
        [ref_perf - 0.2, ref_perf - 0.1, ref_perf, ref_perf + 0.1, ref_perf + 0.2]
    )

    # ax.axhline(ref_perf, color="blue", linestyle="--")
    # ax.axvline(ref_sens, color="blue", linestyle="--")
    # ax.plot(xs, unit_line(xs), color="blue", linestyle="--")
    CB_color_cycle = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]

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
    ax.fill_between(
        xs,
        unit_line(xs),
        ref_perf,
        color="red",
        alpha=alpha,
    )
    # plt.plot(ref_sens, ref_perf, "bo")

    plt.savefig("cartoon_plot.svg")
    plt.show()


def generate_sensitivty_plot():
    csv_path = "/Users/jadkins/ppo_brax_logged_means.csv"

    df = pd.read_csv(csv_path)
    env_list = ["ant", "swimmer", "walker2d", "hopper", "halfcheetah"]
    alg_list = [
        "lambda_ac",
        "advn_norm_ema",
        "advn_norm_max_ema",
        "advn_norm_mean",
        "symlog_critic_targets",
        "symlog_obs",
        "norm_obs",
    ]
    change_labels = {
        "lambda_ac": "PPO",
        "symlog_obs": "Symlog observation",
        "norm_obs": "Observation $\mu -\sigma$ normalization",
        "advn_norm_ema": "Percentile scaling",
        "advn_norm_max_ema": "Lower bounded percentile scaling",
        "advn_norm_mean": "Per-minibatch $\mu -\sigma$ normalization",
        "symlog_critic_targets": "Symlog value target",
    }
    # https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    df = df.assign(
        seed_val=lambda x: 0.5
        * (x.start_seed + x.seed_idx)
        * (x.start_seed + x.seed_idx + 1)
        + x.seed_idx
    )
    df = df.drop(["start_seed", "seed_idx"], axis=1)

    num_points = 1000
    if os.path.exists(f"normalized_dataset"):
        normalized_df = pd.read_pickle("normalized_dataset")
    else:
        normalized_df = normalize_scores(df)
        normalized_df.to_pickle(f"normalized_dataset")

    scores_old_label = compute_scores(normalized_df, alg_list=alg_list)

    if os.path.exists(f"bootstrap_dataset_{num_points}"):
        bootstrap_dataset = pd.read_pickle(f"bootstrap_dataset_{num_points}")
    else:
        bootstrap_dataset = boostrap_scores(
            normalized_df, alg_list, num_points=num_points
        )
        bootstrap_dataset.to_pickle(f"bootstrap_dataset_{num_points}")

    # bootstrap_errors = bootstrap_dataset.quantile(0.975) - bootstrap_dataset.quantile(
    #     0.025
    # )
    CI_975 = {}
    CI_025 = {}
    scores = {}

    for alg in alg_list:
        scores[change_labels[alg]] = {
            "sensitivity": scores_old_label[f"{alg}_sensitivity"],
            "performance": scores_old_label[f"{alg}_performance"],
        }
        CI_975[change_labels[alg]] = {
            "sensitivity": bootstrap_dataset.quantile(0.975)[f"{alg}_sensitivity"]
            - scores_old_label[f"{alg}_sensitivity"],
            "performance": bootstrap_dataset.quantile(0.975)[f"{alg}_performance"]
            - scores_old_label[f"{alg}_performance"],
        }

        CI_025[change_labels[alg]] = {
            "sensitivity": scores_old_label[f"{alg}_sensitivity"]
            - bootstrap_dataset.quantile(0.025)[f"{alg}_sensitivity"],
            "performance": scores_old_label[f"{alg}_performance"]
            - bootstrap_dataset.quantile(0.025)[f"{alg}_performance"],
        }

    CB_color_cycle = {
        "PPO": "#377eb8",
        "Symlog observation": "#ff7f00",
        "Observation $\mu -\sigma$ normalization": "#4daf4a",
        "Percentile scaling": "#f781bf",
        "Lower bounded percentile scaling": "#a65628",
        "Per-minibatch $\mu -\sigma$ normalization": "#984ea3",
        "Symlog value target": "#999999",
        # "#e41a1c",
        # "#dede00",
    }
    for alg in alg_list:
        ref_sens = scores["PPO"]["sensitivity"]
        ref_perf = scores["PPO"]["performance"]
        # cross_env_score = normalized_df.loc[normalized_df["alg_type"] == alg][
        #     "mean normalized score"
        # ].max()
        # tuned_best_score = (
        #     normalized_df.loc[normalized_df["alg_type"] == alg][
        #         [f"{env} normalized score" for env in env_list]
        #     ]
        #     .max()
        #     .mean()
        # )
        # scores[change_labels[alg]] = {
        #     "performance": tuned_best_score,
        #     "sensitivity": tuned_best_score - cross_env_score,
        # }

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
    ax.set_ylim(ref_perf - 0.15, ref_perf + 0.3)
    ax.set_xlim(ref_sens - 0.1, ref_sens + 0.1)
    # ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

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
    ax.fill_between(
        xs,
        unit_line(xs),
        ref_perf,
        color="red",
        alpha=alpha,
    )

    # plt.plot(ref_sens, ref_perf, "bo")

    for key, score_dict in scores.items():
        plt.errorbar(
            score_dict["sensitivity"],
            score_dict["performance"],
            yerr=np.array(
                [CI_025[key]["performance"], CI_975[key]["performance"]]
            ).reshape(2, 1),
            xerr=np.array(
                [CI_025[key]["sensitivity"], CI_975[key]["sensitivity"]]
            ).reshape(2, 1),
            label=key,
            fmt="o",
            linewidth=2,
            capsize=6,
            c=CB_color_cycle[key],
        )
        # plt.plot(
        #     score_dict["sensitivity"],
        #     score_dict["performance"],
        #     color=CB_color_cycle[key],
        #     marker="o",
        # )
    print(CB_color_cycle)
    plt.savefig("real_plot.svg")
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

    # env_list = ["ant", "swimmer", "walker2d", "hopper", "halfcheetah"]
    env_list = ["halfcheetah", "swimmer"]
    csv_path = "/Users/jadkins/ppo_brax_logged_means.csv"

    df = pd.read_csv(csv_path)
    df = df.loc[df["alg_type"] == "norm_obs"]
    df_normed = normalize_scores(df, ["swimmer", "halfcheetah"]).dropna()

    df = df.groupby(
        ["alg_type", "env_name", "actor_lr", "critic_lr", "ent_coef", "gae_lambda"]
    ).mean()
    df = df.reset_index()
    df = df.drop(["seed_idx", "start_seed"], axis=1)

    swimmer_data = pd.DataFrame(df.loc[df["env_name"] == "swimmer"].dropna())
    cheetah_data = pd.DataFrame(df.loc[df["env_name"] == "halfcheetah"].dropna())

    labels = ["swimmer", "halfcheetah"]
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    bplot1 = ax1.boxplot(
        [
            swimmer_data["mean_episode_return"].to_numpy(),
            cheetah_data["mean_episode_return"].to_numpy(),
        ],
        notch=False,  # notch shape
        vert=True,  # vertical box alignment
        patch_artist=True,  # fill with color
        autorange=False,
        labels=labels,
        medianprops={"color": CB_color_cycle[-3]},
    )  # will be used to label x-ticks
    ax1.set_title("Performance (AUC)")
    # ax1.set_yscale("symlog")

    bplot2 = ax2.boxplot(
        [
            df_normed["swimmer normalized score"].to_numpy(),
            df_normed["halfcheetah normalized score"].to_numpy(),
        ],
        notch=False,  # notch shape
        medianprops={"color": CB_color_cycle[-3]},
        vert=True,  # vertical box alignment
        autorange=False,
        patch_artist=True,  # fill with color
        labels=labels,
    )  # will be used to label x-ticks
    ax2.set_title("Standarized Score")

    # colors = ["pink", "lightblue"]
    for bplot in (bplot1, bplot2):
        for i, patch in enumerate(bplot["boxes"]):
            patch.set_facecolor(CB_color_cycle[i])

    hp1 = df_normed.loc[
        (df_normed["actor_lr"] == 0.0001)
        & (df_normed["critic_lr"] == 0.001)
        & (df_normed["ent_coef"] == 0.1)
        & (df_normed["gae_lambda"] == 0.9)
    ][f"swimmer normalized score"]
    hp2 = df_normed.loc[
        (df_normed["actor_lr"] == 0.0001)
        & (df_normed["critic_lr"] == 0.001)
        & (df_normed["ent_coef"] == 0.1)
        & (df_normed["gae_lambda"] == 0.9)
    ][f"halfcheetah normalized score"]
    # print(hp1)
    print(df_normed.loc[df_normed["halfcheetah normalized score"].idxmax()])
    print(df_normed.loc[df_normed["swimmer normalized score"].idxmax()])

    print(df_normed.loc[df_normed["mean normalized score"].idxmax()])

    plt.savefig("return_dist_normalized.svg")

    # plt.show()
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    ax.set_title("Standarized Score")

    bplot = ax.boxplot(
        [
            df_normed["swimmer normalized score"].to_numpy(),
            df_normed["halfcheetah normalized score"].to_numpy(),
        ],
        notch=False,  # notch shape
        medianprops={"color": CB_color_cycle[-3]},
        vert=True,  # vertical box alignment
        autorange=False,
        patch_artist=True,  # fill with color
        labels=labels,
    )  # will be used to label x-ticks
    for i, patch in enumerate(bplot["boxes"]):
        patch.set_facecolor(CB_color_cycle[i])
    plt.savefig("return_dist_normalized_hypers_shown.svg")

    plt.show()


# generate_return_distributions()
generate_sensitivty_plot()
# generate_shaded_dummy_sensitivty_plot()
# generate_performance_bar_plot()
