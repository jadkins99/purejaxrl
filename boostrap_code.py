import pickle as pkl
import pandas as pd
import numpy as np
import psycopg2
import os
import matplotlib.pyplot as plt
import itertools

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({"font.size": 7})
# plt.rcParams["text.usetex"] = True


def cdf_normalize_scores(df, env_list):
    df_normalized_scores = []

    for env in env_list:
        df_env = df.loc[df["env_name"] == env]
        df_env = df_env.sort_values(by="mean_episode_return")

        df_env = df_env.reset_index(drop=True)
        df_env = df_env.reset_index()
        df_env[f"{env} normalized score"] = df_env["index"] / df_env.shape[0]
        df_env = df_env.groupby(
            ["alg_type", "env_name", "actor_lr", "critic_lr", "ent_coef", "gae_lambda"]
        ).mean()
        df_env = df_env.reset_index()
        df_env = df_env.drop(
            ["index", "mean_episode_return", "seed_idx", "start_seed", "env_name"],
            axis=1,
        )

        df_normalized_scores.append(df_env)

    merged_df = df_normalized_scores[0]
    for df in df_normalized_scores[1:]:
        merged_df = pd.merge(merged_df, df, how="inner")

    merged_df["mean normalized score"] = merged_df[
        [f"{env} normalized score" for env in env_list]
    ].mean(axis=1)

    return merged_df


def normalize_scores(df):

    env_return_bounds = {}
    for env in df.env_name.unique():
        env_return_bounds[env] = {
            "max": df.loc[df["env_name"] == env]["mean_episode_return"].max(),
            "min": df.loc[df["env_name"] == env]["mean_episode_return"].min(),
        }
    df["normalized_score"] = df.apply(
        lambda row: 1
        - (env_return_bounds[row.env_name]["max"] - row.mean_episode_return)
        / (
            env_return_bounds[row.env_name]["max"]
            - env_return_bounds[row.env_name]["min"]
        ),
        axis=1,
    )
    df = df.drop("mean_episode_return", axis=1)
    return df


def compute_scores(normalized_df, alg_list):

    score_dict = {}
    # take expectation over seeds
    normalized_df_expected = normalized_df.groupby(
        ["env_name", "alg_type", "actor_lr", "critic_lr", "ent_coef", "gae_lambda"],
        dropna=True,
    ).mean()
    normalized_df_expected = normalized_df.reset_index()
    normalized_df_expected = normalized_df.drop("seed_val", axis=1)

    cross_envs_hypers_algs = normalized_df_expected.groupby(
        ["alg_type", "actor_lr", "critic_lr", "ent_coef", "gae_lambda"],
        dropna=True,
    ).mean(numeric_only=True)["normalized_score"]
    cross_envs_hypers_algs = cross_envs_hypers_algs.reset_index()

    cross_envs_algs = cross_envs_hypers_algs.groupby(["alg_type"]).max()[
        "normalized_score"
    ]
    cross_envs_algs = cross_envs_algs.reset_index()

    max_per_env_algs_envs = normalized_df_expected.groupby(
        ["alg_type", "env_name"]
    ).max()["normalized_score"]
    max_per_env_algs_envs = max_per_env_algs_envs.reset_index()
    max_per_env_algs = max_per_env_algs_envs.groupby(["alg_type"]).mean()[
        "normalized_score"
    ]
    max_per_env_algs = max_per_env_algs.reset_index()

    score_dict = {}
    for alg in alg_list:
        score_dict[f"{alg}_performance"] = max_per_env_algs.loc[
            max_per_env_algs["alg_type"] == alg
        ]["normalized_score"].item()
        score_dict[f"{alg}_sensitivity"] = (
            max_per_env_algs.loc[max_per_env_algs["alg_type"] == alg][
                "normalized_score"
            ].item()
            - cross_envs_algs.loc[cross_envs_algs["alg_type"] == alg][
                "normalized_score"
            ].item()
        )

    return score_dict


def boostrap_scores(normalized_df, alg_list, num_points=100):
    boostrapped_score_data = pd.DataFrame(
        columns=[f"{alg}_sensitivity" for alg in alg_list]
        + [f"{alg}_performance" for alg in alg_list]
    )

    # https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    # df = df.assign(
    #     seed_val=lambda x: 0.5
    #     * (x.start_seed + x.seed_idx)
    #     * (x.start_seed + x.seed_idx + 1)
    #     + x.seed_idx
    # )
    # df = df.drop("start_seed", axis=1)
    # df = df.drop("seed_idx", axis=1)

    sampled_seeds = np.random.choice(
        normalized_df["seed_val"].unique(),
        replace=True,
        size=(num_points, normalized_df["seed_val"].unique().size),
    )

    import time

    for m in range(num_points):
        st = time.time()

        print(f"performing iteration {m}")

        normalized_boostrapped_dataset = pd.concat(
            list(
                map(
                    lambda seed: normalized_df.loc[normalized_df["seed_val"] == seed],
                    sampled_seeds[m],
                )
            ),
            axis="rows",
        )

        scores = compute_scores(normalized_boostrapped_dataset, alg_list)
        boostrapped_score_data = pd.concat(
            [boostrapped_score_data, pd.DataFrame(scores, index=[0])]
        )
        et = time.time()
        print(f"compute time {et-st}")

    return boostrapped_score_data


def best_step_down_plot(
    concat_scores, env_list, alg_list, performance_sensitivity_scores, change_labels
):
    hyper_list = []
    for column in concat_scores.columns:
        if not (
            "normalized" in column or column in ["alg_type", "mean normalized score"]
        ):
            hyper_list.append(column)

    bestm1hyper_dict = {}
    bestm2hyper_dict = {}
    bestm3hyper_dict = {}
    bestm4hyper_dict = {}

    best_dict = {}

    for alg in alg_list:
        theta_a_star = concat_scores.loc[concat_scores["alg_type"] == alg][
            "mean normalized score"
        ].idxmax()

        bestm4hyper_dict[alg] = (
            "all fixed",
            concat_scores.loc[concat_scores["alg_type"] == alg][
                f"mean normalized score"
            ].max(),
        )
        best_score_over_envs = sum(
            [
                concat_scores.loc[concat_scores["alg_type"] == alg][
                    f"{env} normalized score"
                ].max()
                for env in env_list
            ]
        ) / len(env_list)

        best_dict[alg] = ("tune all", best_score_over_envs)

        # compute best leave one out score
        best_score_with_one_fixed = -100
        best_hyper_with_one_fixed = None
        for hyper in hyper_list:
            score = sum(
                [
                    concat_scores.loc[
                        (concat_scores["alg_type"] == alg)
                        & (
                            concat_scores[hyper]
                            == concat_scores.iloc[theta_a_star][hyper]
                        )
                    ][f"{env} normalized score"].max()
                    for env in env_list
                ]
            ) / len(env_list)
            if score > best_score_with_one_fixed:

                best_score_with_one_fixed = score
                best_hyper_with_one_fixed = hyper

        bestm1hyper_dict[alg] = (best_hyper_with_one_fixed, best_score_with_one_fixed)

        # compute best leave 2 out score
        best_score_with_two_fixed = -100
        best_hyper_with_two_fixed = None
        combinations = itertools.combinations(hyper_list, 2)
        for combination in combinations:
            score = sum(
                [
                    concat_scores.loc[
                        (concat_scores["alg_type"] == alg)
                        & (
                            concat_scores[combination[0]]
                            == concat_scores.iloc[theta_a_star][combination[0]]
                        )
                        & (
                            concat_scores[combination[1]]
                            == concat_scores.iloc[theta_a_star][combination[1]]
                        )
                    ][f"{env} normalized score"].max()
                    for env in env_list
                ]
            ) / len(env_list)
            if score > best_score_with_two_fixed:

                best_score_with_two_fixed = score
                best_hyper_with_two_fixed = " ".join(combination)

        bestm2hyper_dict[alg] = (best_hyper_with_two_fixed, best_score_with_two_fixed)

        # compute best leave 3 out score
        best_score_with_three_fixed = -100
        best_hyper_with_three_fixed = None
        combinations = itertools.combinations(hyper_list, 3)
        for combination in combinations:

            score = sum(
                [
                    concat_scores.loc[
                        (concat_scores["alg_type"] == alg)
                        & (
                            concat_scores[combination[0]]
                            == concat_scores.iloc[theta_a_star][combination[0]]
                        )
                        & (
                            concat_scores[combination[1]]
                            == concat_scores.iloc[theta_a_star][combination[1]]
                        )
                        & (
                            concat_scores[combination[2]]
                            == concat_scores.iloc[theta_a_star][combination[2]]
                        )
                    ][f"{env} normalized score"].max()
                    for env in env_list
                ]
            ) / len(env_list)
            if score > best_score_with_three_fixed:

                best_score_with_three_fixed = score
                best_hyper_with_three_fixed = " ".join(combination)

        bestm3hyper_dict[alg] = (
            best_hyper_with_three_fixed,
            best_score_with_three_fixed,
        )

    # alg_list.sort(
    #     key=lambda alg: performance_sensitivity_scores[f"{alg} performance"],
    #     reverse=True,
    # )

    # fig, ax = plt.subplots(3,3)

    # # bar_labels = ["red", "blue", "_red", "orange"]
    colors = [
        "tab:red",
        "tab:blue",
        "tab:orange",
        "tab:purple",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    # bar_xs = list(
    #     itertools.chain.from_iterable(
    #         [
    #             [
    #                 f"{alg} All fixed",
    #                 f"{alg} 1 tuned hyper",
    #                 f"{alg} 2 tuned hypers",
    #                 f"{alg} 3 tuned hypers",
    #                 f"{alg} All tuned",
    #             ]
    #             for alg in alg_list
    #         ]
    #     )
    # )
    # bar_ys = list(
    #     itertools.chain.from_iterable(
    #         [
    #             [
    #                 bestm4hyper_dict[alg][1],
    #                 bestm3hyper_dict[alg][1],
    #                 bestm2hyper_dict[alg][1],
    #                 bestm1hyper_dict[alg][1],
    #                 best_dict[alg][1],
    #             ]
    #             for alg in alg_list
    #         ]
    #     )
    # )
    fig, axs = plt.subplots(3, 2)

    fig.supxlabel(
        "Number of tuned hyperparameters",
        # $\mathbb{E}_{e \in E}[max_{\eta \in H^a}[s(a,e,\eta)]] - max_{\eta \in H^a} [\mathbb{E}_{e \in E}[S(a,e,\eta)]]$"
    )
    fig.supylabel(
        "Performance",
        # \n $\mathbb{E}_{e \in E}[max_{\eta \in H^a}[s(a,e,\eta)]]$",
        rotation="horizontal",
    )

    for row in range(3):
        for column in range(2):
            # we want to map (row,column) to into between 1...6. i.e. the indices of all non ppo algs in alg_list

            alg = alg_list[2 * row + column + 1]
            axs[row, column].set_title(f"PPO vs.  {change_labels[alg]}")
            axs[row, column].set_ylim(0.5, 1.0)
            axs[row, column].set_xticks(np.arange(0, 5, 1.0))
            axs[row, column].set_yticks(np.arange(0.5, 1, 0.1))

            axs[row, column].plot(
                [
                    bestm4hyper_dict["lambda_ac"][1],
                    bestm3hyper_dict["lambda_ac"][1],
                    bestm2hyper_dict["lambda_ac"][1],
                    bestm1hyper_dict["lambda_ac"][1],
                    best_dict["lambda_ac"][1],
                ],
                label=change_labels["lambda_ac"],
                color=colors[0],
            )

            axs[row, column].plot(
                [
                    bestm4hyper_dict[alg][1],
                    bestm3hyper_dict[alg][1],
                    bestm2hyper_dict[alg][1],
                    bestm1hyper_dict[alg][1],
                    best_dict[alg][1],
                ],
                label=change_labels[alg],
                color=colors[2 * row + column + 1],
            )
            axs[row, column].legend(loc="upper left")

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 6)
    plt.savefig("sensitivity_dimension.svg", dpi=100)
    plt.show()

    # for i, alg in enumerate(alg_list):
    #     ax.plot(
    #         [
    #             bestm4hyper_dict[alg][1],
    #             bestm3hyper_dict[alg][1],
    #             bestm2hyper_dict[alg][1],
    #             bestm1hyper_dict[alg][1],
    #             best_dict[alg][1],
    #         ],
    #         label=change_labels[alg],
    #         color=bar_colors[i],
    #     )

    # ax.bar(
    #     bar_xs,
    #     bar_ys,
    #     # label=change_labels[alg],
    #     # color=bar_colors[i],
    # )

    # ax.set_ylabel("Performance")
    # ax.set_xlabel("Number of hyperparameters tuned")

    # # ax.set_title(f" {alg} performance leaving out  hyperparameters")
    # ax.legend()

    # plt.show()
    # plt.clf()
    # plt.bar(alg_list, [bestm4hyper_dict[alg][1] for alg in alg_list])
    # plt.show()


def step_plot(concat_scores, alg_list, env_list):
    hyper_list = []
    for column in concat_scores.columns:
        if not (
            "normalized" in column or column in ["alg_type", "mean normalized score"]
        ):
            hyper_list.append(column)

    bestmhyper_dict = {}
    best_dict = {}
    for alg in alg_list:
        theta_a_star = concat_scores.loc[concat_scores["alg_type"] == alg][
            "mean normalized score"
        ].idxmax()
        # best_score_over_envs is E_e[max_h[S(alg,e,h)]]
        best_score_over_envs = sum(
            [
                concat_scores.loc[concat_scores["alg_type"] == alg][
                    f"{env} normalized score"
                ].max()
                for env in env_list
            ]
        ) / len(env_list)
        best_dict[alg] = best_score_over_envs
        bestmhyper_dict[alg] = {}

        for hyper in hyper_list:
            # best_score_with_hyper_fixed is  E_e[max_h[S(alg,e,h)| h_{hyper} = theta_a_star_{hyper}]]
            best_score_with_hyper_fixed = sum(
                [
                    concat_scores.loc[
                        (concat_scores["alg_type"] == alg)
                        & (
                            concat_scores[hyper]
                            == concat_scores.iloc[theta_a_star][hyper]
                        )
                    ][f"{env} normalized score"].max()
                    for env in env_list
                ]
            ) / len(env_list)
            bestmhyper_dict[alg][hyper] = best_score_with_hyper_fixed

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:olive",
        "tab:purple",
        "tab:brown",
        "tab:red",
        "tab:cyan",
    ]

    for alg in alg_list:

        fig, ax = plt.subplots()

        bar_labels = ["red", "blue", "_red", "orange"]
        bar_colors = ["tab:red", "tab:blue", "tab:red", "tab:orange"]

        ax.axhline(
            y=best_dict[alg],
            linestyle="-",
            label=f"{alg} max performance",
            color=colors[-1],
        )

        ax.bar(
            bestmhyper_dict[alg].keys(),
            bestmhyper_dict[alg].values(),
            label=bar_labels,
            color=bar_colors,
        )

        ax.set_ylabel("Performance")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f" {alg} performance of leave one out hyperparameter")

        # ax.legend(title="Fruit color")

        plt.show()

        # for i, hyper_item in enumerate(delta_dict[alg]):
        #     plt.axhline(
        #         y=hyper_item[1],
        #         linestyle="-",
        #         label=f"{alg} all except {hyper_item[0]}",
        #         color=colors[i],
        #     )
        # epsilons = np.cumsum(list(map(lambda x: x[1], delta_dict[alg])))

        # plt.step(range(0, len(epsilons) + 1), np.hstack([np.array([0.0]), epsilons]))
        # plt.title(f"{alg}")
        # plt.legend()
        # plt.ylim(0, scores[f"{alg} sensitivity"] + 0.03)
        # plt.show()


def make_boostrap_error_plot(
    scores, alg_list, boot_strapped_score_dataset, change_labels
):

    sensitivity_scores = {"obs": [], "critic": [], "advn": []}
    performance_scores = {"obs": [], "critic": [], "advn": []}
    sensitivity_errors = {"obs": [], "critic": [], "advn": []}
    performance_errors = {"obs": [], "critic": [], "advn": []}

    errors = boot_strapped_score_dataset.quantile(
        0.975
    ) - boot_strapped_score_dataset.quantile(0.025)
    for alg in alg_list:
        if alg in ["lambda_ac", "norm_obs", "symlog_obs"]:
            sensitivity_scores["obs"].append((alg, scores[f"{alg}_sensitivity"]))
            performance_scores["obs"].append((alg, scores[f"{alg}_performance"]))
            sensitivity_errors["obs"].append((alg, errors.loc[f"{alg}_sensitivity"]))
            performance_errors["obs"].append((alg, errors.loc[f"{alg}_performance"]))

        if alg in ["lambda_ac", "advn_norm_ema", "advn_norm_max_ema", "advn_norm_mean"]:
            sensitivity_scores["advn"].append((alg, scores[f"{alg}_sensitivity"]))
            performance_scores["advn"].append((alg, scores[f"{alg}_performance"]))
            sensitivity_errors["advn"].append((alg, errors.loc[f"{alg}_sensitivity"]))
            performance_errors["advn"].append((alg, errors.loc[f"{alg}_performance"]))

        if alg in ["lambda_ac", "symlog_critic_targets"]:
            sensitivity_scores["critic"].append((alg, scores[f"{alg}_sensitivity"]))
            performance_scores["critic"].append((alg, scores[f"{alg}_performance"]))
            sensitivity_errors["critic"].append((alg, errors.loc[f"{alg}_sensitivity"]))
            performance_errors["critic"].append((alg, errors.loc[f"{alg}_performance"]))

    plt.style.use("_mpl-gallery")

    alg_list = [
        "lambda_ac",
        "advn_norm_ema",
        "advn_norm_max_ema",
        "advn_norm_mean",
        "symlog_critic_targets",
        "symlog_obs",
        "norm_obs",
    ]

    # make data:
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
    # plot:
    fig, axs = plt.subplots(1, 3, layout="constrained")

    axs[0].set_title("observation normalization")
    axs[1].set_title("critic target normalization")
    axs[2].set_title("advantage normalization")
    fig.supxlabel(
        "Sensitivity",
        # $\mathbb{E}_{e \in E}[max_{\eta \in H^a}[s(a,e,\eta)]] - max_{\eta \in H^a} [\mathbb{E}_{e \in E}[S(a,e,\eta)]]$"
    )
    fig.supylabel(
        "Performance",
        # \n $\mathbb{E}_{e \in E}[max_{\eta \in H^a}[s(a,e,\eta)]]$",
        rotation="horizontal",
    )

    counter = 0
    for j, key in enumerate(sensitivity_scores.keys()):
        for i, _ in enumerate(sensitivity_scores[key]):

            axs[j].errorbar(
                sensitivity_scores[key][i][1],
                performance_scores[key][i][1],
                yerr=performance_errors[key][i][1],
                xerr=sensitivity_errors[key][i][1],
                label=change_labels[sensitivity_scores[key][i][0]],
                fmt="o",
                linewidth=2,
                capsize=6,
                c=colors[sensitivity_scores[key][i][0]],
            )
            counter += 1
            if change_labels[sensitivity_scores[key][i][0]] == "PPO":
                ppo_sensitivty = sensitivity_scores[key][i][1]
                ppo_performance = performance_scores[key][i][1]
                print(ppo_sensitivty)
                print(ppo_performance)

        # point slope form of line wtih slope one passing through PPO point
        axs[j].plot(
            np.array([0, 0.1, 0.2]),
            ppo_performance + 1.0 * ([0, 0.1, 0.2] - ppo_sensitivty),
            "--",
            color="purple",
        )
        axs[j].axvline(
            ppo_sensitivty,
            color="purple",
        )
        axs[j].axhline(
            ppo_performance,
            color="purple",
        )

    for ax in axs:
        ax.grid(False)
        ax.legend(loc="lower right")
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(0.0, 0.2)
        ax.set_xticks([0.0, 0.1, 0.2])
        # ax.set_yticks([0.6, 0.8, 1.0])

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(10, 6)

    plt.savefig(
        f"/Users/jadkins/csensitivity_all_envs.svg",
        dpi=100,
    )

    plt.show()


if __name__ == "__main__":

    csv_path = "/Users/jadkins/ppo_brax_logged_means.csv"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [
            "alg_type",
            "env_name",
            "start_seed",
            "seed_idx",
            "actor_lr",
            "critic_lr",
            "ent_coef",
            "gae_lambda",
            "mean_episode_return",
        ]
        # df["mean_episode_return"] = np.random.normal(size=df["mean_episode_return"].shape)
        df = df.dropna()
        print("df obtained!")
        df = df.assign(
            seed_val=lambda x: 0.5
            * (x.start_seed + x.seed_idx)
            * (x.start_seed + x.seed_idx + 1)
            + x.seed_idx
        )
        df = df.drop("start_seed", axis=1)
        df = df.drop("seed_idx", axis=1)

    else:
        conn = psycopg2.connect(
            host="35.194.35.36",
            password="Bearad01@",
            database="sweep_db",
            user="jadkins",
        )
        print("db connected!")
        df = pd.read_sql("select * from prelim_runs2;", conn)
        df = df.dropna()

        print("df obtained!")
        # df.to_csv(csv_path)

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
        "symlog_obs": "PPO symlog($o_t$)",
        "norm_obs": "PPO $(o_t-\mu_{o_t})/\sigma_{o_t}$",
        "advn_norm_ema": "PPO $A_t/(perc(A_t,95) - perc(A_t,5))$",
        "advn_norm_max_ema": "PPO $A_t/(max(1,perc(A_t,95) - perc(A_t,5)))$",
        "advn_norm_mean": "PPO $(A_t - \mu_{A_t})/\sigma_{A_t}$",
        "symlog_critic_targets": "PPO $symlog(G_{t:k+L}^\lambda)$",
    }

    # alg_list.sort(key=lambda alg: df.loc[df["alg_type"] == alg].shape[0], reverse=True)
    # env_list = ["Pendulum-v1", "CartPole-v1", "MountainCar-v0"]
    env_list = ["ant", "swimmer", "walker2d", "halfcheetah", "hopper"]
    env_list.sort(key=lambda env: df.loc[df["env_name"] == env].shape[0], reverse=True)
    # filter so we only consider env and algs in the lists
    # df = df.loc[df["alg_type"].isin(alg_list)]
    # df = df.loc[df["env_name"].isin(env_list)]

    num_points = 103
    if os.path.exists(f"boot_strapped_score_dataset_{num_points}.pkl"):
        boot_strapped_score_dataset = pkl.load(
            open(f"boot_strapped_score_dataset_{num_points}.pkl", "rb")
        )

    else:

        boot_strapped_score_dataset = boostrap_scores(
            df, alg_list, env_list, num_points
        )

        with open(
            f"boot_strapped_score_dataset_{boot_strapped_score_dataset.shape[0]}.pkl",
            "wb",
        ) as f:
            pkl.dump(boot_strapped_score_dataset, f)

    concat_scores = normalize_scores(df)
    # concat_scores = cdf_normalize_scores(df, env_list=env_list)

    scores = compute_scores(concat_scores, alg_list=alg_list)

    # step_plot(concat_scores, alg_list, env_list)
    # best_step_down_plot(
    #     concat_scores, env_list, alg_list, scores, change_labels=change_labels
    # )

    make_boostrap_error_plot(
        scores, alg_list, boot_strapped_score_dataset, change_labels=change_labels
    )
    # import scipyf

    # for alg in alg_list:
    #     acrobot_ranking = concat_scores[concat_scores["alg_type"] == alg].sort_values(
    #         "acrobot normalized score", ascending=False
    #     )["actent"]
    #     pendulum_ranking = concat_scores[concat_scores["alg_type"] == alg].sort_values(
    #         "pendulum normalized score", ascending=False
    #     )["actent"]
    #     mc_ranking = concat_scores[concat_scores["alg_type"] == alg].sort_values(
    #         "mc normalized score", ascending=False
    #     )["actent"]

    #     a_p_corr = scipy.stats.kendalltau(acrobot_ranking, pendulum_ranking)
    #     a_m_corr = scipy.stats.kendalltau(acrobot_ranking, mc_ranking)
    #     p_m_corr = scipy.stats.kendalltau(pendulum_ranking, mc_ranking)

    #     print(f"alg: {alg}")
    #     print(f"acrobot-pendulum correlation: {a_p_corr}")
    #     print(f"acrobot-MC correlation: {a_m_corr}")
    #     print(f"pendulum-MC correlation: {p_m_corr}")
