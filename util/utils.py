import mlrose_hiive
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator
from mlrose_hiive import SARunner, GARunner, MIMICRunner, RHCRunner
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import string
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool

import seaborn as sns
import matplotlib.pyplot as plt

# switch off the chatter
import logging
import copy

logging.basicConfig(level=logging.WARNING)

def run_experiment_multi_seed(runner, seeds):
    # create a list of runners with the same parameters but different seeds
    runners = [copy.deepcopy(runner) for _ in seeds]
    for i, seed in enumerate(seeds):
        runners[i].seed = seed
        print(f"Runner {i} seed: {runners[i].seed}")

    with Pool() as pool:
        results_list = pool.map(run_experiment, runners)

    return results_list


def run_experiment(runner):
    res = runner.run()
    return res


def runner_results_to_stats(results):
    # Convert the results of the runner to a dictionary of statistics
    results_all_stats = [results[0] for results in results]

    results_mean = results_all_stats[0].copy()
    results_mean["Fitness"] = np.mean(
        [df["Fitness"] for df in results_all_stats], axis=0
    )
    results_mean["Time"] = np.mean([df["Time"] for df in results_all_stats], axis=0)
    results_mean["FEvals"] = np.mean([df["FEvals"] for df in results_all_stats], axis=0)

    # Do the same but with std
    results_std = results_all_stats[0].copy()
    results_std["Fitness"] = np.std([df["Fitness"] for df in results_all_stats], axis=0)
    results_std["Time"] = np.std([df["Time"] for df in results_all_stats], axis=0)
    results_std["FEvals"] = np.std([df["FEvals"] for df in results_all_stats], axis=0)

    # Compute the maximum fitness
    results_max = results_all_stats[0].copy()
    results_max["Fitness"] = np.max([df["Fitness"] for df in results_all_stats], axis=0)
    results_max["Time"] = np.max([df["Time"] for df in results_all_stats], axis=0)
    results_max["FEvals"] = np.max([df["FEvals"] for df in results_all_stats], axis=0)

    # Compute the minimum fitness
    results_min = results_all_stats[0].copy()
    results_min["Fitness"] = np.min([df["Fitness"] for df in results_all_stats], axis=0)
    results_min["Time"] = np.min([df["Time"] for df in results_all_stats], axis=0)
    results_min["FEvals"] = np.min([df["FEvals"] for df in results_all_stats], axis=0)

    return results_mean, results_std, results_min, results_max


def plot_fitness_iteration(
    df_mean, df_std, df_min, df_max, runnner_name, problem_name="", x_axis="Iteration"
):
    # Plot the fitness over the iterations
    plt.figure()
    # Fig size
    plt.figure(figsize=(10, 5))
    # Font size
    plt.rc("font", size=16)
    plt.plot(df_mean[x_axis], df_mean["Fitness"], label=f"{runnner_name}", color="b")
    plt.plot(
        df_max[x_axis],
        df_max["Fitness"],
        label=f"{runnner_name} Max",
        color="r",
        linestyle="--",
    )
    plt.plot(
        df_min[x_axis],
        df_min["Fitness"],
        label=f"{runnner_name} Min",
        color="g",
        linestyle="--",
    )
    plt.fill_between(
        df_mean[x_axis],
        df_mean["Fitness"] - df_std["Fitness"],
        df_mean["Fitness"] + df_std["Fitness"],
        color="b",
        alpha=0.2,
        label=f"{runnner_name} Std",
    )
    plt.xlabel(x_axis)
    plt.ylabel("Fitness")
    plt.title(f"Fitness over {x_axis} ({runnner_name})")
    plt.legend()
    plt.savefig(
        f"figures/{problem_name}_fitness_over_{x_axis}_{runnner_name}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()



def plot_fitness_fevals(
    df_mean, df_std, df_min, df_max, runnner_name, problem_name=""
):
    # Plot the fitness over the iterations
    plt.figure()
    # Fig size
    plt.figure(figsize=(10, 5))
    # Font size
    plt.rc("font", size=16)
    plt.plot(df_mean["FEvals"], df_mean["Fitness"], label=f"{runnner_name}", color="b")
    plt.plot(
        df_max["FEvals"],
        df_max["Fitness"],
        label=f"{runnner_name} Max",
        color="r",
        linestyle="--",
    )
    plt.plot(
        df_min["FEvals"],
        df_min["Fitness"],
        label=f"{runnner_name} Min",
        color="g",
        linestyle="--",
    )
    plt.fill_between(
        df_mean["FEvals"],
        df_mean["Fitness"] - df_std["Fitness"],
        df_mean["Fitness"] + df_std["Fitness"],
        color="b",
        alpha=0.2,
        label=f"{runnner_name} Std",
    )
    plt.xlabel("FEvals")
    plt.ylabel("Fitness")
    plt.title(f"Fitness over iterations ({runnner_name})")
    plt.legend()
    plt.savefig(
        f"figures/{problem_name}_fitness_over_fevals_{runnner_name}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


def get_optimal_hyperparameters(hp_mean, HP_LIST):
    # Get the optimal hyperparameters
    optimal_hyperparameters = {}
    best_run = hp_mean[hp_mean["Fitness"] == hp_mean["Fitness"].max()]
    for key in HP_LIST:
        optimal_hyperparameters[key] = best_run[key].values[0]

    return optimal_hyperparameters

def plot_fitness_vs_hyperparameter(
    hp_mean, hp_std, optimal_HP_dict, hyperparameter, runnner_name, problem_name="", x_axis="Iteration", y_lim=(-2000, -1100)
):
    # Plot the fitness over the hyperparameter
    if hyperparameter not in hp_mean.columns:
        raise ValueError(f"Hyperparameter {hyperparameter} not in dataframe columns")

    # pop the hyperparameter from the optimal_HP_dict
    HP_dict = optimal_HP_dict.copy()
    HP_dict.pop(hyperparameter, None)

    # filter the hp_mean and hp_std for the optimal hyperparameters
    for key, value in HP_dict.items():
        print(key)
        hp_mean = hp_mean[hp_mean[key] == value]
        hp_std = hp_std[hp_std[key] == value]

    # plt.figure()
    # Fig size
    plt.figure(figsize=(10, 5))
    # set y min to -2000
    plt.ylim(y_lim[0], y_lim[1])
    # Font size
    plt.rc("font", size=16)
    # Use a for loop using the temperature_list for each schedule type
    for H_value in hp_mean[hyperparameter].unique():
        temp_df = hp_mean[hp_mean[hyperparameter] == H_value]
        temp_df_std = hp_std[hp_mean[hyperparameter] == H_value]
        if type(hyperparameter) == str:
            plt.plot(
                temp_df[x_axis],
                temp_df["Fitness"],
                label=f"{hyperparameter} = {H_value}",
            )
        else:
            plt.plot(
                temp_df[x_axis],
                temp_df["Fitness"],
                label=f"{hyperparameter} = {H_value:.2f}",
            )

        print(f'{hyperparameter} = {H_value} Fitness: {temp_df["Fitness"].max()}, Time: {temp_df["Time"].max()}')

    plt.xlabel(x_axis)
    plt.ylabel("Fitness")
    plt.title(f"Fitness over {x_axis} ({runnner_name})")
    plt.legend()
    plt.savefig(
        f"figures/{problem_name}_fitness_over_{x_axis}_{hyperparameter}_{runnner_name}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.tight_layout()
    plt.show()


def plot_HP_heatmap(df, x, y, runner_name, problem_name=""):
    best_fitness = (
        df.groupby([x, y])
        .agg({"Fitness": "max"})
        .reset_index()
        .sort_values(by="Fitness", ascending=False)
    )
    heatmap_data = best_fitness.pivot_table(values="Fitness", index=y, columns=x)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="viridis", fmt=".2f", linewidths=0.5)
    plt.title(f"Best Fitness for Each Set of Hyperparameters ({runner_name})")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(
        f"figures/{problem_name}_best_fitness_heatmap_{runner_name}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


def show_TSP_map(problem, ordered_state):
    fig, ax = plt.subplots(1)  # Prepare 2 plots
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    for i, (x, y) in enumerate(problem.coords):
        ax.scatter(x, y, s=1, c="green" if i == 5 else "cornflowerblue")  # plot A

    for i in range(len(ordered_state)):
        start_node = ordered_state[i]
        end_node = ordered_state[(i + 1) % len(ordered_state)]
        start_pos = problem.coords[start_node]
        end_pos = problem.coords[end_node]
        ax.annotate(
            "",
            xy=start_pos,
            xycoords="data",
            xytext=end_pos,
            textcoords="data",
            c="red",
            arrowprops=dict(arrowstyle="->", ec="blue", connectionstyle="arc3"),
        )
    node_labels = {k: str(k) for k in range(len(problem.source_graph.nodes))}

    for i in node_labels.keys():
        x, y = problem.coords[i]
        plt.text(
            x,
            y,
            node_labels[i],
            ha="center",
            va="center",
            c="white",
            fontweight="bold",
            bbox=dict(
                boxstyle=f"circle,pad=0.15",
                fc="green" if i == ordered_state[0] else "red",
            ),
        )

    plt.tight_layout()
    plt.show()


def plotTSPProblem(problem):
    fig, ax = plt.subplots(1)  # Prepare 2 plots
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    for i, (x, y) in enumerate(problem.coords):
        ax.scatter(x, y, s=200, c='cornflowerblue')  # plot A
    node_labels = {k: str(v) for k, v in enumerate(string.ascii_uppercase) if k < len(problem.source_graph.nodes)}
    for i in node_labels.keys():
        x, y = problem.coords[i]
        plt.text(x, y, node_labels[i], ha="center", va="center", c='white', fontweight='bold',
                 bbox=dict(boxstyle=f"circle,pad=0.15", fc='cornflowerblue'))

    plt.tight_layout()
    plt.show()