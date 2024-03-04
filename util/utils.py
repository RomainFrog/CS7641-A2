import mlrose_hiive
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator
from mlrose_hiive import SARunner, GARunner, MIMICRunner, RHCRunner
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import string
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
import ucimlrepo

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
    hp_mean, hp_std, optimal_HP_dict, hyperparameter, runnner_name, problem_name="", x_axis="Iteration", y_lim=(-2000, -1100), show_std=True
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
        print(f"HP: {hyperparameter}, {type(hyperparameter)}")
        if type(hyperparameter) == str:
            plt.plot(
                temp_df[x_axis],
                temp_df["Fitness"],
                label=f"{hyperparameter} = {H_value:.2f}",
            )
        else:
            plt.plot(
                temp_df[x_axis],
                temp_df["Fitness"],
                label=f"{hyperparameter} = {H_value:.2f}",
            )
        if show_std:
            plt.fill_between(
                temp_df[x_axis],
                temp_df["Fitness"] - temp_df_std["Fitness"],
                temp_df["Fitness"] + temp_df_std["Fitness"],
                alpha=0.2,
            )

        print(f'{hyperparameter} = {H_value} Fitness: {temp_df["Fitness"].max()} (+/- {temp_df_std["Fitness"].max()}), Time: {temp_df["Time"].max()}')

    # Add a hline in red for the max fitness
    plt.axhline(y=189, color="r", linestyle="--", label="Max Fitness (189.0)")
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
    print(heatmap_data)
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


def get_kcolor(difficulty):
    if difficulty == "easy":
        return MaxKColorGenerator().generate(seed=42, number_of_nodes=20, max_connections_per_node=2, max_colors=3, maximize=True)
    elif difficulty == "medium":
        return MaxKColorGenerator().generate(seed=42, number_of_nodes=30, max_connections_per_node=3, max_colors=4, maximize=True)
    elif difficulty == "hard":
        return MaxKColorGenerator().generate(seed=42, number_of_nodes=50, max_connections_per_node=6, max_colors=4, maximize=True)
    elif difficulty == "extreme":
        return MaxKColorGenerator().generate(seed=42, number_of_nodes=75, max_connections_per_node=6, max_colors=4, maximize=True)
    else:
        raise ValueError(f"Difficulty {difficulty} not found")
    

def get_four_peaks(difficulty):

    fitness  = mlrose_hiive.FourPeaks(t_pct = 0.099)
    if difficulty == "easy":
        return mlrose_hiive.DiscreteOpt(length = 30, fitness_fn = fitness, maximize=True, max_val=2)
    elif difficulty == "medium":
        return mlrose_hiive.DiscreteOpt(length = 50, fitness_fn = fitness, maximize=True, max_val=2)
    elif difficulty == "hard":
        return mlrose_hiive.DiscreteOpt(length = 100, fitness_fn = fitness, maximize=True, max_val=2)
    else:
        raise ValueError(f"Difficulty {difficulty} not found")

def get_perf_stats(results, param1, param2):

    mmc_hp_mean, mmc_hp_std, mmc_hp_min, mmc_hp_max = runner_results_to_stats(results)

    mmc_optimal_HP = get_optimal_hyperparameters(mmc_hp_mean, [param1, param2])


    best_run = mmc_hp_mean[(mmc_hp_mean[param1] == mmc_optimal_HP[param1]) & (mmc_hp_mean[param2] == mmc_optimal_HP[param2])]
    best_run_max = mmc_hp_max[(mmc_hp_max[param1] == mmc_optimal_HP[param1]) & (mmc_hp_max[param2] == mmc_optimal_HP[param2])]
    print(mmc_optimal_HP)
    print(f'Mean Fitness: {best_run["Fitness"].max()}')
    print(f'MAX fitness: {best_run_max["Fitness"].max()} ')
    print(f'FEvals: {best_run["FEvals"].max()}')
    print(f'Time: {best_run["Time"].max()}')

    # Select the run that reaches 176 of fitness
    MAX_FITNESS = mmc_hp_max['Fitness'].max()
    best_runs = mmc_hp_max[(mmc_hp_max['Fitness'] == MAX_FITNESS)]

    # select the best run with the lowest number of fevals
    best_run = best_runs[(best_runs['FEvals'] == best_runs['FEvals'].min())]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_rows", None)
    print(best_run)



############################################################################################################
#
# NN related functions
#
############################################################################################################
    
from sklearn.metrics import accuracy_score, log_loss

def fit_multiple_seeds(model, X_train, y_train, X_test, y_test, seeds):
    scores = []
    losses = []
    for seed in tqdm(seeds):
        model.set_params(random_state=seed)
        model.fit(X_train, y_train)
        losses.append(copy.deepcopy(model.fitness_curve))
        y_test_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_test_pred))
    return losses, scores



def train_model(model, X_train, y_train, X_val, y_val, n_epochs=500):
    train_losses = []
    val_losses = []
    train_score = []
    validation_score = []
    for i in tqdm(range(n_epochs)):
        # Fit for one iteration
        model.partial_fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_score.append(accuracy_score(y_train, y_train_pred))

        y_val_pred = model.predict(X_val)
        validation_score.append(accuracy_score(y_val, y_val_pred))

        train_losses.append(model.fitness_curve[0][0])

        # Validation loss
        y_val_probas = model.predicted_probs
        val_losses.append(log_loss(y_val, y_val_probas))


    return train_losses, val_losses, train_score, validation_score


def train_model_SA(model, X_train, y_train, X_val, y_val, n_epochs=500):
    train_losses = []
    val_losses = []
    train_score = []
    validation_score = []
    init_temp= model.schedule.init_temp
    for i in tqdm(range(n_epochs)):
        # Fit for one iteration
        if i!=0:
             if init_temp> 0.001:
                #init_temp= init_temp - (0.001 * i)
                init_temp= init_temp*np.exp(-1*0.005*i)
                model.schedule.init_temp=init_temp
        model.partial_fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_score.append(accuracy_score(y_train, y_train_pred))

        y_val_pred = model.predict(X_val)
        validation_score.append(accuracy_score(y_val, y_val_pred))

        train_losses.append(model.fitness_curve[0][0])

        # Validation loss
        y_val_probas = model.predicted_probs
        val_losses.append(log_loss(y_val, y_val_probas))

        

    return train_losses, val_losses, train_score, validation_score

# def train_model_multi_seed(model, X_train, y_train, X_val, y_val, n_epochs=500, seeds=[42]):
#     train_losses = []
#     val_losses = []
#     train_score = []
#     validation_score = []

#     for seed in seeds:
#         current_model = copy.deepcopy(model)
#         current_model.set_params(random_state=seed)
#         results = train_model(current_model, X_train, y_train, X_val, y_val, n_epochs)
#         train_losses.append(results[0])
#         val_losses.append(results[1])
#         train_score.append(results[2])
#         validation_score.append(results[3])
#         # Reset the weights
#         # model.fitted_weights = []

#     return train_losses, val_losses, train_score, validation_score










def load_hand_written_digits():
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    digits = fetch_ucirepo(id=80) 
    
    # data (as pandas dataframes) 
    X = digits.data.features 
    y = digits.data.targets 
    
    X['target'] = y

    return X



def load_titanic():
    """
    Load titanic dataset from seaborn
    """
    import seaborn as sns

    df = sns.load_dataset('titanic')
    # remove the deck column
    df = df.drop('deck', axis=1)
    # remove people who no age
    df = df.dropna(subset=['age'])
    # drop all remaining missing values
    df = df.dropna()
    # rename alive column to target and put it at the end
    df = df.rename(columns={'alive': 'target'})
    # convert target to 0 and 1
    df['target'] = df['target'].map({'yes': 1, 'no': 0})
    # convert alone to 0 and 1
    df['alone'] = df['alone'].map({True: 1, False: 0})
    # drop adult_male column
    df = df.drop('adult_male', axis=1)
    # drop class column
    df = df.drop('class', axis=1)
    # drop who column
    df = df.drop('who', axis=1)
    # create adult column based on age (int)
    df['adult'] = df['age'].apply(lambda x: 1 if x >= 18 else 0)
    # drop embarked column
    df = df.drop('embarked', axis=1)

    # translate embarked_town to one hot encoding (int)
    embark_town = pd.get_dummies(df['embark_town'])
    df = df.join(embark_town)
    df = df.drop('embark_town', axis=1)
    # cast Cherbourg, Queenstown, Southampton to int
    df['Cherbourg'] = df['Cherbourg'].astype(int)
    df['Queenstown'] = df['Queenstown'].astype(int)
    df['Southampton'] = df['Southampton'].astype(int)

    # translate sex to one hot encoding (int)
    sex = pd.get_dummies(df['sex'])
    df = df.join(sex)
    df = df.drop('sex', axis=1)

    # cast sex to int
    df['male'] = df['male'].astype(int)
    df['female'] = df['female'].astype(int)

    # drop survived column
    df = df.drop('survived', axis=1)

    # reset index
    df = df.reset_index(drop=True)

    # 
    return df