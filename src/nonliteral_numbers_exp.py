from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import REPO_PATH


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Data is from Nonliteral understanding of number words, Kao et al. (2014)
    # Can be found at: https://cocolab.stanford.edu/jkao_experiment_data_and_materials/

    # Human prior: P(s|c) is from experiment 3a - s is the state of the world (i.e., the price of the object) and c is the context (i.e., the object)
    # Why normalized? According to the Kao paper (Pager 6 - Under Experiment 3a), the authors re-normalize slider scores so that things sum to 1
    df_prior = pd.read_csv(
        REPO_PATH
        / "data"
        / "exp_data"
        / "nonliteral_numbers"
        / "experiment3a-normalized.csv"
    )
    # Average across all human experimenters
    df_prior = (
        df_prior.groupby(["domain", "state"])["stateProb"]
        .mean()
        .reset_index(name="P(s|c)")
    )
    df_prior = df_prior.rename(columns={"domain": "context"})
    # Sanity check: There are 3 contexts, and 10 states
    # For each context, the sum of the probabilities over all states should be 1 (so 3 times it should be 3)
    assert (
        np.abs(df_prior.groupby("context")["P(s|c)"].sum().sum() - 3.0) < 1e-5
    ), "P(s|c) does not sum to 3.0"

    # Human posterior: P(s|c,u) is from experiment 1 - s is the state of the world (i.e., the price of the object), c is the context (i.e., the object), and u is the utterance (e.g., "The X costs Y$")
    # Why normalized? According to the Kao paper (Pager 6 - Under Experiment 1), the authors normalize by
    # 1. Applying a power law distribution to decrease the human bias of picking extreme values and,
    # 2. Re-normalize so that things sum to 1
    df_human = pd.read_csv(
        REPO_PATH / "data" / "exp_data" / "nonliteral_numbers" / "experiment1-normalized.csv"
    )
    # Compute standard deviation for each averaged group
    std_dev_col = (
        df_human.groupby(["domain", "utterance", "state"])["stateProb"]
        .std()
        .reset_index(name="StdevPHuman(s,c,u)")["StdevPHuman(s,c,u)"]
    )
    std_err_col = std_dev_col / np.sqrt(len(std_dev_col))
    # Average across all human experimenters
    df_human = (
        df_human.groupby(["domain", "utterance", "state"])["stateProb"]
        .mean()
        .reset_index(name="PHuman(s|c,u)")
    )
    df_human = df_human.rename(columns={"domain": "context"})
    df_human["SterrPHuman(s,c,u)"] = std_err_col
    # Sanity check: There are 3 contexts, 10 utterances, and 10 states
    # For each context-utterance pair, the sum of the probabilities over all states should be 1 (so 30 times it should be 30)
    assert (
        np.abs(
            df_human.groupby(["context", "utterance"])["PHuman(s|c,u)"].sum().sum()
            - 30.0
        )
        < 1e-5
    ), "P(s|c,u) does not sum to 30.0"

    # Affect-aware RSA pragmatic listener probabilities: PL1AA-RSA(s|c,u) is from model-predictions.csv - s is the state of the world (i.e., the price of the object), c is the context (i.e., the object), and u is the utterance (e.g., "The X costs Y$")
    df_aarsa = pd.read_csv(
        REPO_PATH / "data" / "exp_data" / "nonliteral_numbers" / "model-predictions.csv"
    )
    # Sanity check: There are 3 contexts, 10 utterances, and 10 states
    # For each context-utterance pair, the sum of the probabilities over all states should be 1 (so 30 times it should be 30)
    df_aarsa = (
        df_aarsa.groupby(["domain", "utterance", "state"])["probability"]
        .sum()
        .reset_index(name="PL1-AARSA(s|c,u)")
    )  # normalized over affect (affect is a separate column)
    df_aarsa = df_aarsa.rename(columns={"domain": "context"})
    assert (
        np.abs(df_aarsa.groupby("context")["PL1-AARSA(s|c,u)"].sum().sum() - 30.0)
        < 1e-5
    ), "P(s|c,u) does not sum to 30.0"
    return df_prior, df_human, df_aarsa


def rhetorical_strategy_function(
    rhetorical_strategy: Literal["halo", "hyperbole", "sincere", "understatement"],
    utterance: Literal[50, 51, 500, 501, 1000, 1001, 5000, 50001, 10000, 10001],
    state: Literal[50, 51, 500, 501, 1000, 1001, 5000, 50001, 10000, 10001],
    context: Literal["electric kettle", "laptop", "watch"],
    epsilon: float = 0.001,
) -> {0, 1, 0.001}:
    # Defines the rhetorical strategy function
    # NOTE: The context is ignored here
    if (
        rhetorical_strategy == "halo"
        and abs(utterance - state) == 1
        and utterance % 10 == 0
    ):
        return 1
    elif rhetorical_strategy == "hyperbole" and utterance - state > 10:
        return 1
    elif rhetorical_strategy == "understatement" and state - utterance > 10:
        return 1
    elif rhetorical_strategy == "sincere" and state == utterance:
        return 1
    else:
        return epsilon  # to avoid any kind of division by zero


def get_rs_prior(
    rhetorical_strategy: Literal["halo", "hyperbole", "sincere", "understatement"],
    context: Literal["electric kettle", "laptop", "watch"],
    utterance: Literal[50, 51, 500, 501, 1000, 1001, 5000, 50001, 10000, 10001],
    prior_type: Literal["uniform"],
) -> float:
    # Computes P(r|c,u) for the marginalization for L0/L1
    # NOTE: Only uniform is implemented as we did not run human elicitation experiments for this
    if prior_type == "uniform":
        # P(r|c,u) is uniform
        return 1 / 4
    else:
        raise ValueError(f"Unknown prior type: {prior_type}. Must be 'uniform'.")


def rsa_two(
    utterances: list[str],
    states: list[str],
    contexts: list[str],
    rhetorical_strategies: list[str],
    df_prior: pd.DataFrame,
    rs_prior_type: str = "uniform",
    alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Computes PL0(s|c,u) and PL1(s|c,u) for the (RSA)^2 model
    # Create mappings for rs, utt, state, contexts to indices
    utt_mapping = {utt: i for i, utt in enumerate(utterances)}
    state_mapping = {state: i for i, state in enumerate(states)}
    context_mapping = {context: i for i, context in enumerate(contexts)}
    # Create arrays for priors
    state_prior = np.array(
        [
            [
                df_prior[(df_prior["context"] == c) & (df_prior["state"] == s)][
                    "P(s|c)"
                ].values[0]
                for s in states
            ]
            for c in contexts
        ]
    )
    assert state_prior.shape == (3, 10), f"{state_prior.shape} != {(3, 10)}"
    rs_prior = np.array(
        [
            [
                [
                    get_rs_prior(
                        rhetorical_strategy=r,
                        context=c,
                        utterance=u,
                        prior_type=rs_prior_type,
                    )
                    for r in rhetorical_strategies
                ]
                for u in utterances
            ]
            for c in contexts
        ]
    )
    assert rs_prior.shape == (3, 10, 4), f"{rs_prior.shape} != {(3, 10, 4)}"
    # Create rsf array
    rsf = np.array(
        [
            [
                [
                    [
                        rhetorical_strategy_function(
                            rhetorical_strategy=rs, utterance=u, state=s, context=c
                        )
                        for s in states
                    ]
                    for u in utterances
                ]
                for c in contexts
            ]
            for rs in rhetorical_strategies
        ]
    )
    assert rsf.shape == (4, 3, 10, 10), f"{rsf.shape} != {(4, 3, 10, 10)}"
    # Compute L0, unnormalized
    l0 = rsf * state_prior[None, :, None, :]
    pl0 = l0 / np.sum(l0, axis=-1, keepdims=True)
    assert pl0.shape == (4, 3, 10, 10), f"{pl0.shape} != {(4, 3, 10, 10)}"
    assert (
        np.abs(pl0.sum() - 4 * 3 * 10) < 0.01
    ), f"PL0(s|c,u,r) does not sum to {4*3*10}. It sums to {pl0.sum()}"
    # Compute marginalized PL0(s|c,u) with P(r|c,u) as defined in our paper
    pl0_marg = (
        np.swapaxes(pl0, 0, -1) * np.swapaxes(rs_prior, 0, 1)[:, :, None, :]
    )  # place the meaning in the first axis and the rs in the last axis
    pl0_marg = pl0_marg.sum(
        axis=-1
    )  # sum across the products with rhetorical strategies
    pl0_marg = np.swapaxes(pl0_marg, 0, -1)  # swap back the meaning to the last axis
    assert np.all(np.abs(pl0_marg.sum(axis=-1) - np.ones((10, 3))) <= 0.01)
    # Convert to a df
    pl0_marg_df = pd.DataFrame(
        data=[
            [c, u, s, pl0_marg[utt_mapping[u], context_mapping[c], state_mapping[s]]]
            for c, u, s in product(contexts, utterances, states)
        ],
        columns=["context", "utterance", "state", "PL0-RSA^2(s|c,u)"],
    )
    # Apply RSA transpose for the speaker probabilities
    s1 = (np.swapaxes(pl0, -2, -1)) ** alpha
    ps1 = s1 / np.sum(
        s1, axis=-1, keepdims=True
    )  # This is PS1(u|c,s,r) as defined in our paper
    assert ps1.shape == (4, 3, 10, 10), f"{ps1.shape} != {(4, 3, 10, 10)}"
    # Compute L1, unnormalized
    l1 = np.swapaxes(pl0, -2, -1) * state_prior[None, :, None, :]
    pl1 = l1 / np.sum(l1, axis=-1, keepdims=True)  # This is PL1(s|c,u,r)
    assert pl1.shape == (4, 3, 10, 10)
    # Compute marginalized PL1(s|c,u) with P(r|c,u) as defined in our paper
    pl1_marg = (
        np.swapaxes(pl1, 0, -1) * np.swapaxes(rs_prior, 0, 1)[:, :, None, :]
    )  # place the meaning in the first axis and the rs in the last axis
    pl1_marg = pl1_marg.sum(
        axis=-1
    )  # sum across the products with rhetorical strategies
    pl1_marg = np.swapaxes(pl1_marg, 0, -1)  # swap back the meaning to the last axis
    assert np.all(np.abs(pl1_marg.sum(axis=-1) - np.ones((10, 3))) <= 0.01)
    # Convert to a df
    pl1_marg_df = pd.DataFrame(
        data=[
            [c, u, s, pl1_marg[utt_mapping[u], context_mapping[c], state_mapping[s]]]
            for c, u, s in product(contexts, utterances, states)
        ],
        columns=["context", "utterance", "state", "PL1-RSA^2(s|c,u)"],
    )
    return pl0_marg_df, pl1_marg_df


def compute_metrics(
    df_human: pd.DataFrame,
    human_column: str,
    model_predictions: dict[str, pd.DataFrame],
    model_names: dict[str, str],
    merge_on: list[str],
) -> pd.DataFrame:
    # Computes all the metrics reported in the paper
    assert len(model_predictions) == len(model_names)
    assert set(model_names.keys()) == set(model_predictions.keys())
    metric_values = {}
    for model_name, model in model_predictions.items():
        # Merge the dataframes
        merge = pd.merge(
            df_human,
            model,
            on=merge_on,
        )
        # Get PL1 column
        prob_column = model_names[model_name]
        # Compute the metrics
        metric_values[model_name] = {
            "mean_abs_diff": np.abs(
                merge[prob_column].values - merge[human_column].values
            ).mean(),
            "max_abs_diff": np.abs(
                merge[prob_column].values - merge[human_column].values
            ).max(),
        }
    return pd.DataFrame(
        data=metric_values,
        index=[
            "mean_abs_diff",
            "max_abs_diff",
            "mean_rel_abs_diff",
            "stdev_abs_diff",
        ],
    )


def plot_probabilities_by_context(df: pd.DataFrame) -> dict[str, plt.Figure]:
    # Function generated with ChatGPT
    # Columns to plot
    prob_cols = [
        "PHuman(s|c,u)",
        "PL1-AARSA(s|c,u)",
        "PL0-RSA^2(s|c,u)",
        "PL1-RSA^2(s|c,u)",
    ]
    # Only standard deviation for human experimenters elicited probabilities
    std_err_column = "SterrPHuman(s,c,u)"
    col_to_label = {
        "PHuman(s|c,u)": "$P(m|c,u)$ - Human",
        "PL1-AARSA(s|c,u)": "$P_{{L_1}}(m|c,u)$ - Affect-Aware RSA",
        "PL0-RSA^2(s|c,u)": "$P_{{L_0}}(m|c,u)$ - RSA$^2$",
        "PL1-RSA^2(s|c,u)": "$P_{{L_1}}(m|c,u)$ - RSA$^2$",
    }
    contexts = df["context"].unique()
    figs = {}
    # Create 3 plots (one for each context) with 10 rows (one for each utterance)
    # Each plot will have groups of 4 bars (one for each probability) for each state
    # The bars for human probabilities will have error bars
    for context in contexts:
        df_context = df[df["context"] == context]
        utterances = df_context["utterance"].unique()
        num_utterances = len(utterances)

        # Determine the split
        split_index = num_utterances // 2
        utterance_splits = [utterances[:split_index], utterances[split_index:]]

        for split_idx, utterance_group in enumerate(utterance_splits):
            num_utterances = len(utterance_group)
            fig, axs = plt.subplots(
                num_utterances,
                1,
                figsize=(8, 4 * num_utterances),
                constrained_layout=True,
            )
            if num_utterances == 1:
                axs = [axs]  # make it iterable

            for i, utterance in enumerate(sorted(utterance_group)):
                df_plot = df_context[df_context["utterance"] == utterance].sort_values(
                    by="state"
                )
                x = df_plot["state"].astype(str)
                width = 0.2
                x_pos = range(len(x))

                for j, col in enumerate(prob_cols):
                    bar_positions = [pos + width * j for pos in x_pos]
                    label = col_to_label[col] if i == 0 else None
                    if col == "PHuman(s|c,u)" and std_err_column in df_plot.columns:
                        axs[i].bar(
                            bar_positions,
                            df_plot[col],
                            width=width,
                            yerr=df_plot[std_err_column],
                            capsize=4,
                            label=label,
                        )
                    else:
                        axs[i].bar(
                            bar_positions, df_plot[col], width=width, label=label
                        )
                        # add grid
                        axs[i].grid(axis="y", linestyle="--", alpha=0.7)

                axs[i].set_title(
                    f'Meaning Probabilities for "The {context} costs {utterance}\$."',
                    fontsize=18,
                )
                axs[i].set_xticks([pos + width * 1.5 for pos in x_pos])
                axs[i].set_xticklabels(x, rotation=45, fontsize=14)
                axs[i].set_ylabel("Probability", fontsize=16)
                axs[i].set_xlabel("State", fontsize=16)
                axs[i].tick_params(axis="y", labelsize=14)

            axs[0].legend(loc="best", fontsize=12)
            figs[f"state_probabilities_context={context}_splitidx={split_idx}"] = fig

    return figs


def generate_plots(df: pd.DataFrame) -> dict[str, plt.Figure]:
    # Generates the plots for the paper
    figs = {}
    figs.update(plot_probabilities_by_context(df))
    return figs


def main():
    ### Load data ###
    df_prior, df_human, df_aarsa = load_data()
    ### Define variables and hyperparameters ###
    rhetorical_strategies = [
        "sincere",
        "hyperbole",
        "understatement",
        "halo",
    ]
    states = df_prior["state"].unique()
    utterances = states
    contexts = df_prior["context"].unique()
    rs_prior_type = "uniform"
    alpha = 1.0
    ### Run (RSA)^2 model ###
    pl0_marg_df, pl1_marg_df = rsa_two(
        utterances=utterances,
        states=states,
        contexts=contexts,
        rhetorical_strategies=rhetorical_strategies,
        df_prior=df_prior,
        rs_prior_type=rs_prior_type,
        alpha=alpha,
    )
    ### Compute metrics ###
    metrics = compute_metrics(
        df_human=df_human,
        human_column="PHuman(s|c,u)",
        model_predictions={
            "PL1-AARSA": df_aarsa,
            "PL0-RSA^2": pl0_marg_df,
            "PL1-RSA^2": pl1_marg_df,
        },
        model_names={
            "PL1-AARSA": "PL1-AARSA(s|c,u)",
            "PL0-RSA^2": "PL0-RSA^2(s|c,u)",
            "PL1-RSA^2": "PL1-RSA^2(s|c,u)",
        },
        merge_on=["context", "utterance", "state"],
    )
    ### Generate plots ###
    # Merge all dataframes
    merge = df_human
    for df in [df_aarsa, pl0_marg_df, pl1_marg_df]:
        merge = pd.merge(
            merge,
            df,
            on=["context", "utterance", "state"],
        )
    plots = generate_plots(df=merge)
    ### Save everything ###
    pl0_marg_df.to_csv(
        REPO_PATH
        / "data"
        / "nonliteral_number_exps"
        / "pl0_marg_df.csv",
        index=False,
    )
    pl1_marg_df.to_csv(
        REPO_PATH
        / "data"
        / "nonliteral_number_exps"
        / "pl1_marg_df.csv",
        index=False,
    )
    merge.to_csv(
        REPO_PATH / "data" / "nonliteral_number_exps" / "meaning_probabilities_merged.csv",
        index=False,
    )
    metrics.to_csv(
        REPO_PATH / "data" / "nonliteral_number_exps" / "metrics.csv",
        index=True,
    )
    for context, fig in plots.items():
        fig.savefig(
            REPO_PATH
            / "data"
            / "nonliteral_number_exps"
            / f"{context}.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
