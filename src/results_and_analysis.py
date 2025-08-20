import os
from ast import literal_eval
from collections import OrderedDict
from itertools import product
from typing import Literal
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf

from src import REPO_PATH

listeners = ["L0", "L1"]

run_name_to_model = {
    "llm_rsa": "llm_rsa",
    "rsa_two_p_r_c_u": "rsa_two",
    "rsa_two_indicator_r_c_u": "rsa_two",
}

# run_name_to_model = {
#     "rsc_rsa": "rsc_rsa",
# }


def populate_average_probabilities_table(average_probabilities: pd.DataFrame) -> str:
    table_to_fill = r"""
        \toprule
        Model & $L_i$ & Correct \(\uparrow\) & Incorrect \(\downarrow\) & Distractor \(\downarrow\) \\
        \midrule
        LLM RSA & $L_0$ & {} & {} & {} \\
        & $L_1$ & {} & {} & {} \\
        \cmidrule{{2-5}}
        \makecell{{LLM \rsatwo\\with $P_N(r|c,u)$}} & $L_0$ & {} & {} & {} \\
        & $L_1$ & {} & {} & {} \\
        \cmidrule{{2-5}}
        \makecell{{LLM \rsatwo\\with $I(r|c,u)$}} & $L_0$ & {} & {} & {} \\
        & $L_1$ & {} & {} & {} \\
        \bottomrule
    """
    table_top = r"""
    \begin{table}[h]
        \centering
        \resizebox{\linewidth}{!}{
        \begin{tabular}{llccc}
    """
    table_bot = r"""
        \end{tabular}
        }
        \caption{Average listener probabilities, $P_{L_i}(m|c,u), i = 0,1$ for the correct, incorrect and distractor intended meanings on the test set averaged across all 50 scenarios.}
        \label{tab:meaning_scores}
    \end{table}
    """
    # Get the max for correct
    average_probabilities["Correct"] = average_probabilities["Correct"].apply(
        lambda x: (
            r"\textbf{" + f"{x:.2f}" + r"}"
            if x == average_probabilities["Correct"].max()
            else f"{x:.2f}"
        )
    )
    # Get the min for incorrect and distractor
    for col in ["Incorrect", "Distractor"]:
        average_probabilities[col] = average_probabilities[col].apply(
            lambda x: (
                r"\textbf{" + f"{x:.2f}" + r"}"
                if x == average_probabilities[col].min()
                else f"{x:.2f}"
            )
        )
    # Create the table
    probabilities = [
        val for vals in average_probabilities.itertuples() for val in vals[1:]
    ]
    table_to_fill = table_to_fill.format(
        *probabilities,
    )
    table_str = table_top + table_to_fill + table_bot
    return table_str


def model_name_map(listener: str, model: str) -> str:
    return f"P{listener}_{model}(m|c,u)"


def load_outputs(
    run_output_path: str, outer_key: str, inner_key: str = None
) -> pd.DataFrame:
    # Returns a DataFrame with {m1: p1, m2: p2, ...} for each utterance
    # Load outputs with pandas lines
    df = pd.read_json(run_output_path, lines=True)
    # Get the correct meaning probability
    df["raw_extracted_probabilities"] = df[outer_key].apply(
        lambda x: x[inner_key] if inner_key is not None else x
    )
    # Get the meaning distribution for the original utterance
    df["raw_extracted_probabilities_original_utterance"] = df.apply(
        lambda row: [
            prob_dict
            for keys, prob_dict in row["raw_extracted_probabilities"].items()
            if literal_eval(keys)[1] == row["original_utterance"]
        ][0],
        axis=1,
    )
    # Get the meaning types rather than their verbalizations
    df["meaning_to_meaning_type"] = df.apply(
        lambda row: {elt["meaning"]: elt["meaning_type"] for elt in row["meanings"]},
        axis=1,
    )
    # Re-index the probabilities to the meaning types
    df["processed_probabilities_original_utterance"] = df.apply(
        lambda row: {
            row["meaning_to_meaning_type"][m]: prob
            for m, prob in row["raw_extracted_probabilities_original_utterance"].items()
        },
        axis=1,
    )
    return pd.DataFrame(
        df["processed_probabilities_original_utterance"].values,
        columns=["P(m_type|c,u_og)"],
    )


def compute_average_probabilities(
    run_output: pd.DataFrame,
    agg_type: Literal[
        "correct_incorrect_distractor", "literal_nonliteral_associate_nonseq"
    ],
) -> dict[str]:
    # Returns a dictionary with the average probabilities
    # of the form based on agg_type
    # agg_type = correct_incorrect_distractor {correct: p_correct, incorrect: p_incorrect, distractor: p_distractor}
    # agg_type = literal_nonliteral_nonseq_associate {NonLiteral: p_NonLiteral, Literal: p_Literal, NonSequitur: p_NonSequitur, Associate: p_Associate}
    meaning_types = {}
    if agg_type == "correct_incorrect_distractor":
        meaning_mapping = {
            "Correct": ["CorrectLiteral", "CorrectNonLiteral"],
            "Incorrect": ["IncorrectLiteral", "IncorrectNonLiteral"],
            "Distractor": ["IncorrectNonSequitur"],
        }
    elif agg_type == "literal_nonliteral_associate_nonseq":
        meaning_mapping = {
            "NonLiteral": ["CorrectNonLiteral", "IncorrectNonLiteral"],
            "Literal": ["CorrectLiteral", "IncorrectLiteral"],
            "Lexical Overlap": ["IncorrectAssociate"],
            "NonSequitur": ["IncorrectNonSequitur"],
        }
    for meaning_category, meaning_types in meaning_mapping.items():
        pd.options.mode.chained_assignment = None  # default='warn'
        run_output.loc[:, meaning_category] = run_output["P(m_type|c,u_og)"].apply(
            lambda x: np.sum(
                [
                    prob if prob is not None else np.nan
                    for meaning_type, prob in x.items()
                    if meaning_type in meaning_types
                ]
            )
        )
        if np.any(run_output[meaning_category].isna()):
            warnings.warn(
                f"NaN values found in {meaning_category} probabilities."
                + f"They are at index: {run_output[run_output[meaning_category].isna()].index.tolist()}"
            )
    # Temporary fix: Use np.nanmean to avoid NaN issues
    return OrderedDict({k: np.nanmean(run_output[k]) for k in meaning_mapping.keys()})


def generate_listener_meaning_plot(ironic_split: dict, literal_split: dict) -> None:
    # Generated by Gemini
    # Hardcoded model names (order matters for legend and colors)
    model_names_mapping = {
        ("llm_rsa", "L0"): r"$L_0$ - LLM RSA ",
        ("llm_rsa", "L1"): r"$L_1$ - LLM RSA",
        ("rsa_two_p_r_c_u", "L0"): r"$L_0$ - LLM (RSA)$^2$ with $P_N(r|c,u)$",
        ("rsa_two_p_r_c_u", "L1"): r"$L_1$ - LLM (RSA)$^2$ with $P_N(r|c,u)$",
        ("rsa_two_indicator_r_c_u", "L0"): r"$L_0$ - LLM (RSA)$^2$ with $I(r|c,u)$",
        ("rsa_two_indicator_r_c_u", "L1"): r"$L_1$ - LLM (RSA)$^2$ with $I(r|c,u)$",
    }

    # Hardcoded colors matched from the example image
    model_colors_ordered = {
        ("llm_rsa", "L0"): "#9467bd",
        ("llm_rsa", "L1"): "#ff7f0e",
        ("rsa_two_p_r_c_u", "L0"): "#2ca02c",
        ("rsa_two_p_r_c_u", "L1"): "#d62728",
        ("rsa_two_indicator_r_c_u", "L0"): "#8c564b",
        ("rsa_two_indicator_r_c_u", "L1"): "#e377c2",
    }

    meaning_types = ["NonLiteral", "Literal", "Lexical\nOverlap", "Non-Sequitur"]
    n_models = len(model_colors_ordered)
    n_meaning_types = len(meaning_types)
    assert (
        len(ironic_split) == len(literal_split) == n_models
    ), f"Expected {n_models} models, got {len(ironic_split)} and {len(literal_split)}"

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # Titles for the subplots
    subplot_titles = [
        "Scenarios where the Intended Meaning is NonLiteral",
        "Scenarios where the Intended Meaning is Literal",
    ]
    datasets = [ironic_split, literal_split]

    # Width of a single bar in a group
    bar_width = 0.8 / n_models  # Total width for all bars in a group is 0.8

    # Iterate over the two subplots (left and right)
    for i, ax in enumerate(axes):
        current_data = datasets[i]
        ax.set_title(subplot_titles[i], fontsize=16, pad=10)
        if i == 0:
            ax.set_ylabel("Probability", fontsize=14)
        ax.set_xlabel("Meaning Type", fontsize=14)
        ax.set_ylim(0, 1.02)  # Set y-axis limits, slight padding at top
        ax.set_yticks(np.arange(0, 1.1, 0.2))  # Y-axis ticks from 0.0 to 1.0

        x_indices = np.arange(n_meaning_types)  # Base x positions for groups

        # Plot bars for each model
        for idx, (model_key, model_name) in enumerate(model_names_mapping.items()):
            probabilities = list(current_data[model_key].values())

            # Calculate the offset for each bar within its group to center the group
            offset = (idx - n_models / 2 + 0.5) * bar_width
            bar_positions = x_indices + offset

            # Add bar labels only for the right plot to generate the legend once
            label_for_legend = model_name if i == 1 else ""

            ax.bar(
                bar_positions,
                probabilities,
                bar_width,
                label=label_for_legend,
                color=model_colors_ordered[model_key],
                edgecolor="black",
                linewidth=0.7,
            )  # Slightly thicker edge

        # Set x-axis ticks and labels
        ax.set_xticks(x_indices)
        ax.set_xticklabels(meaning_types, fontsize=12)
        ax.tick_params(axis="y", labelsize=12)

        # Add horizontal grid lines
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)  # Ensure grid is drawn behind bars

    # Add a main title to the figure
    fig.suptitle(
        r"Average Listener Meaning Probability $P_{L_i}(m|c, u)$ on the Test Set by Scenario Type",
        fontsize=16,
        y=0.92,
    )  # Adjust y for suptitle position

    # Add legend to the right plot (axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:  # Only add legend if there are items to show
        axes[1].legend(
            handles,
            labels,
            loc="upper right",
            fontsize=12,
            title_fontsize=12,
            frameon=True,
            edgecolor="darkgrey",
        )

    # Adjust layout to prevent overlapping elements
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])  # rect=[left, bottom, right, top]

    return fig


def generate_tables(config: DictConfig) -> None:
    if "rsa_runs" not in config:
        print("No key 'rsa_runs' found in config, skipping table generation.")
        return
    print("Generating tables.")
    for run_name, run_dir in config["rsa_runs"].items():
        run_outputs = OrderedDict()
        # Load all the RSA model outputs
        for model_run_name, listener in product(run_name_to_model.keys(), listeners):
            print(
                f"Loading outputs for {run_name} run: {run_dir}, listener: {listener}, model: {model_run_name}"
            )
            model = run_name_to_model[model_run_name]
            # Load the outputs for the run
            run_output = load_outputs(
                run_output_path=REPO_PATH
                / run_dir
                / f"{listener}_{model_run_name}.jsonl",
                outer_key=f"{listener}",
                inner_key=model_name_map(listener, model),
            )
            run_outputs[(model_run_name, listener)] = run_output
        # Compute average probabilities
        average_probabilities = OrderedDict()
        for (model, listener), run_output in run_outputs.items():
            average_probabilities[(model, listener)] = compute_average_probabilities(
                run_output=run_output,
                agg_type="correct_incorrect_distractor",
            )
        average_probabilities = pd.DataFrame(
            average_probabilities
        ).T  # This has cols [correct, incorrect, distractor]
        average_probabilities.to_csv(
            REPO_PATH
            / config["output_dir"]
            / f"{run_name}_average_probabilities_correct_incorrect.csv"
        )
        # Populate table with results
        table_str = populate_average_probabilities_table(
            average_probabilities=average_probabilities
        )
        # Save the table to a file
        with open(
            REPO_PATH
            / config["output_dir"]
            / f"{run_name}_average_probabilities_table.tex",
            "w",
        ) as f:
            f.write(table_str)


def generate_plots(config: DictConfig) -> None:
    if "rsa_runs" not in config:
        print("No key 'rsa_runs' found in config, skipping plot generation.")
    print("Generating plots.")
    for run_name, run_dir in config["rsa_runs"].items():
        run_outputs = OrderedDict()
        # Load all the RSA model outputs
        for model_run_name, listener in product(run_name_to_model.keys(), listeners):
            model = run_name_to_model[model_run_name]
            print(
                f"Loading outputs for {run_name} run: {run_dir}, listener: {listener}, model: {model}"
            )
            # Load the outputs for the run
            run_output = load_outputs(
                run_output_path=REPO_PATH
                / run_dir
                / f"{listener}_{model_run_name}.jsonl",
                outer_key=f"{listener}",
                inner_key=model_name_map(listener, model),
            )
            run_outputs[(model_run_name, listener)] = run_output
        # Compute average probabilities
        average_probabilities = {
            "ironic": OrderedDict(),
            "literal": OrderedDict(),
        }
        for rs_type in average_probabilities.keys():
            for (model, listener), run_output in run_outputs.items():
                if rs_type == "ironic":
                    split = run_output.loc[:25]
                else:
                    split = run_output.loc[25:]
                average_probabilities[rs_type][(model, listener)] = (
                    compute_average_probabilities(
                        run_output=split,
                        agg_type="literal_nonliteral_associate_nonseq",
                    )
                )
        # Save tables
        for rs_type, avg_probs in average_probabilities.items():
            avg_probs_df = pd.DataFrame(avg_probs).T
            avg_probs_df.to_csv(
                REPO_PATH
                / config["output_dir"]
                / f"{run_name}_average_probabilities_{rs_type}.csv"
            )
        # Populate table with results
        fig = generate_listener_meaning_plot(
            ironic_split=average_probabilities["ironic"],
            literal_split=average_probabilities["literal"],
        )
        # Save the figure
        fig.savefig(
            REPO_PATH
            / config["output_dir"]
            / f"{run_name}_listener_meaning_probabilities.pdf",
            bbox_inches="tight",
            dpi=300,
        )


def populate_prior_ablations_table(prior_ablations_table: pd.DataFrame) -> str:
    top_of_table = r"""
        \begin{table}[h]
        \centering
        \resizebox{\linewidth}{!}{
            \begin{tabular}{llcccp{0.5cm}} % Changed p{1cm} to p{2.5cm} for better header wrapping
            \toprule
            Model & $L_i$ & w/o $P(m|c)$ & w/o $P(u|c)$ & w/o $P(r|c,u)$ \\
    """
    bot_of_table = r"""
            \bottomrule
            \end{tabular}

        }
        \caption{Ablations for all of our listeners $L_i, i = 0, 1$ on the test set. We ablate the meaning prior $P(m|c)$, the utterance prior $P(u|c)$ and the rhetorical strategy posterior $P(r|c,u)$. We report the average listener posterior probabilities of the correct meaning on the test (across all 50 scenarios) and the relative change with respect to the unablated model. Listeners with entries containing ``—'' indicate that they are not affect by that particular ablation. The $L_0$ LLM RSA listener is not included at all since it is not affected by any of the ablations.}
    \end{table}
    """
    table_to_fill = r"""
            \midrule
            LLM RSA & $L_1$ & {} & {} & {} \\
            \cmidrule{{2-5}}
            \makecell{{LLM \rsatwo\\with $P_N(r|c,u)$}} & $L_0$ & {} & {} & {} \\
                                                & $L_1$ & {} & {} & {} \\
            \cmidrule{{2-5}}
            \makecell{{LLM \rsatwo\\with $I(r|c,u)$}} & $L_0$ & {} & {} & {} \\
                                                & $L_1$ & {} & {} & {} \\
    """
    prior_ablations_table = prior_ablations_table[
        ["no_p_m_c", "no_p_u_c", "no_p_r_c_u"]
    ]
    values = [val for vals in prior_ablations_table.itertuples() for val in vals[1:]]
    table_to_fill = table_to_fill.format(
        *values,
    )
    # if have "--" then this becomes "+"
    table_to_fill = table_to_fill.replace("--", "+")
    table_str = top_of_table + table_to_fill + bot_of_table
    return table_str


def prior_ablations(config: DictConfig) -> None:
    if "prior_ablations" not in config:
        print("No key 'prior_ablations' found in config, skipping prior ablations.")
        return
    print("Generating prior ablations table.")
    # Create the prior ablations table
    prior_ablations_table = pd.DataFrame(
        [],
        columns=["unchanged", "no_p_m_c", "no_p_u_c", "no_p_r_c_u"],
        index=pd.MultiIndex.from_tuples(
            list(product(run_name_to_model.keys(), listeners))
        ),
    )
    # Populate the table
    for run_name, run_dir in config["prior_ablations"].items():
        print(f"Processing prior ablations for run: {run_name}, dir: {run_dir}")
        for model_run_name, listener in product(run_name_to_model.keys(), listeners):
            # Load the outputs for each model and listener
            model = run_name_to_model[model_run_name]
            if listener == "L0" and run_name in ["no_p_m_c", "no_p_u_c"]:
                prior_ablations_table.loc[(model, listener), run_name] = np.nan
                continue
            if model == "llm_rsa" and run_name == "no_p_r_c_u":
                prior_ablations_table.loc[(model, listener), run_name] = np.nan
                continue
            print(
                f"Loading outputs for {run_name} run: {run_dir}, listener: {listener}, model: {model}"
            )
            # Load the outputs for the run
            run_output = load_outputs(
                run_output_path=REPO_PATH
                / run_dir
                / f"{listener}_{model_run_name}.jsonl",
                outer_key=f"{listener}",
                inner_key=model_name_map(listener, model),
            )
            # Compute average probabilities
            average_probabilities = compute_average_probabilities(
                run_output=run_output,
                agg_type="correct_incorrect_distractor",
            )
            # Populate the table but only with correct probabilities
            prior_ablations_table.loc[(model_run_name, listener), run_name] = (
                average_probabilities["Correct"]
            )
    # Add relative changes to the table and format it with strings
    for col in ["no_p_m_c", "no_p_u_c", "no_p_r_c_u"]:
        prior_ablations_table[col] = prior_ablations_table.apply(
            lambda row: (
                f"{row[col]:.2f} (-{100 * (row['unchanged'] - row[col]) / row['unchanged']:.1f}\\%)"
                if not np.isnan(row[col])
                else "—"
            ),
            axis=1,
        )
    prior_ablations_table.to_csv(
        REPO_PATH / config["output_dir"] / "prior_ablations_table.csv",
    )
    # Remove the ("llm_rsa", "L0") table since it is unaffacted by any of the ablations
    prior_ablations_table = prior_ablations_table.drop(("llm_rsa", "L0"), axis=0)
    # Populate the table
    table_str = populate_prior_ablations_table(
        prior_ablations_table=prior_ablations_table
    )
    # Save the table to a file
    with open(
        REPO_PATH / config["output_dir"] / "prior_ablations_table.tex",
        "w",
    ) as f:
        f.write(table_str)


@hydra.main(
    version_base=None,
)
def main(config: DictConfig) -> None:
    ### Config setup ###
    config = OmegaConf.to_container(config, resolve=True)
    if not os.path.exists(REPO_PATH / config["output_dir"]):
        os.makedirs(REPO_PATH / config["output_dir"], exist_ok=True)
    ## PragMega+ Experiment Results ###
    # Generate the tables
    generate_tables(config=config)
    # Generate the plots
    generate_plots(config=config)
    # ### PragMega+ Experiment Analysis ###
    prior_ablations(config=config)


if __name__ == "__main__":
    main()
