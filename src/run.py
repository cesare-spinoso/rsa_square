import ast
import itertools
import math
import random
import re
import time
import warnings

import hydra
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import REPO_PATH
from src.llms.base_llm import BaseLLM
from src.utils.decorators import cache_response, get_cache_file, log_run
from src.utils.helpers import append_jsonlines, read_jsonlines

# Alternative utterance generation
# NOTE: Only the completion-based generation is left, see previous history for the prompt-based
# generation, though we experimentally showed that this did not work very well which is why we decided
# to use the completion-based generation.


def load_llm(config: dict, gen: bool = False) -> BaseLLM:
    if gen:
        if "instruct" in config["gen_llm"].lower():
            warnings.warn(
                f"You are using an instruction-tuned model ({config['gen_llm']}) for alternative utterance generation."
                + "Make sure that this is what you want."
            )
        llm = BaseLLM(model_id=config["gen_llm"], seed=config["seed"])
    else:
        if "instruct" not in config["prob_llm"].lower():
            warnings.warn(
                f"You are using a base pre-trained model ({config['prob_llm']}) for probability estimate computation."
                + "Make sure that this is what you want."
            )
        llm = BaseLLM(model_id=config["prob_llm"], seed=config["seed"])
    return llm


def convert_scenario_to_prompt(scenario: str) -> str:
    return "".join(scenario.split('"')[:-2]) + '"'


def convert_completions_to_utterances(
    scenario_prompt: str, completions: list[str]
) -> list[str]:
    utterances = []
    for completion in completions:
        utterance = completion.partition(scenario_prompt)[2].strip()
        utterance = '"' + utterance.split('"')[0] + '"'
        print(f"Alternative utterance: {utterance}")
        utterances.append(utterance)
    return utterances


def generate_completion_based_utterances(config: dict) -> None:
    # Load the dataset
    dataset = pd.read_csv(REPO_PATH / config["dataset_path"])
    # Filter to only keep unique scenarios
    dataset = dataset[["label", "id", "complete_scenario"]].drop_duplicates()
    # Collect existing alternative utterances so don't re-generate
    alternative_utterances = read_jsonlines(
        file_path=REPO_PATH / config["output_dir"] / f'{config["output_file"]}'
    )
    alternative_key = [(elt["label"], elt["id"]) for elt in alternative_utterances]
    # Load LM for generation
    alt_utt_llm = load_llm(config=config, gen=True)
    for _, elt in tqdm(dataset.iterrows()):
        if (elt["label"], elt["id"]) in alternative_key:
            continue
        prompt = convert_scenario_to_prompt(elt["complete_scenario"])
        print(f"Generating alternative utterances for {prompt}")
        generated_completions = alt_utt_llm.get_completion(
            prompt=prompt,
            generation_kwargs=config["generation_kwargs"],
        )
        alternative_utterances = convert_completions_to_utterances(
            scenario_prompt=prompt, completions=generated_completions
        )
        for i in range(len(alternative_utterances)):
            alternative = {
                "label": elt["label"],
                "id": elt["id"],
                "alt_id": i + 1,
                "alt_utt": alternative_utterances[i],
            }
            append_jsonlines(
                dict_to_write=[alternative],
                output_directory=REPO_PATH / config["output_dir"],
                file_name=config["output_file"],
            )


def generate_alternative_utterances(config: dict) -> None:
    if "use_scenario_as_prompt" in config:
        # This uses completions of the scenario (e.g. Alice says " -> LLM completion)
        generate_completion_based_utterances(config=config)
    else:
        raise NotImplementedError(
            "No other alternative utterance generation implementation supported in this version."
            + "Talk to the authors for others things they might have tried (e.g., using prompt templates for alternative utterance generation, which did not work very well)."
        )


# LLM-based probability computations *which relies on prompt templates*
# NOTE: After many iterations, this has been refactored to ONLY generate LLM-based probabilities
# This is because computing RSA-based probabilities is much faster when vectorized, using the LLM
# probabilities as starting points.


def get_meanings_info(grouped_df: pd.DataFrame) -> list[dict]:
    meanings_info = grouped_df[["option", "option_type", "is_correct", "options"]]
    meanings_info = meanings_info.rename(
        {"option": "meaning_id", "option_type": "meaning_type", "options": "meaning"},
        axis=1,
    )
    meanings_info = meanings_info.drop_duplicates()
    meanings_info = meanings_info.to_dict(orient="records")
    return meanings_info


def get_utterance_intro(grouped_df: pd.DataFrame) -> str:
    utterance_intros = grouped_df["utterance_intro"].drop_duplicates().values
    assert len(utterance_intros) == 1
    utterance_intro = list(utterance_intros)[0]
    utterance_intro = utterance_intro.strip()
    return utterance_intro


def get_complete_scenario(grouped_df: pd.DataFrame) -> str:
    # Get the scenario with the context and utterance and the preset question
    scenarios_w_question = grouped_df["complete_scenario"].drop_duplicates().values
    assert len(scenarios_w_question) == 1
    scenario_w_question = list(scenarios_w_question)[0]
    return scenario_w_question


def get_scenario_wo_question(scenario: str) -> str:
    # Remove the preset question from the scenario
    scenario_wo_question = re.sub(r"What (did|does) .* want to convey\?", "", scenario)
    return scenario_wo_question


def get_context(grouped_df: str, is_context_w_something: bool = False) -> str:
    if not is_context_w_something:
        # NOTE: This is kept for backwards compatibility
        # The context is like the state of the world. It should not include the utterance.
        # Remove the question from the scenario
        scenario = get_complete_scenario(grouped_df=grouped_df)
        utterance_intro = get_utterance_intro(grouped_df=grouped_df)
        context = re.sub(r"What (did|does) .* want to convey\?", "", scenario)
        context = re.sub(r"What is .* likely to say\?", "", context)
        # Remove the utterance and it's intro
        context = context.partition(utterance_intro)[0]
        context = context.strip()
    else:
        context = grouped_df["context_w_something"].drop_duplicates().values
        assert len(context) == 1
        context = list(context)[0]
    return context


def get_context_w_something(
    grouped_df: pd.DataFrame, is_context_w_something: bool = False
) -> str:
    if not is_context_w_something:
        return None
    else:
        context = grouped_df["context_w_something"].drop_duplicates().values
        assert len(context) == 1
        context = list(context)[0]
    return context


def get_utterance_from_scenario(scenario: str) -> str:
    utterance = re.findall(r'"[^"]*"', scenario)[-1]
    utterance = remove_quotation_marks(utterance)
    return utterance


def get_option_encoding(options: list[str]) -> dict[str, str]:
    # NOTE: The meaning encoding is hard-coded to be index-1
    encoding = {
        tuple(t) if isinstance(t, list) else t: str(i + 1)
        for i, t in enumerate(options)
    }
    return encoding


def generate_option_orders(
    options: list[str] | list[tuple[str]], num_order_shuffles: int, seed: int = 42
) -> list[list[str]] | list[list[list[str]]]:
    if all(isinstance(opt, str) for opt in options):
        all_option_orders = list(itertools.permutations(options))
        random.seed(seed)
        return random.sample(all_option_orders, num_order_shuffles)
    else:
        raise ValueError(f"Invalid options: {options=}")


def remove_quotation_marks(s: str) -> str:
    while s.startswith('"'):
        s = s[1:]
    while s.endswith('"'):
        s = s[:-1]
    return s


def load_alternative_utterances(config: dict) -> pd.DataFrame:
    alternatives = read_jsonlines(REPO_PATH / config["alternative_utterances_path"])
    alternatives = pd.DataFrame(alternatives)
    alternatives["alt_utt"] = alternatives["alt_utt"].apply(
        lambda x: remove_quotation_marks(x)
    )
    return alternatives


def merge_datasets(
    jenn_hu_dataset: pd.DataFrame,
    alternatives: pd.DataFrame,
    config: dict,
) -> list[dict]:
    # Merge the datasets and create list of dicts which can be used for all the prompt creations
    df_merged = jenn_hu_dataset.merge(alternatives, on=["label", "id"])
    merged_for_prompting = (
        []
    )  # one element per scenario, containing all information needed for prompt creation
    for keys, grouped_df in tqdm(
        df_merged.groupby(["label", "id"]),
        desc="Merging Jenn Hu dataset and alternative utterances.",
    ):
        # Drop label since always "IV"
        dict_to_add = {"id": int(keys[1])}
        # Meaning info as a list of dict [{meaning_id: ..., meaning_type: ..., meaning: ...}]
        meanings_information = get_meanings_info(grouped_df=grouped_df)
        dict_to_add["meanings"] = meanings_information
        # Complete scenario includes "What was X trying to convey?"
        complete_scenario = get_complete_scenario(grouped_df=grouped_df)
        scenario_wo_question = get_scenario_wo_question(scenario=complete_scenario)
        dict_to_add["scenario_wo_question"] = scenario_wo_question.strip()
        # Fetch the context
        # NOTE: New in v2.csv, "...X says something." is the context string
        context = get_context(
            grouped_df=grouped_df,
            is_context_w_something=config["is_context_w_something"],
        )
        dict_to_add["context"] = context.strip()
        # Identify the speaker name
        dict_to_add["speaker"] = re.search(
            r"What (did|does) (.*) want to convey\?", complete_scenario
        ).group(2)
        # Isolate the speaker's original utterance
        dict_to_add["original_utterance"] = remove_quotation_marks(
            get_utterance_from_scenario(scenario=complete_scenario)
        )
        # Get all the alternative utterances
        # Use option == 1 since they're duplicated 4 times
        dict_to_add["alternative_utterances"] = [
            remove_quotation_marks(utt)
            for utt in grouped_df[grouped_df["option"] == 1]["alt_utt"].tolist()
        ]
        merged_for_prompting.append(dict_to_add)
    return merged_for_prompting


def get_options(
    merged: list[dict], prob_to_compute: str, prob_to_compute_config: dict
) -> pd.DataFrame:
    # PLLM(m|c,r) has been dropped as 1. It doesn't work well and 2. We can assume conditional
    # independence in the model
    if prob_to_compute in ["PLLM(m|c)", "PLLM(m|c,u)", "PLLM(m|c,u,r)", "PLLM(m|c,r)"]:
        options = []
        for elt in merged:
            options.append(
                {"id": elt["id"], "options": [m["meaning"] for m in elt["meanings"]]}
            )
    elif prob_to_compute in ["PLLM(r|c,u)"]:
        options = [
            {
                "id": elt["id"],
                "options": list(prob_to_compute_config["prompt_options"].values()),
            }
            for elt in merged
        ]
    else:
        raise ValueError(f"Invalid probs_to_compute: {prob_to_compute}")
    return pd.DataFrame(options)


def get_df_to_compute_probs(
    merged: list[dict], prob_to_compute: str, prob_to_compute_config: dict
) -> pd.DataFrame:
    # Convert to a df and create columns which are common to all probabilities
    df_merged = pd.DataFrame(merged)
    # Apply question replacement
    df_merged["question_w_speaker"] = df_merged.apply(
        lambda row: prob_to_compute_config["question"].replace("X", row["speaker"]),
        axis=1,
    )
    # Filter columns based on the conditional in probability
    if prob_to_compute in ["PLLM(m|c)", "PLLM(m|c,r)"]:
        df_merged = df_merged[["id", "context", "question_w_speaker"]]
    elif prob_to_compute in ["PLLM(m|c,u)", "PLLM(m|c,u,r)", "PLLM(r|c,u)"]:
        df_merged["utterance"] = df_merged.apply(
            lambda row: [row["original_utterance"]] + row["alternative_utterances"],
            axis=1,
        )
        df_merged = df_merged.explode("utterance")
        df_merged["scenario_w_replaced_utterance_wo_question"] = df_merged.apply(
            lambda row: row["scenario_wo_question"].replace(
                row["original_utterance"], row["utterance"]
            ),
            axis=1,
        )
        df_merged = df_merged[
            [
                "id",
                "context",
                "utterance",
                "scenario_w_replaced_utterance_wo_question",
                "question_w_speaker",
            ]
        ]
    # Add the rhetorical strategy to the df if it's in the conditional
    rhetorical_strategy_instructions = [
        {"rhetorical_strategy": rs, "rhetorical_strategy_instruction": rsi}
        for rs, rsi in prob_to_compute_config.get(
            "rhetorical_strategy_instructions", dict()
        ).items()
    ]
    if len(rhetorical_strategy_instructions) > 0:
        df_merged = df_merged.merge(
            pd.DataFrame(rhetorical_strategy_instructions), how="cross"
        )
    # Add the options (i.e. the target_rv) to the df
    options = get_options(
        merged=merged,
        prob_to_compute=prob_to_compute,
        prob_to_compute_config=prob_to_compute_config,
    )
    df_to_compute_probs = pd.merge(df_merged, options, on="id")
    return df_to_compute_probs


def apply_prompt_template(
    row: pd.Series, prob_to_compute: str, prob_to_compute_config: dict
) -> str:
    rhetorical_strategy_instruction = (
        row["rhetorical_strategy_instruction"]
        if "rhetorical_strategy_instruction" in row
        else ""
    )
    option_str = "\n".join(
        [f"{num}. {option}" for option, num in row["option_encoding"].items()]
    )
    if prob_to_compute in ["PLLM(m|c)", "PLLM(m|c,r)"]:
        scenario = row["context"]
    elif prob_to_compute in ["PLLM(m|c,u)", "PLLM(m|c,u,r)", "PLLM(r|c,u)"]:
        scenario = row["scenario_w_replaced_utterance_wo_question"]
    question = row["question_w_speaker"]
    return prob_to_compute_config["prompt_template"].format(
        rhetorical_strategy_instruction=rhetorical_strategy_instruction,
        scenario=scenario,
        question=question,
        options=option_str,
    )


def create_dfs_to_compute_probs(
    merged: pd.DataFrame, config: dict
) -> dict[str, pd.DataFrame]:
    dfs_to_compute_probs = {}
    for prob_to_compute in tqdm(
        config["probs_to_compute"], desc="Creating prompt dfs."
    ):
        assert prob_to_compute in [
            "PLLM(m|c)",
            "PLLM(m|c,u)",
            "PLLM(m|c,u,r)",
            # "PLLM(m|c,r)" -> Use PLLM(m|c) as a simplifying assumption,
            "PLLM(r|c,u)",
        ]
        prob_to_compute_config = config["probs_to_compute"][prob_to_compute]
        # Create a df with all the conditional rv and the options for the target rv
        # in a side-by-side manner. This df is different for each prob_to_compute
        # but makes streamlining much easier.
        df_to_compute_probs = get_df_to_compute_probs(
            merged=merged,
            prob_to_compute=prob_to_compute,
            prob_to_compute_config=prob_to_compute_config,
        )
        # Apply order shuffling generation to the options
        # NOTE: The same set of options should be shuffled in the same way.
        # This is desired because caching is being used for the LLM prompt probabilities.
        df_to_compute_probs["options"] = df_to_compute_probs["options"].apply(
            lambda x: generate_option_orders(
                options=x,
                num_order_shuffles=prob_to_compute_config["num_order_shuffles"],
            )
        )
        # Blow up the options column to create rows with only one option list
        df_to_compute_probs = df_to_compute_probs.explode("options")
        # Encode the options (so can invert the probabilities from the MCQ format back to their strings)
        df_to_compute_probs["option_encoding"] = df_to_compute_probs["options"].apply(
            lambda x: get_option_encoding(options=x)
        )
        # Apply the prompt template
        df_to_compute_probs["prompt"] = df_to_compute_probs.apply(
            lambda row: apply_prompt_template(
                row=row,
                prob_to_compute=prob_to_compute,
                prob_to_compute_config=prob_to_compute_config,
            ),
            axis=1,
        )
        dfs_to_compute_probs[prob_to_compute] = df_to_compute_probs
    return dfs_to_compute_probs


@cache_response
def compute_llm_next_token_prob(
    prompt: list[str],
    option_encoding: list[dict[str, str]],
    llm: BaseLLM,
    config: dict,
) -> list[dict[str, dict[str, float]]]:
    # Compute next token probs
    time_now = time.time()
    next_token_probs = llm.get_next_token_probs(
        prompt=prompt, next_tokens=[list(o.values()) for o in option_encoding]
    )
    probs = []
    for o_enc, nt_probs in zip(option_encoding, next_token_probs):
        inverse_encoding = {v: k for k, v in o_enc.items()}
        option_probs = {
            inverse_encoding[token]: prob for token, prob in nt_probs.items()
        }
        probs.append({"next_token_probs": nt_probs, "option_probs": option_probs})
    print(f"Time taken for llm call: {time.time() - time_now:.2f}s")
    return probs


def compute_llm_probs(
    dfs_to_compute_probs: dict[str, pd.DataFrame], llm: BaseLLM, config: dict
) -> dict[str, pd.DataFrame]:
    dfs_probs = {}
    for prob_to_compute, df_to_compute_prob in tqdm(dfs_to_compute_probs.items()):
        print(f"Computing probabilities for: {prob_to_compute}")
        # Do a fast cache retrieval first
        cache_file = get_cache_file(llm=llm, config=config)
        jsonlines_data = read_jsonlines(file_path=cache_file)
        jsonlines_df = pd.DataFrame(jsonlines_data)
        # Remove duplicate prompts (caching using jsonlines)
        cached_llm_probs = pd.DataFrame([])
        if not jsonlines_df.empty:
            jsonlines_df = jsonlines_df[~jsonlines_df["prompt"].duplicated()]
            # Merge with prompting data
            cached_llm_probs = pd.merge(
                left=df_to_compute_prob, right=jsonlines_df, on="prompt"
            )
            # Remove cached rows
            df_to_compute_prob = df_to_compute_prob[
                ~df_to_compute_prob["prompt"].isin(cached_llm_probs["prompt"])
            ]
        if len(df_to_compute_prob) == 0:
            print(f"No new rows to compute for {prob_to_compute}")
            dfs_probs[prob_to_compute] = cached_llm_probs
            continue
        # Batching the computation of probabilities
        # Convert the option_encoding dict to str so the dataloader keeps lists of dict
        df_to_compute_prob["option_encoding"] = df_to_compute_prob.apply(
            lambda row: str(row["option_encoding"]), axis=1
        )
        dataloader = DataLoader(
            df_to_compute_prob.to_dict("records"),
            batch_size=config["batch_size"],
            shuffle=False,
        )
        probs = []
        for batch in tqdm(dataloader):
            probs.extend(
                compute_llm_next_token_prob(
                    prompt=batch["prompt"],
                    option_encoding=[
                        ast.literal_eval(str_dict)
                        for str_dict in batch["option_encoding"]
                    ],
                    llm=llm,
                    config=config,
                )
            )
        assert len(probs) == len(
            df_to_compute_prob
        ), f"{len(probs)=}, {len(df_to_compute_prob)=}"
        # Add missing columns to probs
        dfs_probs[prob_to_compute] = pd.concat(
            [df_to_compute_prob.reset_index(drop=True), pd.DataFrame(probs)], axis=1
        )
        # Add cached probabilities + corresponding columns as well
        dfs_probs[prob_to_compute] = pd.concat(
            [cached_llm_probs, dfs_probs[prob_to_compute]], axis=0
        )
    return dfs_probs


def get_grouping_cols(prob_to_compute: str) -> list[str]:
    # probabilities, making sure that they are high for the correct meaning/rhetorical strategy
    if prob_to_compute == "PLLM(m|c)":
        grouping_cols = ["context"]
    # elif prob_to_compute == "PLLM(m|c,r)":
    #     grouping_cols = ["context", "rhetorical_strategy"]
    elif prob_to_compute in ["PLLM(m|c,u)", "PLLM(r|c,u)"]:
        grouping_cols = ["context", "utterance"]
    elif prob_to_compute == "PLLM(m|c,u,r)":
        grouping_cols = ["context", "utterance", "rhetorical_strategy"]
    elif prob_to_compute == "PLLM(m|u)":
        grouping_cols = ["utterance"]
    return grouping_cols


def process_probs(
    dfs_probs: dict[str, pd.DataFrame], config: dict
) -> dict[str, pd.DataFrame]:
    processed_dfs = {}
    for prob_to_compute, df_probs in dfs_probs.items():
        # Base on the prob to compute, select columns needed for grouping
        grouping_cols = get_grouping_cols(prob_to_compute=prob_to_compute)
        grouping_cols = ["id"] + grouping_cols
        # Group and average the option probabilities
        df_probs = df_probs.groupby(by=grouping_cols)["option_probs"].apply(
            # Not doing .to_dict() because pandas converts this to columns automatically
            lambda x: [
                item for item in pd.DataFrame(x.values.tolist()).mean(axis=0).items()
            ]
        )
        df_probs = df_probs.reset_index()
        df_probs["option_probs"] = df_probs["option_probs"].apply(
            lambda tuples: {k: v for k, v in tuples}
        )
        # If there is some kind of mapping between the option as a string in the prompt
        # and the string representing the option (e.g., the Sincere rhetorical strategy
        # can be stated as "Literal" or "The character is being literal."), revert back
        # to the string representing the option.
        if "prompt_options" in config["probs_to_compute"][prob_to_compute]:
            reverse_option_mapping = {
                v: k
                for k, v in config["probs_to_compute"][prob_to_compute][
                    "prompt_options"
                ].items()
            }
            df_probs["option_probs"] = df_probs["option_probs"].apply(
                lambda x: {reverse_option_mapping[k]: v for k, v in x.items()}
            )
        processed_dfs[prob_to_compute] = df_probs
    return processed_dfs


def save_probs(
    dfs_probs: dict[str, pd.DataFrame], jenn_hu_dataset: pd.DataFrame, config: dict
) -> None:
    for id_, grouped_df in jenn_hu_dataset.groupby(by="id"):
        # Add common information first
        dict_to_add = {"id": id_}
        meanings_information = get_meanings_info(
            grouped_df=grouped_df
        )  # Function requires a dataframe
        dict_to_add["meanings"] = meanings_information
        complete_scenario = get_complete_scenario(
            grouped_df=grouped_df
        )  # Function requires a dataframe
        scenario_wo_question = get_scenario_wo_question(scenario=complete_scenario)
        dict_to_add["scenario_wo_question"] = scenario_wo_question.strip()
        dict_to_add["original_utterance"] = get_utterance_from_scenario(
            scenario=complete_scenario
        )
        dict_to_add["context_w_something"] = get_context_w_something(
            grouped_df=grouped_df,
            is_context_w_something=config["is_context_w_something"],
        )
        dict_to_add["hyperparameters"] = {
            "llm": config["prob_llm"],
            "dataset_path": config["dataset_path"],
        }
        # Add probabilities
        dict_to_add["PLLM"] = {}
        for prob_to_compute, df_probs in dfs_probs.items():
            filtered_by_id = df_probs[df_probs["id"] == id_]
            grouping_cols = get_grouping_cols(prob_to_compute=prob_to_compute)
            # Preserves format of previous implementation except that single conditional variables
            # becomes tuples of that variable rather than just the string itself
            dict_to_add["PLLM"][prob_to_compute] = dict(
                zip(
                    list(
                        filtered_by_id[grouping_cols].itertuples(index=False, name=None)
                    ),
                    filtered_by_id["option_probs"],
                )
            )
            # Convert the tuple to string for writing purposes
            dict_to_add["PLLM"][prob_to_compute] = {
                str(k): v for k, v in dict_to_add["PLLM"][prob_to_compute].items()
            }
        append_jsonlines(
            dict_to_write=[dict_to_add],
            output_directory=REPO_PATH / config["output_dir"],
            file_name="llm_prompt_probs.jsonl",
        )


def generate_llm_prompt_probs(config: dict) -> None:
    # Load the dataset and the alternative utterances
    dataset = pd.read_csv(REPO_PATH / config["dataset_path"])
    # NOTE: For the completion-based generation, may have duplicate utterances
    alternatives = load_alternative_utterances(config=config)
    # Load the language model (so don't have to keep doing it)
    llm = load_llm(config=config, gen=False)
    # Merge the datasets
    merged = merge_datasets(
        jenn_hu_dataset=dataset, alternatives=alternatives, config=config
    )
    # Create dfs which will be directly passed to LLM for probability estimation
    dfs_to_compute_probs = create_dfs_to_compute_probs(merged=merged, config=config)
    # Compute the probabilities
    dfs_probs = compute_llm_probs(
        dfs_to_compute_probs=dfs_to_compute_probs, llm=llm, config=config
    )
    # Process probabilities i.e. average across shuffles
    dfs_probs = process_probs(dfs_probs=dfs_probs, config=config)
    # Save probabilities in the same format as previous version
    save_probs(dfs_probs=dfs_probs, jenn_hu_dataset=dataset, config=config)


# Compute LLM-based utterance probabilities i.e. PLLM(u|c)
# This is placed in a separate run type because it requires a different setup
# than the prompt-based probabilities.


def add_quotation_marks(string: str) -> str:
    if string.startswith('"') and string.endswith('"'):
        return string
    elif string.startswith('"'):
        return string + '"'
    elif string.endswith('"'):
        return '"' + string
    else:
        return '"' + string + '"'


def get_utterance_probs(
    llm: BaseLLM,
    prompt: str,
    utterances: list[str],
) -> list[float]:
    assert prompt.endswith('"')
    assert all(utt.startswith('"') and utt.endswith('"') for utt in utterances)
    removed_quote_prompt = prompt[:-1]
    utterance_probs = []
    for utt in tqdm(utterances, desc="Computing utterance probabilities."):
        completion_log_prob = llm.get_completion_log_probs(
            prompt=removed_quote_prompt, completion=utt
        )
        utterance_probs.append(math.exp(completion_log_prob))
    return utterance_probs


def generate_utterance_probs(config: dict) -> None:
    # Load the dataset and the alternative utterances
    dataset = pd.read_csv(REPO_PATH / config["dataset_path"])
    dataset = dataset[
        ["id", "complete_scenario", "context_w_something"]
    ].drop_duplicates()
    # NOTE: For the completion-based generation, may have duplicate utterances
    alternatives = load_alternative_utterances(config=config)
    # Load the language model (so don't have to keep doing it)
    llm = load_llm(config=config, gen=True)
    # Merge the datasets
    df_merged = pd.merge(left=dataset, right=alternatives, on=["id"])
    # Collect original and alternative utterances and compute their probabilities
    for id_, grouped_df in df_merged.groupby(by="id"):
        # Add common information first
        # Scenario id
        dict_to_add = {"id": id_}
        # Fetch the context (this is only for saving purposes)
        # NOTE: New in v2.csv, "...X says something." is the context string
        context = get_context(
            grouped_df=grouped_df,
            is_context_w_something=config["is_context_w_something"],
        )
        context = context.strip()
        # Convert the complete scenario to a prompt
        complete_scenario = get_complete_scenario(grouped_df=grouped_df)
        original_utterance = get_utterance_from_scenario(scenario=complete_scenario)
        prompt = convert_scenario_to_prompt(scenario=complete_scenario)
        utterances = [original_utterance] + grouped_df["alt_utt"].tolist()
        utterances_w_quotations = [
            add_quotation_marks(string=utt) for utt in utterances
        ]
        utterances_probs = get_utterance_probs(
            llm=llm,
            prompt=prompt,
            utterances=utterances_w_quotations,
        )
        # Normalize, while accounting for duplicate utterances by giving them more weight
        utterance_probs = pd.DataFrame(
            zip(utterances, utterances_probs), columns=["utterance", "prob"]
        )
        utterance_probs = utterance_probs.groupby("utterance").sum().reset_index()
        utterance_probs["prob"] = (
            utterance_probs["prob"] / utterance_probs["prob"].sum()
        )
        dict_to_add["PLLM"] = {}
        dict_to_add["PLLM"]["PLLM(u|c)"] = {
            str((context,)): dict(
                zip(utterance_probs["utterance"], utterance_probs["prob"])
            )
        }
        append_jsonlines(
            dict_to_write=[dict_to_add],
            output_directory=REPO_PATH / config["output_dir"],
            file_name="llm_utterance_probs.jsonl",
        )
    # [Optional] Merge the utterance probabilities to the prompt probabilities
    if "llm_prompt_probs_path" in config:
        llm_prompt_probs = read_jsonlines(REPO_PATH / config["llm_prompt_probs_path"])
        llm_utterance_probs = read_jsonlines(
            REPO_PATH / config["output_dir"] / "llm_utterance_probs.jsonl"
        )
        for llm_prompt, llm_utterance in zip(llm_prompt_probs, llm_utterance_probs):
            assert llm_prompt["id"] == llm_utterance["id"]
            llm_prompt_utterances = [
                ast.literal_eval(k)[1] for k in llm_prompt["PLLM"]["PLLM(m|c,u)"].keys()
            ]
            llm_utterances = list(llm_utterance["PLLM"]["PLLM(u|c)"].values())[0].keys()
            assert set(llm_prompt_utterances) == set(llm_utterances), (
                "Are you sure you're using the correct mathing jsonlines files?\n\n"
                + f"{llm_prompt_utterances=}\n\n{llm_utterances=}"
            )
            llm_prompt["PLLM"]["PLLM(u|c)"] = llm_utterance["PLLM"]["PLLM(u|c)"]
        append_jsonlines(
            dict_to_write=llm_prompt_probs,
            output_directory=REPO_PATH / config["output_dir"],
            file_name="llm_probs.jsonl",
        )


@log_run
def run_config(config: dict):
    if config["run_type"] == "generate_alternative_utterances":
        generate_alternative_utterances(config)
    elif config["run_type"] == "generate_llm_prompt_probs":
        generate_llm_prompt_probs(config)
    elif config["run_type"] == "generate_utterance_probs":
        generate_utterance_probs(config)
    else:
        raise ValueError(f"Invalid run_type: {config['run_type']}")


@hydra.main(
    version_base=None,
)
def main(config: dict) -> None:
    run_config(config=config)


if __name__ == "__main__":
    main()
