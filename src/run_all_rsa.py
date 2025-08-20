import os
from copy import deepcopy
from itertools import product

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from rsa import rsa
from src import REPO_PATH
from src.utils.helpers import read_yaml, write_yaml


def set_prior_ablation_config(
    prior_ablation_config: DictConfig, prior_ablation: str
) -> DictConfig:
    all_models = ["llm_rsa", "rsa_two", "rsc_rsa"]
    if prior_ablation == "no_p_m_c":
        prior_key = "prior_meaning"
        models = all_models
    elif prior_ablation == "no_p_r_c_u":
        prior_key = "prior_rs"
        models = ["rsa_two", "rsc_rsa"]
    elif prior_ablation == "no_p_u_c":
        prior_key = "prior_utterance"
        models = all_models
    else:
        raise ValueError(f"Unknown prior ablation: {prior_ablation}")
    excluded_models = set(all_models) - set(models)
    for i, rsa_run_config in enumerate(prior_ablation_config["rsa_runs"]):
        if rsa_run_config["run_type"] in excluded_models:
            prior_ablation_config["rsa_runs"].pop(i)
    for rsa_run_config in prior_ablation_config["rsa_runs"]:
        run_type = rsa_run_config["run_type"]
        rsa_run_config[run_type][prior_key] = "uniform"
    return prior_ablation_config


def set_ration_param_config(
    ration_param_config: DictConfig, ration_param: str, value: float
) -> DictConfig:
    all_models = ["llm_rsa", "rsa_two", "rsc_rsa"]
    if ration_param == "alpha":
        ration_key = "alpha"
        models = all_models
    elif ration_param == "lambda":
        ration_key = "alpha_pllm_rs"
        models = ["rsa_two", "rsc_rsa"]
    else:
        raise ValueError(f"Unknown rationality parameter: {ration_param}")
    for model in models:
        ration_param_config[model][ration_key] = value
    for excluded_model in set(all_models) - set(models):
        del ration_param_config[excluded_model]
    return ration_param_config


def set_num_clusters_config(
    num_clusters_ablation_config: DictConfig, num_clusters: int
) -> DictConfig:
    all_models = ["llm_rsa", "rsa_two", "rsc_rsa"]
    models = ["rsc_rsa"]
    for model in models:
        num_clusters_ablation_config[model]["number_of_clusters"] = num_clusters
    for excluded_model in set(all_models) - set(models):
        del num_clusters_ablation_config[excluded_model]
    return num_clusters_ablation_config


def clear_probability_files(dir_path: str) -> None:
    # First check if the directory exists
    if not os.path.exists(dir_path):
        return
    # Delete all rsa-related files in the directory
    rsa_file_names = [
        f"{agent}{n}_{model}.jsonl"
        for agent, n, model in product(
            ["L", "S"],
            list(range(0, 6)),
            [
                "llm_rsa",
                "rsa_two",
                "rsa_two_p_r_c_u",
                "rsa_two_indicator_r_c_u",
                "rsc_rsa",
            ],
        )
    ]
    for file_name in os.listdir(dir_path):
        if file_name not in rsa_file_names:
            continue
        print(f"Clearing file: {file_name}")
        with open(os.path.join(dir_path, file_name), "w") as f:
            f.write("")


# Runs all the instances of RSA for the results and analysis sections
@hydra.main(
    version_base=None,
)
def main(config: DictConfig):
    config = OmegaConf.to_container(config, resolve=True)
    # Store all the runs in a results_and_analysis config so that all the runs
    # can immediately be run on the results_and_analysis.py
    main_run_name = f"gen_llm={config['gen_llm']}_prob_llm={config['prob_llm']}" + (
        config["extra_details"] if "extra_details" in config else ""
    )
    results_and_analysis_config = {
        "output_dir": os.path.join(
            config["results_and_analysis_output_dir"], main_run_name
        )
    }
    # Run the main RSA run
    # This is the one that will be in the main paper
    base_config = read_yaml(REPO_PATH / config["base_config"])
    for common_key in config["base_config_common_keys"]:
        base_config[common_key] = config["base_config_common_keys"][common_key]
    main_rsa_config = deepcopy(base_config)
    main_rsa_config["output_dir"] = os.path.join(
        main_rsa_config["output_dir"],
        main_run_name,
        "main_run",
    )
    print(f"Running main RSA with config: {main_rsa_config}")
    clear_probability_files(dir_path=REPO_PATH / main_rsa_config["output_dir"])
    rsa(config=main_rsa_config)
    results_and_analysis_config["rsa_runs"] = {
        main_run_name: main_rsa_config["output_dir"]
    }
    # Run the prior ablation runs
    if "prior_ablations" in config:
        results_and_analysis_config["prior_ablations"] = {
            "unchanged": main_rsa_config["output_dir"]
        }
        for prior_ablation in set(config["prior_ablations"]) - {"unchanged"}:
            prior_ablation_config = deepcopy(base_config)
            prior_ablation_config = set_prior_ablation_config(
                prior_ablation_config=prior_ablation_config,
                prior_ablation=prior_ablation,
            )
            prior_ablation_config["output_dir"] = os.path.join(
                prior_ablation_config["output_dir"],
                main_run_name,
                prior_ablation,
            )
            print(
                f"Running prior ablation {prior_ablation} with config: {prior_ablation_config}"
            )
            clear_probability_files(
                dir_path=REPO_PATH / prior_ablation_config["output_dir"]
            )
            rsa(config=prior_ablation_config)
            results_and_analysis_config["prior_ablations"][prior_ablation] = (
                prior_ablation_config["output_dir"]
            )
    # Run the rationality parameter ablations
    if "rationality_parameter_ablations" in config:
        rationality_params = list(config["rationality_parameter_ablations"].keys())
        results_and_analysis_config["rationality_parameter_ablations"] = {
            ration_param: {} for ration_param in rationality_params
        }
        for ration_param in rationality_params:
            ration_param_values = sorted(
                [
                    (value, eval(value))
                    for value in config["rationality_parameter_ablations"][ration_param]
                ],
                key=lambda x: x[1],
            )
            for str_value, float_value in ration_param_values:
                ration_param_config = deepcopy(base_config)
                ration_param_config["output_dir"] = os.path.join(
                    ration_param_config["output_dir"],
                    main_run_name,
                    f"{ration_param}={str_value}",
                )
                ration_param_config = set_ration_param_config(
                    ration_param_config=ration_param_config,
                    ration_param=ration_param,
                    value=float_value,
                )
                print(
                    f"Running rationality parameter ablation {ration_param}={str_value} with config: {ration_param_config}"
                )
                clear_probability_files(
                    dir_path=REPO_PATH / ration_param_config["output_dir"]
                )
                rsa(config=ration_param_config)
                results_and_analysis_config["rationality_parameter_ablations"][
                    ration_param
                ][str_value] = ration_param_config["output_dir"]
    # Run the number of clusters ablation
    if "num_clusters_ablations" in config:
        results_and_analysis_config["num_clusters_ablations"] = {}
        num_clusters_ablations = config["num_clusters_ablations"]
        for num_clusters in num_clusters_ablations:
            num_clusters_ablation_config = deepcopy(base_config)
            num_clusters_ablation_config["output_dir"] = os.path.join(
                num_clusters_ablation_config["output_dir"],
                main_run_name,
                f"num_clusters={num_clusters}",
            )
            num_clusters_ablation_config = set_num_clusters_config(
                num_clusters_ablation_config=num_clusters_ablation_config,
                num_clusters=num_clusters,
            )
            print(
                f"Running number of clusters ablation with config: {num_clusters_ablation_config}"
            )
            clear_probability_files(
                dir_path=REPO_PATH / num_clusters_ablation_config["output_dir"]
            )
            rsa(config=num_clusters_ablation_config)
            results_and_analysis_config["num_clusters_ablations"][num_clusters] = (
                num_clusters_ablation_config["output_dir"]
            )
    # Write all the run paths to a yaml file
    write_yaml(
        yaml_contents=results_and_analysis_config,
        yaml_file_path=REPO_PATH
        / config["this_output_dir"]
        / f"{main_run_name}_results_and_analysis.yaml",
    )


if __name__ == "__main__":
    main()
