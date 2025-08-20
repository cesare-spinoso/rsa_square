from ast import literal_eval
import fcntl
import json
import os
from datetime import datetime
import warnings

import jsonlines
from omegaconf import DictConfig, OmegaConf

from src import REPO_PATH, REPO_LOG_FILE
from src.llms.base_llm import BaseLLM
from src.utils.helpers import (
    read_jsonlines,
    write_yaml,
)


def log_run(func):
    def wrapper(config: dict | DictConfig):
        # Extract the config
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        # Get the start time
        start_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        config["run_name"] = start_time
        config["start_time"] = start_time
        # Log the config to stdout
        print(f"Running the config {config['run_name']}. Here are the details:")
        print(json.dumps(config, indent=4))
        print(f"These details will also be logged to {REPO_LOG_FILE}")
        # Log the config to the log file
        with open(REPO_LOG_FILE, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write("=" * 60 + "\n")
            f.write(json.dumps(config, indent=4))
            f.write("\n" + "=" * 60)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        # Create dedicated output directory
        if "output_dir" in config:
            new_output_dir = os.path.join(config["output_dir"], start_time)
            config["output_dir"] = new_output_dir
        # Run the damn thing!
        func(config)
        # Log the end time
        end_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        config["end_time"] = end_time
        # Log the config to stdout
        print(
            f"Finished running the config {config['run_name']}. Here are the details:"
        )
        print(json.dumps(config, indent=4))
        print(f"These details will also be logged to {REPO_LOG_FILE}")
        # Log the config to the log file
        with open(REPO_LOG_FILE, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write("=" * 60)
            f.write(json.dumps(config, indent=4))
            f.write("=" * 60)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
        # Log the config to the dedicated directory
        if "output_dir" in config:
            if not os.path.exists(os.path.join(REPO_PATH, config["output_dir"])):
                warnings.warn(
                    f"The output directory {config['output_dir']} does not exist. Creating it."
                    + "However, this is NOT the expected behaviour as the directory should have"
                    + "been created by the function that was run."
                )
                os.makedirs(os.path.join(REPO_PATH, config["output_dir"]))
            write_yaml(
                yaml_contents=config,
                yaml_file_path=os.path.join(
                    REPO_PATH, config["output_dir"], "config.yaml"
                ),
            )

    return wrapper


def get_cache_file(llm: BaseLLM, config: dict) -> str:
    model_id = llm.model_id.replace("/", "_")
    cache_path = REPO_PATH / config["cache_dir"] / f"{model_id}.jsonl"
    return cache_path


def cache_response(func):
    def wrapper(*args, **kwargs):
        # Conditions needed for using decorator
        assert (
            "config" in kwargs
            and "cache_dir" in kwargs["config"]
            and "llm" in kwargs
            and isinstance(kwargs["llm"], BaseLLM)
            and "prompt" in kwargs
        )
        # Convert to list if str for compatibility with batching
        prompt = kwargs["prompt"]
        if isinstance(prompt, str):
            prompt = [prompt]
        # Open cache and check for cache hits, only return if all are hits
        cache_path = get_cache_file(llm=kwargs["llm"], config=kwargs["config"])
        cache = read_jsonlines(file_path=cache_path)
        cache_prompts_to_idx = {c["prompt"]: i for i, c in enumerate(cache)}
        cache_hits = [
            cache[cache_prompts_to_idx[p]] for p in prompt if p in cache_prompts_to_idx
        ]
        if len(cache_hits) == len(prompt):
            print("Cache hit!")
            return [
                {
                    "option_probs": r["option_probs"],
                    "next_token_probs": r["next_token_probs"],
                }
                for r in cache_hits
            ]
        print("Cache miss!")
        if isinstance(prompt, list) and len(set(prompt)) < len(prompt):
            # Check if there are inefficient calls to func e.g. duplicated prompts
            # If yes, only call once and then assign results in post
            warnings.warn(
                "Duplicate prompts were identified in this batch. They are being removed for efficiency."
                + "Please ensure that this is the expected behaviour."
            )
            prompt_info = [(str(p), str(o)) for p, o in zip(kwargs["prompt"], kwargs["option_encoding"])]
            prompt_info = list(set(prompt_info))
            kwargs["prompt"] = [p for p, _ in prompt_info]
            kwargs["option_encoding"] = [literal_eval(o) for _, o in prompt_info]
        response = func(*args, **kwargs)
        # Post-process in case there were duplicates which need to be copied to the output
        assert len(response) == len(kwargs["prompt"])
        response_map = {p: r for p, r in zip(kwargs["prompt"], response)}
        dict_to_write = [
            {
                "prompt": p,
                "option_probs": response_map[p]["option_probs"],
                "next_token_probs": response_map[p]["next_token_probs"],
            }
            for p in prompt # prompt contains the original kwargs["prompt"]
        ]
        cache = cache + dict_to_write
        with jsonlines.open(cache_path, "w") as writer:
            writer.write_all(cache)
        # Response should have same length as input
        response = [
            {
                "option_probs": d["option_probs"],
                "next_token_probs": d["next_token_probs"],
            }
            for d in dict_to_write
        ]
        assert len(response) == len(prompt)
        return response

    return wrapper
