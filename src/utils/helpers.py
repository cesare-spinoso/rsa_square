import json
import os
import pickle as pkl
import random

import jsonlines
import numpy as np
import torch
import yaml

# PKL


def read_pkl(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "rb") as reader:
        return pkl.load(reader)


def write_pkl(dict_to_write, output_path):
    with open(output_path, "wb") as writer:
        pkl.dump(dict_to_write, writer)


# JSON


def read_json(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as reader:
        return json.load(reader)


def write_json(dict_to_write, output_path):
    with open(output_path, "w") as writer:
        json.dump(dict_to_write, writer, indent=4)


# JSONLINES


def read_jsonlines(file_path):
    jsonlines_dict = []
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    if not os.path.exists(file_path):
        return []
    with jsonlines.open(file_path, "r") as reader:
        for elt in reader:
            jsonlines_dict.append(elt)
    return jsonlines_dict

def write_jsonlines(dict_to_write, file_path):
    with jsonlines.open(file_path, "w") as writer:
        writer.write_all(dict_to_write)

def append_jsonlines(dict_to_write, output_directory, file_name):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    if not file_name.endswith(".jsonl"):
        file_name += ".jsonl"
    with jsonlines.open(os.path.join(output_directory, file_name), "a") as writer:
        writer.write_all(dict_to_write)


def write_jsonlines(dicts_to_write, output_directory, file_name):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not file_name.endswith(".jsonl"):
        file_name += ".jsonl"
    with jsonlines.open(os.path.join(output_directory, file_name), "w") as writer:
        writer.write_all(dicts_to_write)


# CACHING (WITH JSON)
# Assumes the following structure:
# {
#     key1: [
#         {key2: elt1},
#         {key2: elt2},
#         ...
#     ],
#     ...
# }


def convert_keys_to_strings(d: dict) -> dict:
    return {str(k): v for k, v in d.items()}


def convert_to_cache_key(key: any) -> str:
    if key is None:
        return None
    if isinstance(key, str):
        return key
    else:
        return str(key)


def check_cache(cache_file_path: str, key1: any, key2: any = None) -> dict | None:
    key1 = convert_to_cache_key(key1)
    key2 = convert_to_cache_key(key2)
    cache_content = read_json(cache_file_path)
    if key1 not in cache_content:
        return None
    if key2 is None:
        return cache_content[key1]
    for elt in cache_content[key1]:
        if key2 in elt:
            return elt[key2]
    else:
        return None


def cache_elt(cache_file_path: str, elt: any, key1: any, key2: any = None) -> None:
    key1 = convert_to_cache_key(key1)
    key2 = convert_to_cache_key(key2)
    assert key1 is not None
    cache_content = read_json(cache_file_path)
    if key2 is None:
        cache_content[key1] = elt
        write_json(cache_content, cache_file_path)
        return
    if key1 in cache_content:
        cache_content[key1].append({key2: elt})
    else:
        cache_content[key1] = [{key2: elt}]
    write_json(cache_content, cache_file_path)


def cache_count(cache_file_path: str, key1: any) -> int:
    key1 = convert_to_cache_key(key1)
    cache_value = check_cache(cache_file_path, key1)
    if cache_value is None:
        return 0
    else:
        if isinstance(cache_value, (int, float, str)):
            return 1
        elif isinstance(cache_value, (dict, list, tuple)):
            return len(cache_value)
        else:
            raise ValueError(f"Unexepected cache value type {type(cache_value)=}")


# YAML


def read_yaml(yaml_file_path: str) -> dict:
    with open(yaml_file_path, "r") as f:
        yaml_contents = yaml.safe_load(f)
    if yaml_contents is None:
        return {}
    return yaml_contents


def write_yaml(yaml_contents: dict, yaml_file_path: str) -> None:
    with open(yaml_file_path, "w") as f:
        yaml.dump(yaml_contents, f)


# Experiment utils

def seed_everything(seed: int = 42) -> None:
    # Seed the training algorithm
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch RNG
    torch.cuda.manual_seed(seed)  # CUDA RNG
    torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility
