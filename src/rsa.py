import ast
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Literal, Union

import hydra
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

from src import REPO_PATH
from src.utils.decorators import log_run
from src.utils.helpers import (
    append_jsonlines,
    read_jsonlines,
    read_yaml,
    seed_everything,
)

# Utility functions for priors

# PROB_TYPES = ["L0", "S1", "L1"]
RSA_TYPES = ["llm_rsa", "rsa_two", "rsc_rsa"]


def get_context(pllm: pd.DataFrame) -> str:
    # This assumes pllm is either PLLM(m|c,u) or PLLM(m|c,u,r)
    contexts = [
        ast.literal_eval(elt)[0] if isinstance(elt, str) else elt[0]
        for elt in pllm.index
    ]
    assert len(set(contexts)) == 1
    return contexts[0]


def get_utterances(pllm: pd.DataFrame) -> list[str]:
    # This assumes, as get_context, that pllm is either PLLM(m|c,u) or PLLM(m|c,u,r)
    utterances = [
        ast.literal_eval(elt)[1] if isinstance(elt, str) else elt[1]
        for elt in pllm.index
    ]
    assert len(utterances) == len(set(utterances))
    return utterances


def get_rhetorical_strategies(pllm: pd.DataFrame) -> list[str]:
    # This assumes that pllm is PLLM(m|c,u,r)
    rhetorical_strategies = [
        ast.literal_eval(elt)[2] if isinstance(elt, str) else elt[2]
        for elt in pllm.index
    ]
    assert len(set(rhetorical_strategies)) == 2
    return list(set(rhetorical_strategies))


def get_meaning_model_str(config: dict) -> str:
    l0_probs_path = "/".join(config["l0_probs_path"].split("/")[:-1])
    l0_config_path = os.path.join(REPO_PATH, l0_probs_path, "config.yaml")
    l0_config = read_yaml(l0_config_path)
    llm_string = l0_config["llm"]
    return llm_string


def get_generation_model_str(config: dict) -> str:
    l0_probs_path = "/".join(config["l0_probs_path"].split("/")[:-1])
    l0_config_path = os.path.join(REPO_PATH, l0_probs_path, "config.yaml")
    l0_config = read_yaml(l0_config_path)
    alternative_utterances_path = l0_config["alternative_utterances_path"]
    alternative_utterances_config_path = os.path.join(
        REPO_PATH, "/".join(alternative_utterances_path.split("/")[:-1]), "config.yaml"
    )
    alternative_utterances_config = read_yaml(alternative_utterances_config_path)
    llm_string = alternative_utterances_config["llm"]
    return llm_string


def get_pllm(
    elt: dict,
    rsa_type: str,
) -> pd.DataFrame:
    if rsa_type in ["llm_rsa", "rsc_rsa"]:
        pllm = pd.DataFrame(elt["PLLM"]["PLLM(m|c,u)"])
    elif rsa_type == "rsa_two":
        pllm = pd.DataFrame(elt["PLLM"]["PLLM(m|c,u,r)"])
    else:
        raise ValueError(f"Unrecognized rsa_type: {rsa_type}")
    return pllm


def get_utterance_prior(
    num_utterances: int,
    utterances: list[str],
    prior_type: str,
    elt: dict,
    config: dict,
    prior_utterance_bias: float = None,
) -> np.array:
    if prior_type == "uniform":
        return np.ones(num_utterances)[:, None]
    elif prior_type == "P_u_c":
        pllm = pd.DataFrame(elt["PLLM"]["PLLM(u|c)"]).T
        pllm = pllm[utterances].T
        pllm = pllm.to_numpy()
        return pllm
    elif prior_type == "P_u_c_biased_towards_og":
        pllm = pd.DataFrame(elt["PLLM"]["PLLM(u|c)"]).T
        pllm = pllm[utterances]
        original_utterance = elt["original_utterance"]
        pllm[original_utterance] = pllm[original_utterance] * prior_utterance_bias
        pllm = pllm.T
        pllm = pllm.to_numpy()
        pllm = pllm / pllm.sum(axis=0, keepdims=True)
        return pllm
    else:
        raise ValueError(f"Invalid prior_utt: {config['prior_utt']}")


def get_meaning_prior(
    elt: dict,
    meanings: pd.Index,
    prior_type: Literal["uniform", "P_m_c", "P_m_c_r"],
    rhetorical_strategy: str = None,
) -> np.array:
    if prior_type == "uniform":
        # Only for uniform
        prior = np.ones(len(meanings))[None, :]
    elif prior_type == "P_m_c":
        prior = pd.DataFrame(list(elt["PLLM"]["PLLM(m|c)"].values())[0], index=[0])[
            meanings
        ]
        prior = prior.to_numpy()
    elif prior_type == "P_m_c_r":
        priors = pd.DataFrame(elt["PLLM"]["PLLM(m|c,r)"])
        priors.columns = pd.MultiIndex.from_tuples([eval(x) for x in priors.columns])
        priors = priors.T
        prior = priors.xs(rhetorical_strategy, level=1, axis=0)
        prior = prior.to_numpy()
    else:
        raise ValueError(f"Invalid prior_meaning: {prior_type}")
    return prior


def get_rs_prior(
    rsa_type: Literal["rsa_two", "rsc_rsa"],
    prior_type: Literal["uniform", "P_r_c_u"],
    elt: dict = None,
    rhetorical_strategies: list[str] = None,
    embeddings: np.array = None,
    kmeans: KMeans = None,
) -> Union[dict[Union[int, str], np.array]]:
    if rsa_type == "rsa_two":
        if prior_type == "uniform":
            prior_r = np.ones(len(rhetorical_strategies)) / len(rhetorical_strategies)
            prior_r = {
                rs: prior_r[i].reshape(-1, 1)
                for i, rs in enumerate(rhetorical_strategies)
            }
        elif prior_type == "P_r_c_u":
            pllm = pd.DataFrame(elt["PLLM"]["PLLM(r|c,u)"]).T
            prior_r = {
                rs: pllm[rs].to_numpy().reshape(-1, 1) for rs in rhetorical_strategies
            }
        elif prior_type == "indicator_P_r_c_u":
            pllm = pd.DataFrame(elt["PLLM"]["PLLM(r|c,u)"]).T
            pllm["Ironic"] = pllm.apply(
                lambda row: 1 if row["Ironic"] > row["Sincere"] else 0, axis=1
            )
            pllm["Sincere"] = 1 - pllm["Ironic"]
            prior_r = {
                rs: pllm[rs].to_numpy().reshape(-1, 1) for rs in rhetorical_strategies
            }
    elif rsa_type == "rsc_rsa":
        if prior_type == "uniform":
            prior_r = (
                np.ones(kmeans.cluster_centers_.shape[0])
                / kmeans.cluster_centers_.shape[0]
            )
            prior_r = {
                cluster_id: prior_r[cluster_id].reshape(-1, 1)
                for cluster_id in range(len(prior_r))
            }
        elif prior_type == "P_r_c_u":
            cluster_ids = kmeans.predict(embeddings)
            unique, counts = np.unique(cluster_ids, return_counts=True)
            assert (np.sort(unique) == unique).all()
            prior_r = counts / counts.sum()
            prior_r = {
                cluster_id: prior_r[cluster_id].reshape(-1, 1)
                for cluster_id in range(len(prior_r))
            }
    return prior_r


# Utility functions for I/O


def get_prob_name(
    prob_type: str,
    rsa_type: str,
    rhetorical_strategy: str = None,
    cluster_id: int = None,
) -> str:
    if rsa_type == "llm_rsa":
        if prob_type.startswith("L"):
            return f"P{prob_type}_{rsa_type}(m|c,u)"
        elif prob_type.startswith("S"):
            return f"P{prob_type}_{rsa_type}(u|c,m)"
    elif rsa_type == "rsa_two":
        rs_string = (
            f",r={rhetorical_strategy}" if rhetorical_strategy is not None else ""
        )
        if prob_type.startswith("L"):
            return f"P{prob_type}_{rsa_type}(m|c,u{rs_string})"
        elif prob_type.startswith("S"):
            return f"P{prob_type}_{rsa_type}(u|c,m{rs_string})"
    elif rsa_type == "rsc_rsa":
        cluster_id_str = (
            f",r(cluster_id)={cluster_id}" if cluster_id is not None else ""
        )
        if prob_type.startswith("L"):
            return f"P{prob_type}_{rsa_type}(m|c,u{cluster_id_str})"
        elif prob_type.startswith("S"):
            return f"P{prob_type}_{rsa_type}(u|c,m{cluster_id_str})"
    else:
        raise ValueError(f"Invalid rsa_type: {rsa_type}")


def format_prob_jsonlines(
    prob: np.array,
    utterances: list[str],
    meanings: list[str],
    context: str,
    prob_type: str,
    prob_name: str,
    elt: dict,
):
    elt = deepcopy(elt)
    # assert len(meanings) != len(utterances)
    if prob_type.startswith("L"):
        conditional_rvs, target_rvs = utterances, meanings
    elif prob_type.startswith("S"):
        conditional_rvs, target_rvs = meanings, utterances
    else:
        raise ValueError(f"Invalid prob_type: {prob_type}")
    if prob.shape == (len(target_rvs), len(conditional_rvs)):
        prob = prob.T
    assert prob.shape == (len(conditional_rvs), len(target_rvs))
    if prob_type not in elt:
        elt[prob_type] = {}
    elt[prob_type][prob_name] = {}
    for i, conditional in enumerate(conditional_rvs):
        elt[prob_type][prob_name][str((context, conditional))] = dict(
            zip(target_rvs, prob[i, :].tolist())
        )
    return elt


def write_prob(
    prob_type: str,
    rsa_type: str,
    run_name: str,
    elt: dict,
    config: dict,
) -> dict:
    run_name = "" if run_name is None else f"_{run_name}"
    append_jsonlines(
        dict_to_write=[elt],
        output_directory=REPO_PATH / config["output_dir"],
        file_name=f"{prob_type}_{rsa_type}{run_name}.jsonl",
    )


# Main RSA functions

# Standard RSA equations


def get_prob_types(n: int = 1) -> list[str]:
    probs_types = ["L0"]
    for i in range(n):
        previous = int(probs_types[-1].replace("L", ""))
        probs_types.append(f"S{previous+1}")
        probs_types.append(f"L{previous+1}")
    return probs_types


def pragmatic_listener(
    prior_meaning: np.ndarray, ps1: np.ndarray, num_utterances: int, num_meanings: int
) -> np.ndarray:
    assert prior_meaning.shape == (1, num_meanings)
    assert ps1.shape == (num_meanings, num_utterances)
    pl1_unormalized = ps1.T * prior_meaning
    normalizing_constants_l1 = pl1_unormalized.sum(axis=-1, keepdims=True)
    pl1 = pl1_unormalized / normalizing_constants_l1
    return pl1


def pragmatic_speaker(
    prior_utt: np.ndarray,
    pl0: np.ndarray,
    alpha: float,
    num_utterances: int,
    num_meanings: int,
) -> np.ndarray:
    assert prior_utt.shape == (num_utterances, 1)
    if np.any(pl0 <= 0):
        warnings.warn(
            f"PL0 should be positive, got {pl0[pl0 <= 0]} with shape {pl0.shape}."
            + "This is probably due to some floating point magic."
            + "Applying absolute value."
        )
        pl0 = np.abs(pl0)
    ps1_unormalized = (prior_utt * (pl0**alpha)).T
    assert ps1_unormalized.shape == (
        num_meanings,
        num_utterances,
    )
    normalizing_constants_s1 = ps1_unormalized.sum(axis=-1, keepdims=True)
    ps1 = ps1_unormalized / normalizing_constants_s1
    return ps1


def literal_listener(
    pllm: np.ndarray,
    prior_meaning: np.ndarray,
    num_utterances: int,
    num_meanings: int,
) -> np.ndarray:
    assert prior_meaning.shape == (1, num_meanings), f"{prior_meaning.shape=}"
    assert pllm.shape == (num_utterances, num_meanings), f"{pllm.shape=}"
    # pl0_unormalized = pllm * prior_meaning
    # normalizing_constants = pl0_unormalized.sum(axis=-1, keepdims=True)
    # pl0 = pl0_unormalized / normalizing_constants
    pl0 = pllm
    return pl0


def standard_rsa_equations(
    pllm: np.ndarray,
    prior_meaning: np.ndarray,
    prior_utt: np.ndarray,
    alpha: float,
    num_utterances: int,
    num_meanings: int,
) -> dict[str, np.ndarray]:
    # L0
    pl0 = literal_listener(
        pllm=pllm,
        prior_meaning=prior_meaning,
        num_utterances=num_utterances,
        num_meanings=num_meanings,
    )
    # S1
    ps1 = pragmatic_speaker(
        pl0=pl0,
        prior_utt=prior_utt,
        alpha=alpha,
        num_utterances=num_utterances,
        num_meanings=num_meanings,
    )
    # L1
    pl1 = pragmatic_listener(
        ps1=ps1,
        prior_meaning=prior_meaning,
        num_utterances=num_utterances,
        num_meanings=num_meanings,
    )
    return {"L0": pl0, "S1": ps1, "L1": pl1}


def standard_rsa_equations_up_to_n(
    pllm: np.ndarray,
    prior_meaning: np.ndarray,
    prior_utt: np.ndarray,
    alpha: float,
    num_utterances: int,
    num_meanings: int,
    n: int = 1,
) -> dict[str, np.ndarray]:
    assert n >= 1
    rsa_probs = {}
    for i in range(n):
        # Why i + 2?
        # i = 0 we should have L0, S1, L1
        # i = 1 we should have S2, L2
        # i = 2 we should have S3, L3
        if i == 0:
            rsa_prob = standard_rsa_equations(
                pllm=pllm,
                prior_meaning=prior_meaning,
                prior_utt=prior_utt,
                alpha=alpha,
                num_utterances=num_utterances,
                num_meanings=num_meanings,
            )
            rsa_probs = {**rsa_prob}
        else:
            speaker = pragmatic_speaker(
                prior_utt=prior_utt,
                pl0=rsa_probs[f"L{i}"],
                alpha=alpha,
                num_utterances=num_utterances,
                num_meanings=num_meanings,
            )
            listener = pragmatic_listener(
                prior_meaning=prior_meaning,
                ps1=speaker,
                num_utterances=num_utterances,
                num_meanings=num_meanings,
            )
            rsa_probs = {
                **rsa_probs,
                **{f"S{i+1}": speaker, f"L{i+1}": listener},
            }
    return rsa_probs


# LLM RSA: Use the standard RSA equations with the LLM probabilities


def llm_rsa(elt: dict, config: dict, run_name: str = None) -> None:
    if "llm_rsa" not in config:
        warnings.warn(
            "llm_rsa is not in the config, skipping LLM RSA.",
        )
        return
    # Fetch everything required for LLM RSA
    pllm = get_pllm(
        elt=elt,
        rsa_type="llm_rsa",
    )
    pllm = pllm.T
    # Get C, M, U (Not the university! Ha.)
    # NOTE: Ensure that the order of columns and rows is unchanged
    meanings = pllm.columns
    num_meanings = len(meanings)
    utterances = get_utterances(pllm=pllm)
    num_utterances = len(utterances)
    context = get_context(pllm=pllm)
    # Convert to numpy and get priors
    pllm = pllm.to_numpy()
    assert pllm.shape == (num_utterances, num_meanings)
    prior_meaning = get_meaning_prior(
        elt=elt,
        meanings=meanings,
        prior_type=config["llm_rsa"]["prior_meaning"],
    )
    assert prior_meaning.shape == (1, num_meanings)
    prior_utt = get_utterance_prior(
        num_utterances=num_utterances,
        utterances=utterances,
        prior_type=config["llm_rsa"]["prior_utterance"],
        elt=elt,
        config=config,
        prior_utterance_bias=config["llm_rsa"].get("prior_utterance_bias"),
    )
    assert prior_utt.shape == (num_utterances, 1)
    # Begin RSA!
    probs = standard_rsa_equations_up_to_n(
        pllm=pllm,
        prior_meaning=prior_meaning,
        prior_utt=prior_utt,
        alpha=config["llm_rsa"]["alpha"],
        num_utterances=num_utterances,
        num_meanings=num_meanings,
        n=config["llm_rsa"]["n"],
    )
    # Save
    for prob_type, prob in probs.items():
        prob_name = get_prob_name(prob_type=prob_type, rsa_type="llm_rsa")
        elt = format_prob_jsonlines(
            prob=prob,
            utterances=utterances,
            meanings=meanings,
            context=context,
            prob_type=prob_type,
            prob_name=prob_name,
            elt=elt,
        )
        write_prob(
            prob_type=prob_type,
            rsa_type="llm_rsa",
            run_name=run_name,
            elt=elt,
            config=config,
        )


# RSA^2: Use the standard RSA equations with the LLM probabilities conditioned on the rhetorical strategy and marginalize over rhetorical strategies


def marginalize_rhetorical_strategies(
    probs: dict[str, np.ndarray],
    prior_rs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    # Multiple by prior
    for rs, prob in probs.items():
        for prob_type, prob in prob.items():
            if not prob_type.startswith("L"):
                continue
            probs[rs][prob_type] = prob * prior_rs[rs]
    # Marginalize by summing
    marginalized_probs = defaultdict(int)
    for prob in probs.values():
        for prob_type, prob in prob.items():
            if not prob_type.startswith("L"):
                continue
            marginalized_probs[prob_type] += prob
    return marginalized_probs


def rsa_two(
    elt: dict,
    config: dict,
    run_name: str = None,
) -> None:
    if "rsa_two" not in config:
        warnings.warn(
            "rsa_two is not in the config, skipping (RSA)^2.",
        )
        return
    # Fetch everything required for (RSA)^2
    pllm = get_pllm(
        elt=elt,
        rsa_type="rsa_two",
    )
    pllm.columns = pd.MultiIndex.from_tuples([eval(x) for x in pllm.columns])
    pllm = pllm.T
    # Do standard RSA for each rhetorical strategy
    probs = {}  # Use this later to save the probs to jsonlines
    rs_probs = {}  # Use this to marginalize
    rhetorical_strategies = get_rhetorical_strategies(pllm=pllm)
    for rs in rhetorical_strategies:
        # Filter PLLM(m|c,u,r) for each rhetorical strategy
        pllm_r = pllm.xs(rs, level=2, axis=0)
        context = get_context(pllm=pllm_r)
        meanings = pllm_r.columns
        num_meanings = len(meanings)
        utterances = get_utterances(pllm=pllm_r)
        num_utterances = len(utterances)
        # Convert to numpy and get priors
        pllm_r = pllm_r.to_numpy()
        assert pllm_r.shape == (num_utterances, num_meanings)
        # Filter PLLM(m|c,r) for each rhetorical strategy
        prior_meaning = get_meaning_prior(
            elt=elt,
            meanings=meanings,
            prior_type=config["rsa_two"]["prior_meaning"],
            rhetorical_strategy=rs,
        )
        assert prior_meaning.shape == (1, num_meanings)
        prior_utt = get_utterance_prior(
            num_utterances=num_utterances,
            utterances=utterances,
            prior_type=config["rsa_two"]["prior_utterance"],
            elt=elt,
            config=config,
            prior_utterance_bias=config["rsa_two"].get("prior_utterance_bias"),
        )
        assert prior_utt.shape == (num_utterances, 1)
        # Run standard RSA equations on the filtered probabilities
        rs_probs[rs] = standard_rsa_equations_up_to_n(
            pllm=pllm_r,
            prior_meaning=prior_meaning,
            prior_utt=prior_utt,
            alpha=config["rsa_two"]["alpha"],
            num_utterances=num_utterances,
            num_meanings=num_meanings,
            n=config["rsa_two"]["n"],
        )
    # Save in a format that's easier for append_jsonlines
    prob_types = get_prob_types(n=config["rsa_two"]["n"])
    probs = {
        prob_type: {
            get_prob_name(
                prob_type=prob_type, rsa_type="rsa_two", rhetorical_strategy=rs
            ): rs_probs[rs][prob_type]
            for rs in rs_probs.keys()
        }
        for prob_type in prob_types
    }
    # Marginalize at L0 and L1 level
    prior_rs = get_rs_prior(
        rsa_type="rsa_two",
        prior_type=config["rsa_two"]["prior_rs"],
        elt=elt,
        rhetorical_strategies=rhetorical_strategies,
    )
    assert len(prior_rs) == len(rhetorical_strategies) and all(
        v.shape == (num_utterances, 1) or v.shape == (1, 1) for v in prior_rs.values()
    )
    marginalized_probs = marginalize_rhetorical_strategies(
        probs=rs_probs,
        prior_rs=prior_rs,
    )
    probs = {
        prob_type: (
            {
                **probs[prob_type],
                **{
                    get_prob_name(
                        prob_type=prob_type, rsa_type="rsa_two"
                    ): marginalized_probs[prob_type]
                },
            }
            if prob_type in marginalized_probs
            else probs[prob_type]
        )
        for prob_type in prob_types
    }
    # Save both the non-marginalized and marginalized probs to the same prob_type file
    for prob_type in prob_types:
        elt_to_write = None
        for prob_name in probs[prob_type].keys():
            elt_to_write = format_prob_jsonlines(
                prob=probs[prob_type][prob_name],
                utterances=utterances,
                meanings=meanings,
                context=context,
                prob_type=prob_type,
                prob_name=prob_name,
                elt=elt if elt_to_write is None else elt_to_write,
            )
        write_prob(
            prob_type=prob_type,
            rsa_type="rsa_two",
            run_name=run_name,
            elt=elt_to_write,
            config=config,
        )


# RSC-RSA: Induce the rhetorical strategies by clustering the utterances and then apply the same logic as (RSA)^2


def rsc_rsa(
    elt: dict,
    config: dict,
    embedding_model: SentenceTransformer,
    run_name: str = None,
) -> None:
    if "rsc_rsa" not in config:
        warnings.warn(
            "rsc_rsa is not in the config, skipping RSC-RSA.",
        )
        return
    # Fetch everything required for the clustering LLM RSA
    pllm = get_pllm(
        elt=elt,
        rsa_type="rsc_rsa",
    )
    pllm = pllm.T
    number_of_clusters = config["rsc_rsa"]["number_of_clusters"]
    # NOTE: Ensure that the order of columns and rows is unchanged
    context = get_context(pllm=pllm)
    utterances = get_utterances(
        pllm=pllm
    )  # NOTE: This includes both the original and alternative utterance
    num_utterances = len(utterances)
    meanings = pllm.columns
    num_meanings = len(meanings)
    # Extract embeddings for each utterance
    embeddings = embedding_model.encode(utterances, normalize_embeddings=True)
    # Run kmeans
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    cluster_centroids = kmeans.cluster_centers_
    # Compute Fr_muc
    cluster_ids = kmeans.predict(embeddings)
    assert cluster_ids.min() == 0 and cluster_ids.max() == number_of_clusters - 1
    masks = np.arange(number_of_clusters)[:, None] == cluster_ids
    # TODO: Vectorize with matrix multiply
    Fr_muc = []
    for mask in masks:
        fr_muc = pllm[mask].sum(axis=0) / mask.sum()
        Fr_muc.append(fr_muc)
    Fr_muc = np.array(Fr_muc)
    assert Fr_muc.shape == (
        number_of_clusters,
        num_meanings,
    )
    if (
        config["rsc_rsa"]["f_rmuc_formula"]
        == "mean_{i in cluster} PLLM(m|c,ui) / |cluster|"
    ):
        Fr_muc = (
            np.ones((number_of_clusters, num_utterances, num_meanings))
            * Fr_muc[:, None, :]
        )
    elif (
        config["rsc_rsa"]["f_rmuc_formula"]
        == "mean_{i in cluster} PLLM(m|c,ui) / |cluster| * similarity"
    ):
        # Compute similarity and softmax it
        cosine_sim = embedding_model.similarity(embeddings, cluster_centroids.tolist())
        cosine_sim = cosine_sim.numpy()
        cosine_sim = cosine_sim.T
        assert cosine_sim.shape == (
            number_of_clusters,
            num_utterances,
        )
        # Use the similarity values to weight the fr_muc for each u (so that f_r is not the same across all utterances)
        Fr_muc = cosine_sim[:, :, None] * Fr_muc[:, None, :]
    elif (
        config["rsc_rsa"]["f_rmuc_formula"]
        == "(mean_{i in cluster} PLLM(m|c,ui) / |cluster|) / PLLM(m|c,u)"
    ):
        # Compute Fr_muc as the ratio of the cluster mean and the utterance PLLM(m|c,u)
        Fr_muc = Fr_muc[:, None, :] / pllm.to_numpy()[None, :, :]
    else:
        raise ValueError(
            f"Invalid f_rmuc_formula: {config['rsc_rsa']['f_rmuc_formula']}"
        )
    assert Fr_muc.shape == (
        number_of_clusters,
        num_utterances,
        num_meanings,
    )
    # Do standard RSA for each rhetorical strategy cluster
    probs = {}  # Use this later to save the probs to jsonlines
    cluster_probs = {}  # Use this to marginalize
    for cluster_id in range(number_of_clusters):
        # Get corresponding f_rmuc
        f_rmuc = Fr_muc[cluster_id]
        # Filter PLLM(m|c,r) for each rhetorical strategy
        prior_meaning = get_meaning_prior(
            elt=elt,
            meanings=meanings,
            prior_type=config["rsc_rsa"]["prior_meaning"],
        )
        assert prior_meaning.shape == (1, num_meanings)
        prior_utt = get_utterance_prior(
            num_utterances=num_utterances,
            utterances=utterances,
            prior_type=config["rsc_rsa"]["prior_utterance"],
            elt=elt,
            config=config,
            prior_utterance_bias=config["rsc_rsa"].get("prior_utterance_bias"),
        )
        assert prior_utt.shape == (num_utterances, 1)
        # Run standard RSA equations on the filtered probabilities
        cluster_probs[cluster_id] = standard_rsa_equations_up_to_n(
            pllm=f_rmuc,
            prior_meaning=prior_meaning,
            prior_utt=prior_utt,
            alpha=config["rsc_rsa"]["alpha"],
            num_utterances=num_utterances,
            num_meanings=num_meanings,
            n=config["rsc_rsa"]["n"],
        )
    # Save in a format that's easier for append_jsonlines
    prob_types = get_prob_types(n=config["rsc_rsa"]["n"])
    probs = {
        prob_type: {
            get_prob_name(
                prob_type=prob_type,
                rsa_type="rsc_rsa",
                cluster_id=cluster_id,
            ): cluster_probs[cluster_id][prob_type]
            for cluster_id in range(number_of_clusters)
        }
        for prob_type in prob_types
    }
    # Marginalize at L0 and L1 level
    prior_rs_cluster = get_rs_prior(
        rsa_type="rsc_rsa",
        prior_type=config["rsc_rsa"]["prior_rs"],
        elt=elt,
        kmeans=kmeans,
        embeddings=embeddings,
    )
    assert len(prior_rs_cluster) == number_of_clusters and all(
        v.shape == (num_utterances, 1) or v.shape == (1, 1)
        for v in prior_rs_cluster.values()
    )
    marginalized_probs = marginalize_rhetorical_strategies(
        probs=cluster_probs,
        prior_rs=prior_rs_cluster,
    )
    probs = {
        prob_type: (
            {
                **probs[prob_type],
                **{
                    get_prob_name(
                        prob_type=prob_type, rsa_type="rsc_rsa"
                    ): marginalized_probs[prob_type]
                },
            }
            if prob_type in marginalized_probs
            else probs[prob_type]
        )
        for prob_type in prob_types
    }
    # Save both the non-marginalized and marginalized probs to the same prob_type file
    for prob_type in prob_types:
        elt_to_write = None
        for prob_name in probs[prob_type].keys():
            elt_to_write = format_prob_jsonlines(
                prob=probs[prob_type][prob_name],
                utterances=utterances,
                meanings=meanings,
                context=context,
                prob_type=prob_type,
                prob_name=prob_name,
                elt=elt if elt_to_write is None else elt_to_write,
            )
        write_prob(
            prob_type=prob_type,
            rsa_type="rsc_rsa",
            run_name=run_name,
            elt=elt_to_write,
            config=config,
        )


def rsa(config: dict) -> list[dict]:
    seed_everything(seed=config["seed"])
    # Implement all RSA versions via vectorization
    # Should include all the LLM probs i.e. PLLM(m|c), PLLM(m|c,u), PLLM(m|c,u,r), PLLM(r|c,u), PLLM(u|c)
    llm_probs = read_jsonlines(REPO_PATH / config["llm_probs_path"])
    # Load models needed for all RSA versions before for loop
    embedding_model = None
    for rsa_config in config["rsa_runs"]:
        if rsa_config["run_type"] == "rsc_rsa":
            embedding_model = rsa_config["rsc_rsa"]["embedding_model"]
            embedding_model = SentenceTransformer(
                embedding_model,
                trust_remote_code=True,
            )
    for elt in tqdm(llm_probs):
        for rsa_config in config["rsa_runs"]:
            if rsa_config["run_type"] == "llm_rsa":
                # LLM RSA
                rsa_config["output_dir"] = config["output_dir"]
                run_name = rsa_config["llm_rsa"].get("run_name")
                llm_rsa(elt=elt, config=rsa_config, run_name=run_name)
            elif rsa_config["run_type"] == "rsa_two":
                # (RSA)^2
                rsa_config["output_dir"] = config["output_dir"]
                run_name = rsa_config["rsa_two"].get("run_name")
                rsa_two(elt=elt, config=rsa_config, run_name=run_name)
            elif rsa_config["run_type"] == "rsc_rsa":
                # RSC RSA
                rsa_config["output_dir"] = config["output_dir"]
                run_name = rsa_config["rsc_rsa"].get("run_name")
                rsc_rsa(
                    elt=elt,
                    config=rsa_config,
                    embedding_model=embedding_model,
                    run_name=run_name,
                )
            else:
                warnings.warn(
                    f"Invalid rsa_run_type: {rsa_config['run_type']}. "
                    + "Skipping this RSA run.",
                )


@log_run
def run_config(config: dict):
    if config["run_type"] == "run_all_rsa_versions":
        rsa(config)
    else:
        raise ValueError(f"Invalid run_type: {config['run_type']}")


@hydra.main(
    version_base=None,
)
def main(config: dict) -> None:
    run_config(config=config)


if __name__ == "__main__":
    main()
