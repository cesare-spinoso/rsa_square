import os
from itertools import product
from typing import Literal

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from tqdm import tqdm

from src import REPO_PATH
from src.utils.helpers import seed_everything

# Helper code for all experiments


def create_train_test_split(
    interpretation_w_irony: pd.DataFrame,
) -> tuple[pd.DataFrame]:
    # Train/val/test split for the ironic weather utterances dataset
    train_ids, test_ids = train_test_split(
        interpretation_w_irony["workerID"].unique(), train_size=0.6, random_state=42
    )
    val_ids, test_ids = train_test_split(test_ids, train_size=0.5, random_state=42)
    train = interpretation_w_irony[interpretation_w_irony["workerID"].isin(train_ids)]
    val = interpretation_w_irony[interpretation_w_irony["workerID"].isin(val_ids)]
    test = interpretation_w_irony[interpretation_w_irony["workerID"].isin(test_ids)]
    return train, val, test


# (RSA)^2 specific functions


def preprocess_data(
    df_prior: pd.DataFrame, interpretation_w_irony: pd.DataFrame
) -> pd.DataFrame:
    # Preprocess prior and interpretation data:
    # 1. Format context columns such that they respect {context_type}_{image_id} format
    # 2. Derive probabilities from the prior with Laplace-1 smoothing
    df_prior["context"] = df_prior.apply(
        lambda x: x["imageCategory"] + "_" + str(x["imageID"]), axis=1
    )
    df_prior = (
        df_prior.groupby(["context", "stateRating"]).size().reset_index(name="counts")
    )
    for image_category in df_prior["context"].unique():
        for state_rating in range(1, 6):
            if (
                state_rating
                not in df_prior[df_prior["context"] == image_category][
                    "stateRating"
                ].values
            ):
                df_prior.loc[len(df_prior)] = [image_category, state_rating, 0]
    df_prior = df_prior.sort_values(by=["context", "stateRating"])
    df_prior["new_counts"] = df_prior.groupby("context")["counts"].transform(
        lambda x: x
        + 1  # Use Laplace-1 smoothing for the priors since this was done in Let’s talk (ironically) about the weather: Modeling verbal irony, Kao et al. (2015)
    )
    df_prior["P(s)"] = df_prior.groupby("context")["new_counts"].transform(
        lambda x: x / x.sum()
    )
    df_prior = df_prior[["context", "stateRating", "P(s)"]]
    interpretation_w_irony["context"] = interpretation_w_irony.apply(
        lambda x: x["imageCategory"] + "_" + str(x["imageID"]), axis=1
    )
    return df_prior


def create_category_mappings() -> tuple[dict[str, int]]:
    context_to_idx = {
        v: i
        for i, v in enumerate(
            [
                "amazing_1",
                "amazing_2",
                "amazing_3",
                "ok_4",
                "ok_5",
                "ok_6",
                "terrible_7",
                "terrible_8",
                "terrible_9",
            ]
        )
    }
    utterances_to_idx = {
        v: i for i, v in enumerate(["terrible", "bad", "ok", "good", "amazing"])
    }
    meanings_to_idx = {v: i for i, v in enumerate(range(1, 6))}
    rhetorical_strategies_to_idx = {v: i for i, v in enumerate(["sincere", "irony"])}
    return (
        context_to_idx,
        utterances_to_idx,
        meanings_to_idx,
        rhetorical_strategies_to_idx,
    )


def get_categories_product_encoder(
    rhetorical_strategies_to_idx: dict[str, int],
    context_to_idx: dict[str, int],
    utterances_to_idx: dict[str, int],
) -> OneHotEncoder:
    # Categorize the context, utterance and rhetorical strategies into one-hot encodings
    # This is done so we can pass symbolic features to the rhetorical function network
    categories_product = [
        x
        for x in product(
            rhetorical_strategies_to_idx.keys(),
            context_to_idx.keys(),
            utterances_to_idx.keys(),
        )
    ]
    categories_product_encoder = OneHotEncoder()
    categories_product_encoder.fit(categories_product)
    return categories_product_encoder


class RhetoricalFunctionNetwork(nn.Module):
    # Rhetorical strategy function network
    # Implemented as a feedforward neural network
    # The type is slightly different from the one in our paper for vectorization convenience
    # In the paper we define f_r : C x U x M -> [0,1]
    # Here we define a single function f: R x C x U -> [0,1]^|M|
    def __init__(self, architecture: str):
        super().__init__()
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        # Different architectures for the network
        # We tested the first one on validation and noticed that it was underfitting
        # Adding an extra layer was enough to get it to fit
        if architecture == "rs x c x u -> linear -> sigmoid -> m":
            self.stack = nn.Sequential(
                # input dim: 16 [irony, sincere] + [terrible, ok, amazing] x 3 + [terrible, bad, ok, good, amazing] with one-hot encoding
                # output dim: 5 possible meanings, sigmoid for each output dim
                # assuming an ordinal order for the meanings: i.e state 1 to 5 == index 0 to 4
                nn.Linear(16, 5),
                nn.Sigmoid(),
            )
        elif (
            architecture == "rs x c x u -> linear -> sigmoid -> linear -> sigmoid -> m"
        ):
            self.stack = nn.Sequential(
                nn.Linear(16, 16),
                nn.Sigmoid(),
                nn.Linear(16, 5),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Architecture {architecture} not supported")

    def forward(self, x):
        rf_values = self.stack(x)
        return rf_values


def get_expected_shapes(len_train: int, architecture: str) -> tuple[tuple[int]]:
    # Get the expected shapes of all the tensors throughout training
    # This is for sanity checking purposes
    if architecture in [
        "rs x c x u -> linear -> sigmoid -> m",
        "rs x c x u -> linear -> sigmoid -> linear -> sigmoid -> m",
    ]:
        encoded_input_shape = (len_train, 2, 9, 5, 16)
        rf_values_shape = (len_train, 2, 9, 5, 5)
        priors_shape = (1, 1, 9, 1, 5)
        l0_shape = (len_train, 2, 9, 5, 5)
        l1_shape = (len_train, 2, 9, 5, 5)
        final_dimension = 4
        utterance_dim = 3
        context_dim = 2
        return (
            encoded_input_shape,
            rf_values_shape,
            priors_shape,
            l0_shape,
            l1_shape,
            final_dimension,
            utterance_dim,
            context_dim,
        )
    else:
        raise ValueError(f"Architecture {architecture} not supported")


def marginalize(
    pl: torch.Tensor,
    data: pd.DataFrame,
    utterance_dim: int,
    utterances_to_idx: dict[str, int],
    context_dim: int,
    context_to_idx: dict[str, int],
) -> None:
    # Marginalize the listener probability (either literal or pragmatic)
    # by marginalizing over r using the equation from our paper
    # Dimension manipulation to get things to vectorize correctly
    # when marginalizing over R
    ### Can double check all this code with:
    # print(pl1[0, ...]) # before the gather
    # print(utterances_to_idx[train["utterance"].iloc[0]])
    # print(pl1[0, ...]) # after the gather
    ###
    pl = torch.stack(
        [
            torch.index_select(
                pl[i, ...],
                dim=utterance_dim - 1,
                index=torch.tensor([utterances_to_idx[x]]),
            )
            for i, x in enumerate(data["utterance"])
        ],
        dim=0,
    ).squeeze(utterance_dim)
    pl = torch.stack(
        [
            torch.index_select(
                pl[i, ...],
                dim=context_dim - 1,
                index=torch.tensor([context_to_idx[x]]),
            )
            for i, x in enumerate(data["context"])
        ],
        dim=0,
    ).squeeze(context_dim)
    assert pl.shape == (len(data), 2, 5)
    # Marginalize over the rhetorical strategy
    p_rs = torch.tensor(
        np.hstack(
            [
                1 - data["ironyRating"].values.reshape(-1, 1),
                data["ironyRating"].values.reshape(-1, 1),
            ]
        )
    )
    assert p_rs.shape == (len(data), 2)
    l = pl * p_rs.unsqueeze(2)
    pl = l.sum(axis=1)
    assert pl.shape == (len(data), 5)
    return pl


def convert_rf_vals_to_df(
    rhetorical_strategies_to_idx: dict[str, int],
    context_to_idx: dict[str, int],
    utterances_to_idx: dict[str, int],
    meanings_to_idx: dict[str, int],
    rf_values: torch.Tensor,
) -> pd.DataFrame:
    # Get the outputs of the network for each (c,u,r) as a dataframe
    # Note that these values will be the same REGARDLESS of the human subject
    # The probabilities differ when the marginalization is done because P(r|c,u) is user specific
    rf_values_decoded = {
        (rs, c, u, m): float(rf_values[0, idx_rs, idx_c, idx_u, idx_m].detach().item())
        for rs, idx_rs in rhetorical_strategies_to_idx.items()
        for c, idx_c in context_to_idx.items()
        for u, idx_u in utterances_to_idx.items()
        for m, idx_m in meanings_to_idx.items()
    }
    rf_values_decoded = pd.DataFrame.from_dict(
        rf_values_decoded, orient="index", columns=["rf_values"]
    ).reset_index()
    rf_values_decoded["rhetorical_strategy"] = rf_values_decoded["index"].apply(
        lambda x: x[0]
    )
    rf_values_decoded["context"] = rf_values_decoded["index"].apply(lambda x: x[1])
    rf_values_decoded["utterance"] = rf_values_decoded["index"].apply(lambda x: x[2])
    rf_values_decoded["stateRating"] = rf_values_decoded["index"].apply(lambda x: x[3])
    rf_values_decoded = rf_values_decoded.drop(columns=["index"])
    rf_values_decoded = rf_values_decoded[
        [
            "rhetorical_strategy",
            "context",
            "utterance",
            "stateRating",
            "rf_values",
        ]
    ]
    return rf_values_decoded


def rsa_w_rf_network(
    rhetorical_strategies_to_idx: dict[str, int],
    context_to_idx: dict[str, int],
    utterances_to_idx: dict[str, int],
    meanings_to_idx: dict[str, int],
    data: pd.DataFrame,
    rf_network: RhetoricalFunctionNetwork,
    architecture: str,
    categories_product_encoder: OneHotEncoder,
    df_prior: pd.DataFrame,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    # RSA equations PL0 -> PS1 -> PL1 using the RF network, all in vectorized form
    # Compute rhetorical function values
    (
        encoded_input_shape,
        rf_values_shape,
        priors_shape,
        l0_shape,
        l1_shape,
        final_dimension,
        utterance_dim,
        context_dim,
    ) = get_expected_shapes(len_train=len(data), architecture=architecture)
    encoded_input = (
        categories_product_encoder.transform(
            [
                [rs, c, u]
                for _ in range(len(data))
                for rs in rhetorical_strategies_to_idx.keys()
                for c in context_to_idx.keys()
                for u in utterances_to_idx.keys()
            ]
        )
        .toarray()
        .reshape(encoded_input_shape)
    )
    rf_values = rf_network(torch.tensor(encoded_input).float())
    assert rf_values.shape == rf_values_shape
    rf_values_decoded = convert_rf_vals_to_df(
        rhetorical_strategies_to_idx=rhetorical_strategies_to_idx,
        context_to_idx=context_to_idx,
        utterances_to_idx=utterances_to_idx,
        meanings_to_idx=meanings_to_idx,
        rf_values=rf_values,
    )
    # Compute PL0(m|c,u,r)
    priors = torch.tensor(
        np.array(
            [
                df_prior[df_prior["context"] == c]["P(s)"].values
                for c in context_to_idx.keys()
            ]
        )
    )
    priors = priors[None, None].unsqueeze(-2)
    assert priors.shape == priors_shape
    l0 = rf_values * priors
    assert l0.shape == l0_shape
    pl0 = l0 / l0.sum(axis=final_dimension, keepdims=True)
    pl0_marg = marginalize(
        pl=pl0,
        data=data,
        utterance_dim=utterance_dim,
        utterances_to_idx=utterances_to_idx,
        context_dim=context_dim,
        context_to_idx=context_to_idx,
    )
    assert pl0_marg.shape == (len(data), 5)
    # Compute PS1(u|c,m,r)
    alpha = 1.0
    s1 = torch.swapaxes((pl0**alpha), final_dimension - 1, final_dimension)
    ps1 = s1 / s1.sum(axis=4, keepdims=True)
    # Compute PL1(m|c,u,r)
    l1 = torch.swapaxes(ps1, final_dimension - 1, final_dimension) * priors
    pl1 = l1 / l1.sum(axis=final_dimension, keepdims=True)
    assert pl1.shape == l1_shape
    pl1_marg = marginalize(
        pl=pl1,
        data=data,
        utterance_dim=utterance_dim,
        utterances_to_idx=utterances_to_idx,
        context_dim=context_dim,
        context_to_idx=context_to_idx,
    )
    return rf_values_decoded, pl0_marg, pl1_marg


def insert_probabilities(
    meanings_to_idx: dict[str, int],
    data: pd.DataFrame,
    pl: torch.Tensor,
    listener: Literal["L0", "L1"],
):
    # Helper function to insert the probabilities into the dataframe in the training loop
    for m, idx in meanings_to_idx.items():
        data.insert(
            len(data.columns),
            f"P{listener}({m}|c,u)",
            pl[:, idx].detach().numpy().reshape(-1),
        )


def train_loop(
    rhetorical_strategies_to_idx: dict[str, int],
    context_to_idx: dict[str, int],
    utterances_to_idx: dict[str, int],
    meanings_to_idx: dict[str, int],
    train: pd.DataFrame,
    val: pd.DataFrame,
    rf_network: RhetoricalFunctionNetwork,
    architecture: str,
    categories_product_encoder: OneHotEncoder,
    df_prior: pd.DataFrame,
    output_dir: str,
    lr: float = 1e-3,
    weight_decay: float = 0,
    epochs: int = 10000,
) -> None:
    # Custom training loop for the RF network
    # We use the validation set to determine whether the gradient update
    # has led to a better model
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        rf_network.parameters(), lr=lr, weight_decay=weight_decay
    )
    rf_network.train()
    val_loss = None
    best_model = None
    for e in tqdm(range(epochs)):
        # Forward pass
        _, pl0, pl1 = rsa_w_rf_network(
            rhetorical_strategies_to_idx=rhetorical_strategies_to_idx,
            context_to_idx=context_to_idx,
            utterances_to_idx=utterances_to_idx,
            meanings_to_idx=meanings_to_idx,
            data=train,
            rf_network=rf_network,
            architecture=architecture,
            categories_product_encoder=categories_product_encoder,
            df_prior=df_prior,
        )
        target = torch.tensor([meanings_to_idx[s] for s in train["stateRating"]])
        loss = loss_fn(
            torch.log(pl1), target
        )  # Log is applied because NLLLoss expects log probabilities
        print(f"Training loss: {loss.item()}")
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Validate
        _, pl0, pl1 = rsa_w_rf_network(
            rhetorical_strategies_to_idx=rhetorical_strategies_to_idx,
            context_to_idx=context_to_idx,
            utterances_to_idx=utterances_to_idx,
            meanings_to_idx=meanings_to_idx,
            data=val,
            rf_network=rf_network,
            architecture=architecture,
            categories_product_encoder=categories_product_encoder,
            df_prior=df_prior,
        )
        target = torch.tensor([meanings_to_idx[s] for s in val["stateRating"]])
        loss = loss_fn(torch.log(pl1), target)
        print(f"Validation loss: {loss.item()}")
        if val_loss is None or loss < val_loss:
            val_loss = loss
            best_model = rf_network
            print("!" * 60)
            print(f"New best model found (based on val loss), saving to {output_dir}")
            torch.save(rf_network.state_dict(), REPO_PATH / output_dir / "model.pth")
    # Write the predictions of the best model on the training set to a CSV
    best_model.eval()
    rf_values, pl0, pl1 = rsa_w_rf_network(
        rhetorical_strategies_to_idx=rhetorical_strategies_to_idx,
        context_to_idx=context_to_idx,
        utterances_to_idx=utterances_to_idx,
        meanings_to_idx=meanings_to_idx,
        data=train,
        rf_network=best_model,
        architecture=architecture,
        categories_product_encoder=categories_product_encoder,
        df_prior=df_prior,
    )
    # Save the rf values
    rf_values.to_csv(REPO_PATH / output_dir / "trained_rf_values.csv", index=False)
    # Save the PL0/PL1 values
    insert_probabilities(
        meanings_to_idx=meanings_to_idx,
        data=train,
        pl=pl0,
        listener="L0",
    )
    insert_probabilities(
        meanings_to_idx=meanings_to_idx,
        data=train,
        pl=pl1,
        listener="L1",
    )
    train.to_csv(REPO_PATH / output_dir / "train_predictions.csv", index=False)


def compute_distributions(predictions_path: str) -> dict[str, pd.DataFrame]:
    # Convert the saved predictions to distribution dataframe
    # for P(m|c,u). There should be 5 * 9 * 5 = 225 rows, though
    # it's possible that the number of rows is less than that because,
    # for example, in the human derived distributions there might not
    # be any data for a particular condition pair (c,u).
    # Load the predictions
    predictions = pd.read_csv(predictions_path)
    # Compute the PL0 and PL1 distributions
    # To do so, we average over all predicted probabilities for the same (c,u)
    # This is necessary because for a given (c,u) each participant provides their own P(r|c,u)
    listeners = [f"P{l}({state}|c,u)" for l in ["L0", "L1"] for state in range(1, 6)]
    listener_dist = (
        predictions.groupby(["context", "utterance"])[listeners].mean().reset_index()
    )
    # Melt the table to create a stateRating column
    pl0 = listener_dist.melt(
        id_vars=["context", "utterance"],
        value_vars=[col for col in listeners if "L0" in col],
        var_name="stateRating",
        value_name="PL0(s|c,u)",
    )
    pl0["stateRating"] = pl0["stateRating"].str.extract(r"PL0\((\d).*\)").astype(int)
    pl0 = pl0.sort_values(by=["context", "utterance", "stateRating"]).reset_index(
        drop=True
    )
    pl1 = listener_dist.melt(
        id_vars=["context", "utterance"],
        value_vars=[col for col in listeners if "L1" in col],
        var_name="stateRating",
        value_name="PL1(s|c,u)",
    )
    pl1["stateRating"] = pl1["stateRating"].str.extract(r"PL1\((\d).*\)").astype(int)
    pl1 = pl1.sort_values(by=["context", "utterance", "stateRating"]).reset_index(
        drop=True
    )
    merged_listener_dist = pd.merge(
        pl0, pl1, on=["context", "utterance", "stateRating"]
    )
    # Compute human distributions
    human_dist = (
        predictions.groupby(["context", "utterance", "stateRating"])["stateRating"]
        .count()
        .reset_index(name="counts")
    )
    # Add 0s where stateRatings are missing
    for context in human_dist["context"].unique():
        for utterance in human_dist["utterance"].unique():
            for state_rating in range(1, 6):
                if (
                    len(
                        human_dist[
                            (human_dist["context"] == context)
                            & (human_dist["utterance"] == utterance)
                            & (human_dist["stateRating"] == state_rating)
                        ]
                    )
                    == 0
                ):
                    human_dist.loc[len(human_dist)] = [
                        context,
                        utterance,
                        state_rating,
                        0,
                    ]
    human_dist = human_dist.sort_values(by=["context", "utterance", "stateRating"])
    # Add count total
    # Unlike in the original Kao paper, we do not perform Laplace smoothing to the human distributions
    # This explains why there are most states with probability zero than in the figures of the original paper
    count_totals = (
        human_dist.groupby(["context", "utterance"])["counts"]
        .sum()
        .reset_index(name="countTotal")
    )
    human_dist = pd.merge(human_dist, count_totals, on=["context", "utterance"])
    human_dist["P(s|c,u)"] = human_dist.apply(
        lambda x: x["counts"] / x["countTotal"] if x["countTotal"] > 0 else 0, axis=1
    )
    human_dist = human_dist[["context", "utterance", "stateRating", "P(s|c,u)"]]
    return {
        "rsa_two": merged_listener_dist,
        "human_dist": human_dist,
    }


def run_rsa_two_exps(config: dict):
    # Seed experiment if a seed is present in the config
    if "seed" in config:
        seed_everything(seed=config["seed"])
    # Load data
    df_prior = pd.read_csv(REPO_PATH / config["prior_path"])
    interpretation_w_irony = pd.read_csv(
        REPO_PATH / config["interpretation_w_irony_path"]
    )
    # Preprocess data
    df_prior = preprocess_data(
        df_prior=df_prior, interpretation_w_irony=interpretation_w_irony
    )
    # Fix idx mapping for the categories
    context_to_idx, utterances_to_idx, meanings_to_idx, rhetorical_strategies_to_idx = (
        create_category_mappings()
    )
    # Create one-hot encoder for the product of categories
    categories_product_encoder = get_categories_product_encoder(
        rhetorical_strategies_to_idx=rhetorical_strategies_to_idx,
        context_to_idx=context_to_idx,
        utterances_to_idx=utterances_to_idx,
    )
    # Create train/val/test split
    train, val, test = create_train_test_split(
        interpretation_w_irony=interpretation_w_irony
    )
    # Create rhetorical function network
    rf_network = RhetoricalFunctionNetwork(architecture=config["architecture"])
    if config["train_network"]:
        # Train the network - Save the best model + train/test csv outputs
        os.makedirs(REPO_PATH / config["output_dir"], exist_ok=True)
        train_loop(
            rhetorical_strategies_to_idx=rhetorical_strategies_to_idx,
            context_to_idx=context_to_idx,
            utterances_to_idx=utterances_to_idx,
            meanings_to_idx=meanings_to_idx,
            train=train,
            val=val,
            rf_network=rf_network,
            architecture=config["architecture"],
            categories_product_encoder=categories_product_encoder,
            df_prior=df_prior,
            output_dir=config["output_dir"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            epochs=config["epochs"],
        )
    # Compute the human and model-induced distributions for the training set
    distributions = compute_distributions(
        predictions_path=REPO_PATH / config["output_dir"] / "train_predictions.csv",
    )
    for dist_name, dist_df in distributions.items():
        dist_df.to_csv(
            REPO_PATH / config["output_dir"] / f"{dist_name}_train.csv",
            index=False,
        )
    # Load best checkpoint
    rf_network.load_state_dict(
        torch.load(REPO_PATH / config["output_dir"] / "model.pth")
    )
    # Evaluate the network
    rf_network.eval()
    _, pl0, pl1 = rsa_w_rf_network(
        rhetorical_strategies_to_idx=rhetorical_strategies_to_idx,
        context_to_idx=context_to_idx,
        utterances_to_idx=utterances_to_idx,
        meanings_to_idx=meanings_to_idx,
        data=test,
        rf_network=rf_network,
        architecture=config["architecture"],
        categories_product_encoder=categories_product_encoder,
        df_prior=df_prior,
    )
    insert_probabilities(
        meanings_to_idx=meanings_to_idx,
        data=test,
        pl=pl0,
        listener="L0",
    )
    insert_probabilities(
        meanings_to_idx=meanings_to_idx,
        data=test,
        pl=pl1,
        listener="L1",
    )
    test.to_csv(REPO_PATH / config["output_dir"] / "test_predictions.csv", index=False)
    # Compute the human and model-induced distributions for the test set
    distributions = compute_distributions(
        predictions_path=REPO_PATH / config["output_dir"] / "test_predictions.csv",
    )
    for dist_name, dist_df in distributions.items():
        dist_df.to_csv(
            REPO_PATH / config["output_dir"] / f"{dist_name}_test.csv",
            index=False,
        )


# Affect-aware RSA exps


def create_semantic_indicator_function(
    context_to_idx: dict[str, int],
    utterances_to_idx: dict[str, int],
    meanings_to_idx: dict[str, int],
) -> pd.DataFrame:
    # Create the indicator function and return it as a dataframe
    semantic_understanding_domain = list(
        product(
            list(context_to_idx.keys()),
            list(meanings_to_idx.keys()),
            list(utterances_to_idx.keys()),
        )
    )
    semantic_understanding_function = {
        "terrible": [1],
        "bad": [2],
        "ok": [3],
        "good": [4],
        "amazing": [5],
    }
    semantic_indicator_function = pd.DataFrame(
        semantic_understanding_domain, columns=["context", "stateRating", "utterance"]
    )
    semantic_indicator_function["1_s_in_[[u]]"] = semantic_indicator_function.apply(
        lambda x: (
            1
            if int(x["stateRating"]) in semantic_understanding_function[x["utterance"]]
            else 0
        ),
        axis=1,
    )
    semantic_indicator_function = semantic_indicator_function[
        ["context", "utterance", "stateRating", "1_s_in_[[u]]"]
    ]
    semantic_indicator_function = semantic_indicator_function.sort_values(
        by=["context", "utterance", "stateRating"]
    )
    return semantic_indicator_function


def compute_priors(
    df_prior: pd.DataFrame,
    context_to_idx: dict[str, int],
    meanings_to_idx: dict[str, int],
) -> pd.DataFrame:
    # Compute the priors for the affect dimensions i.e. valence and arousal
    # Used the steps described in Let’s talk (ironically) about the weather: Modeling verbal irony, Kao et al. (2015)
    # Section: Experiment 1/Results
    # Run PCA on the elicited emotions
    priors = pd.read_csv(
        REPO_PATH
        / "data"
        / "exp_data"
        / "ironic_weather_utterances"
        / "priors"
        / "long.csv"
    )
    pca = PCA(n_components=2)
    x = pca.fit_transform(
        priors[["sad", "disgusted", "angry", "neutral", "content", "happy", "excited"]]
    )
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print("Ratio reported in the original Kao et al. (2015) paper: 69.14\% and 13.86%")
    # Convert the principal components to a probability distribution P(affect|s)
    x_stand = (x - x.mean(axis=0)) / x.std(axis=0)  # standardize each column
    x_cdf = scipy.stats.norm.cdf(x_stand)
    for i in range(2):
        if i == 1:
            # Why do I flip here? If you don't flip then you will see that the second
            # column probabilities after the grouping+averaging don't match the ones
            # in Figure 4 of the Kao paper
            x_cdf[:, i] = 1 - x_cdf[:, i]
        priors.insert(len(priors.columns), f"P(A_{i}|s)", x_cdf[:, i])
    affect_priors = priors.groupby("stateRating")[
        [f"P(A_{i}|s)" for i in range(2)]
    ].mean()
    affect_priors = affect_priors.reset_index()
    affect_priors = affect_priors.rename(
        columns={
            f"P(A_{i}|s)": "P(valence|s)" if i == 0 else "P(arousal|s)"
            for i in range(2)
        }
    )
    print(
        f"The following table should align approximately with Figure 4 in the Kao paper: {affect_priors}"
    )
    # So don't have to recompute them every time, compute P(valence=positive|s) AND P(valence=negative|s)
    # Also, P(arousal=high|s) AND P(arousal=low|s)
    affect_priors["valence"] = "positive"
    affect_priors["arousal"] = "high"
    flip_affect_priors = affect_priors.copy(deep=True)
    flip_affect_priors[["P(valence|s)", "P(arousal|s)"]] = (
        1 - flip_affect_priors[["P(valence|s)", "P(arousal|s)"]]
    )
    flip_affect_priors["valence"] = "negative"
    flip_affect_priors["arousal"] = "low"
    affect_priors = pd.concat([affect_priors, flip_affect_priors], axis=0)
    print(
        f"Affect priors with both positive/negative valence values and high/low arousal values: {affect_priors}"
    )
    # Create valence and arousal priors as separate dataframes
    valence_priors = affect_priors[["stateRating", "valence", "P(valence|s)"]]
    arousal_priors = affect_priors[["stateRating", "arousal", "P(arousal|s)"]]
    state_priors = df_prior.copy(deep=True)
    # Combine all priors together
    # This will make the computation in the next step "easier" (you'll see...)
    prior_variables = product(
        list(context_to_idx.keys()),
        list(meanings_to_idx.keys()),
        ["low", "high"],
        ["positive", "negative"],
    )
    merged_priors = []
    for context, state_rating, arousal, valence in prior_variables:
        state_prior = state_priors[
            (state_priors["context"] == context)
            & (state_priors["stateRating"] == state_rating)
        ]["P(s)"].values[0]
        arousal_prior = arousal_priors[
            (arousal_priors["stateRating"] == state_rating)
            & (arousal_priors["arousal"] == arousal)
        ]["P(arousal|s)"].values[0]
        valence_prior = valence_priors[
            (valence_priors["stateRating"] == state_rating)
            & (valence_priors["valence"] == valence)
        ]["P(valence|s)"].values[0]
        merged_priors.append(
            [
                context,
                state_rating,
                arousal,
                valence,
                state_prior,
                arousal_prior,
                valence_prior,
            ]
        )
    merged_priors = pd.DataFrame(
        merged_priors,
        columns=[
            "context",
            "stateRating",
            "arousal",
            "valence",
            "P(s)",
            "P(arousal|s)",
            "P(valence|s)",
        ],
    )
    print(f"All priors merged together: {merged_priors}")
    return merged_priors


def get_qud_fn(qud: str) -> callable:
    if qud == "q_state":
        return lambda s, v, a: s
    elif qud == "q_valence":
        return lambda s, v, a: v
    elif qud == "q_arousal":
        return lambda s, v, a: a
    else:
        raise ValueError(f"Invalid QUD {qud}")


def get_qud_fn(qud: str) -> callable:
    if qud == "q_state":
        return lambda s, v, a: s
    elif qud == "q_valence":
        return lambda s, v, a: v
    elif qud == "q_arousal":
        return lambda s, v, a: a
    else:
        raise ValueError(f"Invalid QUD {qud}")


def normalize_conditional_prob(
    df: pd.DataFrame, conditional_cols: list[str], value_col: str
) -> pd.DataFrame:
    # Normalize a dataframe-based conditional probabilities
    # Group by the conditional columns and sum the value column
    grouped = df.groupby(conditional_cols)[value_col].sum().reset_index()
    # Merge the grouped values back to the original dataframe
    df = df.merge(grouped, on=conditional_cols, suffixes=("", "_total"))
    # Normalize the value column by dividing by the total
    df[value_col] = df[value_col] / df[value_col + "_total"]
    # Drop the total column
    df.drop(columns=[value_col + "_total"], inplace=True)
    return df


def marginalize_over_quds(
    dfs: dict[str, pd.DataFrame],
    quds: dict[str, float],
    common_cols: list[str],
    listener: Literal["L0", "L1"],
) -> pd.DataFrame:
    # Marginalize over the QUDs i.e. sum_{q \in Q} P(s,a,v|c,u,q) * P(q|c)
    # Rename the listeners and sort the rows
    listener = "PL0" if listener == "L0" else "L1"
    for qud in quds.keys():
        dfs[qud].rename(
            columns={f"{listener}(s,a,v|c,u,q)": f"{listener}(s,a,v|c,u,{qud})"},
            inplace=True,
        )
        dfs[qud] = dfs[qud].sort_values(by=common_cols)
    # Merge the dataframes
    common_cols = dfs[list(quds.keys())[0]][common_cols]
    concated = pd.concat(
        [
            common_cols,
            *[dfs[qud][f"{listener}(s,a,v|c,u,{qud})"] for qud in quds.keys()],
        ],
        axis=1,
    )
    # Compute the marginalization
    concated[f"{listener}(s,a,v|c,u)"] = sum(
        [concated[f"{listener}(s,a,v|c,u,{qud})"] * quds[qud] for qud in quds.keys()]
    )
    # If the listener is L1, this was not a true marginalization
    # so we need to normalize the probability
    if listener == "L1":
        concated = normalize_conditional_prob(
            df=concated,
            conditional_cols=["context", "utterance"],
            value_col=f"{listener}(s,a,v|c,u)",
        )
        listener = "PL1"
        concated.rename(
            columns={"L1(s,a,v|c,u)": f"{listener}(s,a,v|c,u)"},
            inplace=True,
        )
    # Sanity check, ensure that for every c,u the sum over s,a,v is 1
    probability_sanity_check(
        concated,
        conditionals=["context", "utterance"],
        value_col=f"{listener}(s,a,v|c,u)",
    )
    return concated


def marginalize_conditioned_variables(
    df: pd.DataFrame, grouped_cols: list[str], value_col: str, new_col: str
) -> pd.DataFrame:
    # Marginalize some set of conditioned variables sum_{x \in X} P(x,y)
    # Group by the conditioned columns and sum the value column
    grouped = df.groupby(grouped_cols)[value_col].sum().reset_index(name=new_col)
    return grouped


def probability_sanity_check(
    df: pd.DataFrame, conditionals: list[str], value_col: str
) -> None:
    # Check that the sum of the value column is 1 for each combination of the conditionals
    # Group by the conditionals and sum the value column
    grouped = (
        df.groupby(conditionals)[value_col]
        .sum()
        .reset_index(name=f"sum_{{{value_col}}}")
    )
    # Check if the sum is 1 for each combination of the conditionals
    assert (np.abs(grouped[f"sum_{{{value_col}}}"] - 1) < 1e-3).all()


def affect_aware_rsa(
    all_priors: pd.DataFrame,
    indicator_function: pd.DataFrame,
    contexts: list[str],
    utterances: list[str],
    states: list[str],
    p_q: dict[str, float] = None,
    lambda_: float = 1,
) -> pd.DataFrame:
    """
    Implement affect-aware RSA from scratch using the equations described in the Kao et al. (2015) paper.
    This paper does not provide extensive details on the equations, and so we consulted another paper:
    Nonliteral understanding of number words - Kao et al. (2014) for additional details (specifically in the Materials and Methods section)
    While this section does not completely elucidate how the equations should be computed, we estimate that they should be the following:
    1. PL0(s,a,v|c,u) \propto P(s,a,v|c) * 1_s_in_[[u]] [the prior] * 1_s_in_[[u]] [the indicator], where the prior is decomposed into P(s) * P(a|s) * P(v|s)
    2. PL0(s,a,v|c,u,q) \propto sum_{x \in X} 1_{q(x) = q(s,a,v)} * PL0(x|c,u), where X = S x A x V
    3. PS1(u|c,s,a,v,q) \propto PL0(s,a,v|c,u,q) [assuming lambda = 1/uniform utterance prior P(u|c)]
    4. L1(s,a,v|c,u,q) = P(s,a,v|c) * PS1(u|c,s,a,v,q) [NOTE: This is not a true probability distribution, it represents the right hand side of Equation 2 in Kao et al. (2015) (page 2)]
    The qud variable q can be marginalized out from either listener with the following equations:
    5-L0. PL0(s,a,v|c,u) = sum_{q \in Q} PL0(s,a,v|c,u,q) * P(q|c) [In this case, the equality is true because PL0 is a true probability distribution]
    5-L1. PL1(s,a,v|c,u) \propto sum_{q \in Q} L1(s,a,v|c,u,q) * P(q|c) [In this case, we must use \propto since L1 is not a true probability distribution]
    [Does it matter? Yes, I tried to normalize PL1 and this leads to results different from Kao's Figure 5. Feel free to try it by modifying the code here.]
    Finally, we can marginalize out the affect variables a and v from the PLi(s,a,v|c,u) to get PLi(s|c,u).
    6. PLi(s|c,u) = sum_{a \in A} sum_{v \in V} PLi(s,a,v|c,u)
    """
    assert lambda_ >= 0
    assert sum(p_q.values()) == 1
    indicator_function = indicator_function[
        ["stateRating", "utterance", "1_s_in_[[u]]"]
    ].drop_duplicates()
    priors_x_indicator = all_priors.merge(indicator_function, on="stateRating")
    assert len(indicator_function) == 25  # 5 states x 5 utterances
    assert len(all_priors) == 180  # 9 contexts x 5 states x 2 arousal x 2 valence
    assert (
        len(priors_x_indicator) == 900
    )  # 9 contexts x 5 states x 5 utterances x 2 arousal x 2 valence
    # Compute PL0(s,a,v|c,u) for every quintuple using equation 1.
    pl0 = (
        priors_x_indicator["1_s_in_[[u]]"]
        * priors_x_indicator["P(s)"]
        * priors_x_indicator["P(valence|s)"]
        * priors_x_indicator["P(arousal|s)"]
    )
    pl0 = pd.concat(
        [
            priors_x_indicator[
                ["context", "utterance", "stateRating", "arousal", "valence"]
            ],
            pl0,
        ],
        axis=1,
    )
    pl0.rename({0: "PL0(s,a,v|c,u)"}, axis=1, inplace=True)
    pl0 = normalize_conditional_prob(
        df=pl0, conditional_cols=["context", "utterance"], value_col="PL0(s,a,v|c,u)"
    )
    probability_sanity_check(
        pl0, conditionals=["context", "utterance"], value_col="PL0(s,a,v|c,u)"
    )
    # Compute PL0(s,a,v|c,u,q) for each qud using equation 2.
    pl0_qud = {}
    valences = priors_x_indicator["valence"].unique()
    arousals = priors_x_indicator["arousal"].unique()
    for qud in ["q_state", "q_valence", "q_arousal"]:
        qud_fn = get_qud_fn(qud)
        pl0_qud[qud] = []
        # Repeat equation 2 for every pair of (c,u)
        # NOTE: This is done without vectorization to keep track of all the moving parts
        for context, utterance in product(contexts, utterances):
            # Fix (c,u) for this loop
            pl0_fixed_cu = pl0[
                (pl0["context"] == context) & (pl0["utterance"] == utterance)
            ]
            assert (
                len(pl0_fixed_cu) == 20
            )  # 900 total (s,a,v,c,u) divided by 9 contexts x 5 states
            # Compute the output of the qud function applied to every (s,a,v) triple (for a fixed (c,u))
            qud_indicator_values = pl0_fixed_cu.apply(
                lambda row: qud_fn(row["stateRating"], row["valence"], row["arousal"]),
                axis=1,
            )
            # For each (s,a,v), sum all the PL0(s,a,v|c,u) where the qud_fn(s,a,v) == qud_indicator_values
            for s, v, a in product(states, valences, arousals):
                sum_sva = (
                    (qud_fn(s, v, a) == qud_indicator_values)
                    * pl0_fixed_cu["PL0(s,a,v|c,u)"]
                ).sum()
                pl0_qud[qud].append([context, utterance, s, v, a, sum_sva])
        pl0_qud[qud] = pd.DataFrame(
            pl0_qud[qud],
            columns=[
                "context",
                "utterance",
                "stateRating",
                "valence",
                "arousal",
                "PL0(s,a,v|c,u,q)",
            ],
        )
        pl0_qud[qud] = normalize_conditional_prob(
            df=pl0_qud[qud],
            conditional_cols=["context", "utterance"],
            value_col="PL0(s,a,v|c,u,q)",
        )
        probability_sanity_check(
            pl0_qud[qud],
            conditionals=["context", "utterance"],
            value_col="PL0(s,a,v|c,u,q)",
        )
    # Compute PS1(u|c,s,a,v,q) for each qud using equation 3.
    s1_qud = {}
    for qud in ["q_state", "q_valence", "q_arousal"]:
        # From equation 3., computation of S1 is a renormalization across different utterances i.e. using c,s,a,v as the conditional rather than c,u
        s1_qud[qud] = pl0_qud[qud].rename(
            columns={"PL0(s,a,v|c,u,q)": "PS1(u|c,s,a,v,q)"}
        )
        s1_qud[qud] = normalize_conditional_prob(
            df=s1_qud[qud],
            conditional_cols=["context", "stateRating", "arousal", "valence"],
            value_col="PS1(u|c,s,a,v,q)",
        )
        probability_sanity_check(
            s1_qud[qud],
            conditionals=["context", "stateRating", "arousal", "valence"],
            value_col="PS1(u|c,s,a,v,q)",
        )
    # Compute L1(s,a,v|c,u,q) for each qud using equation 4. (Note that this is not a true probability distribution)
    # That's why normalization is not done here
    l1_qud = {}
    for qud in ["q_state", "q_valence", "q_arousal"]:
        l1_qud[qud] = pd.merge(
            left=all_priors,
            right=s1_qud[qud],
            on=["context", "stateRating", "arousal", "valence"],
        )
        l1 = (
            l1_qud[qud]["P(s)"]
            * l1_qud[qud]["P(arousal|s)"]
            * l1_qud[qud]["P(valence|s)"]
            * l1_qud[qud]["PS1(u|c,s,a,v,q)"]
        )
        l1 = pd.concat(
            [
                l1_qud[qud][
                    ["context", "utterance", "stateRating", "arousal", "valence"]
                ],
                l1,
            ],
            axis=1,
        )
        l1.rename(
            {0: "L1(s,a,v|c,u,q)"}, axis=1, inplace=True
        )  # Not a true probability distribution, see equation 2 of Kao et al. (2015)
        l1_qud[qud] = l1
        # Uncomment the following if you would like to make l1 for each qud a true probability distribution
        # l1_qud[qud] = normalize_conditional_prob(
        #     df=l1_qud[qud],
        #     conditional_cols=["context", "utterance"],
        #     value_col="L1(s,a,v|c,u,q)",
        # )
        # l1_qud[qud].rename(
        #     columns={"L1(s,a,v|c,u,q)": "PL1(s,a,v|c,u,q)"}, inplace=True
        # )
        # probability_sanity_check(
        #     l1_qud[qud],
        #     conditionals=["context", "utterance"],
        #     value_col="PL1(s,a,v|c,u,q)",
        # )
    # Marginalize L0 and L1 listeners over quds using equations 5-L0 and 5-L1 respectively
    pl0_marg_qud = marginalize_over_quds(
        dfs=pl0_qud,
        quds=p_q,
        common_cols=["context", "utterance", "stateRating", "arousal", "valence"],
        listener="L0",
    )
    pl1_marg_qud = marginalize_over_quds(
        dfs=l1_qud,
        quds=p_q,
        common_cols=["context", "utterance", "stateRating", "arousal", "valence"],
        listener="L1",
    )
    # Marginalize PL0 and PL1 listeners over arousal and valence using equation 6.
    pl0_marg_affect = marginalize_conditioned_variables(
        df=pl0_marg_qud,
        grouped_cols=["context", "utterance", "stateRating"],
        value_col="PL0(s,a,v|c,u)",
        new_col="PL0(s|c,u)",
    )
    pl1_marg_affect = marginalize_conditioned_variables(
        df=pl1_marg_qud,
        grouped_cols=["context", "utterance", "stateRating"],
        value_col="PL1(s,a,v|c,u)",
        new_col="PL1(s|c,u)",
    )
    # Sanity check to ensure that for every c,u, sum over s is 1
    probability_sanity_check(
        pl0_marg_affect, conditionals=["context", "utterance"], value_col="PL0(s|c,u)"
    )
    probability_sanity_check(
        pl1_marg_affect, conditionals=["context", "utterance"], value_col="PL1(s|c,u)"
    )
    return pl0_marg_affect, pl1_marg_affect


def make_fancy_legend(fig: plt) -> None:
    # Generated by ChatGPT
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    # Legend elements for Weather type (color)
    weather_legend_elements = [
        plt.Line2D(
            [0], [0], color="red", marker="o", linestyle="None", label="Positive"
        ),
        plt.Line2D(
            [0],
            [0],
            color="mediumturquoise",
            marker="o",
            linestyle="None",
            label="Neutral",
        ),
        plt.Line2D(
            [0], [0], color="steelblue", marker="o", linestyle="None", label="Negative"
        ),
    ]

    # Legend elements for Interpretation (linestyle)
    interpretation_legend_elements = [
        plt.Line2D([0], [0], color="black", linestyle="-", label="Human-induced"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="Affect-aware RSA"),
    ]

    # Draw the legends
    legend1 = fig.legend(
        handles=weather_legend_elements,
        title="Weather type",
        loc="center left",
        bbox_to_anchor=(0.85, 0.6),
    )
    legend2 = fig.legend(
        handles=interpretation_legend_elements,
        title="$P(s|c,u)$ distribution type",
        loc="center left",
        bbox_to_anchor=(0.85, 0.55),
    )

    # Make sure the legend titles are bold like in the example
    plt.setp(legend1.get_title(), weight="bold")
    plt.setp(legend2.get_title(), weight="bold")

    return legend1, legend2


def plot_l1_affect_aware(
    df_l1: pd.DataFrame, df_human: pd.DataFrame, utterances: list[str], output_dir: str
) -> None:
    # Reproduce Figure 5 from the Kao et al. (2015) paper
    fig, axs = plt.subplots(9, 5, figsize=(20, 25))

    def color_map(context: str) -> str:
        # Map the context to a color
        if "amazing" in context:
            return "red"
        elif "ok" in context:
            return "mediumaquamarine"
        elif "terrible" in context:
            return "steelblue"
        else:
            raise ValueError(f"Invalid context {context}")

    # Hardcoded so the order aligns with Figure 5
    contexts = [
        "amazing_3",
        "amazing_2",
        "amazing_1",
        "ok_6",
        "ok_5",
        "ok_4",
        "terrible_8",
        "terrible_7",
        "terrible_9",
    ]  # terrible_7 and terrible_9 look basically the same, this is also true in the Kao figure
    contexts_map = {context_name: f"W{i+1}" for i, context_name in enumerate(contexts)}

    state_mapping = {
        1: "terrible",
        2: "bad",
        3: "neutral",
        4: "good",
        5: "amazing",
    }

    for i, context in enumerate(contexts):
        for j, utterance in enumerate(utterances):
            # Model plot
            l1_filtered = df_l1[
                (df_l1["context"] == context) & (df_l1["utterance"] == utterance)
            ]
            axs[i, j].plot(
                l1_filtered["stateRating"],
                l1_filtered["PL1(s|c,u)"],
                marker="o",
                c=color_map(context),
                linestyle="--",
            )
            # Human-induced probabilities plot
            plus_one_counts = (
                df_human[
                    (df_human["context"] == context)
                    & (df_human["utterance"] == utterance)
                ]
                .groupby(by="stateRating")
                .size()
                .reindex(range(1, 6), fill_value=0)
                + 1
            )
            plus_one_counts_denominator = plus_one_counts.sum()
            true_props = plus_one_counts / plus_one_counts_denominator
            axs[i, j].plot(
                np.array(sorted(true_props.index)),
                true_props,
                marker="o",
                c=color_map(context),
            )
            axs[i, j].set_ylim(0, 1)
            axs[i, j].grid(True)
            if i == 0:
                axs[i, j].set_title(f'Utterance:\n"The weather is {utterance}".')
            if i != len(contexts) - 1:
                axs[i, j].set_xticks(list(range(1, 6)), labels=[""] * 5)
            if i == len(contexts) - 1:
                axs[i, j].set_xticks(
                    list(range(1, 6)),
                    labels=[state_mapping[i] for i in range(1, 6)],
                    rotation=45,
                )
            if j == len(utterances) - 1:
                axs[i, j].yaxis.set_label_position("right")
                axs[i, j].set_ylabel(
                    f"Weather context: {contexts_map[context]}",
                    rotation=270,
                    labelpad=10,
                )

    # Shared axis labels
    fig.supxlabel(
        "Weather state",
        fontsize=14,
    )
    fig.supylabel(
        "Probability\n",
        fontsize=14,
    )

    # Move legend outside the plot
    legend1, legend2 = make_fancy_legend(fig)

    # Tighter layout to leave space for external legend
    plt.savefig(
        REPO_PATH / output_dir / "affect_aware_rsa_l1.png",
        dpi=300,
        bbox_extra_artists=[legend1, legend2],
    )


def run_affect_aware_rsa_exps(config: dict) -> None:
    # Seed everything
    if "seed" in config:
        seed_everything(seed=config["seed"])
    # Load data
    df_prior = pd.read_csv(REPO_PATH / config["prior_path"])
    interpretation_w_irony = pd.read_csv(
        REPO_PATH / config["interpretation_w_irony_path"]
    )
    # Preprocess data
    df_prior = preprocess_data(
        df_prior=df_prior, interpretation_w_irony=interpretation_w_irony
    )
    # Fix idx mapping for the categories
    context_to_idx, utterances_to_idx, meanings_to_idx, _ = create_category_mappings()
    # Create indicator function for the semantic understanding indicator
    semantic_indicator_function = create_semantic_indicator_function(
        context_to_idx=context_to_idx,
        utterances_to_idx=utterances_to_idx,
        meanings_to_idx=meanings_to_idx,
    )
    print(semantic_indicator_function)
    # Create affect priors and merge them with state priors
    all_priors = compute_priors(
        df_prior=df_prior,
        context_to_idx=context_to_idx,
        meanings_to_idx=meanings_to_idx,
    )
    # Compute affect-aware RSA
    l0_affect_aware, l1_affect_aware = affect_aware_rsa(
        all_priors=all_priors,
        indicator_function=semantic_indicator_function,
        contexts=list(context_to_idx.keys()),
        utterances=list(utterances_to_idx.keys()),
        states=list(meanings_to_idx.keys()),
        p_q=config["p_quds"],
    )
    l_affect_aware = pd.merge(
        left=l0_affect_aware,
        right=l1_affect_aware,
        on=["context", "utterance", "stateRating"],
        how="inner",
    )
    l_affect_aware.to_csv(
        REPO_PATH / config["output_dir"] / "affect_aware_rsa.csv",
        index=False,
    )
    # Plot the l1 distributions to compare with Figure 5 from Kao et al. (2015)
    plot_l1_affect_aware(
        df_l1=l1_affect_aware,
        df_human=interpretation_w_irony,
        utterances=list(utterances_to_idx.keys()),
        output_dir=config["output_dir"],
    )


# Run all exps, compute metrics and design plots


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

def plot_bar_plots_by_context(
    df_human: pd.DataFrame,
    df_affect_aware: pd.DataFrame,
    df_rsa_two: pd.DataFrame,
    utterances: list[str],
):
    # Column renaming
    df_affect_aware = df_affect_aware.rename(
        columns={"PL0(s|c,u)": "PL0_affect", "PL1(s|c,u)": "PL1_affect"}
    )
    df_rsa_two = df_rsa_two.rename(
        columns={"PL0(s|c,u)": "PL0_rsa2", "PL1(s|c,u)": "PL1_rsa2"}
    )
    df_human = df_human.rename(columns={"P(s|c,u)": "PHuman"})

    # Merge all on shared keys
    merged = df_human.merge(df_affect_aware, on=["context", "utterance", "stateRating"])
    merged = merged.merge(df_rsa_two, on=["context", "utterance", "stateRating"])

    col_to_label = {
        "PHuman": "$P(m|c,u)$ - Human",
        "PL0_affect": "$P_{{L_0}}(m|c,u)$ - Affect-Aware RSA",
        "PL1_affect": "$P_{{L_1}}(m|c,u)$ - Affect-Aware RSA",
        "PL0_rsa2": "$P_{{L_0}}(m|c,u)$ - RSA$^2$",
        "PL1_rsa2": "$P_{{L_1}}(m|c,u)$ - RSA$^2$",
    }

    col_to_color = {
        "PHuman": "#1f77b4",
        "PL0_affect": "#9467bd",
        "PL1_affect": "#ff7f0e",
        "PL0_rsa2": "#2ca02c",
        "PL1_rsa2": "#d62728",
    }

    # Hardcode contexts so order is same as in Figure 5
    contexts = [
        "amazing_3",
        "amazing_2",
        "amazing_1",
        "ok_6",
        "ok_5",
        "ok_4",
        "terrible_8",
        "terrible_7",
        "terrible_9",
    ]

    # Context name mapping
    contexts_map = {context: f"{i+1}" for i, context in enumerate(contexts)}

    state_mapping = {
        1: "terrible",
        2: "bad",
        3: "neutral",
        4: "good",
        5: "amazing",
    }

    # Group by context
    figs = {}
    for context in contexts:
        df_context = merged[merged["context"] == context]

        non_empty_utterances = [
            utt
            for utt in utterances
            if len(df_context[df_context["utterance"] == utt]) > 0
        ]

        n_utts = len(non_empty_utterances)

        fig, axs = plt.subplots(n_utts, 1, figsize=(12, 5 * n_utts), sharex=True)

        if n_utts == 1:
            axs = [axs]  # Make iterable

        for ax, utt in zip(axs, non_empty_utterances):
            df_u = df_context[df_context["utterance"] == utt].sort_values("stateRating")
            x = df_u["stateRating"].values
            width = 0.13
            offsets = np.linspace(-width * 2, width * 2, len(col_to_label))

            for offset, col in zip(offsets, col_to_label.keys()):
                ax.bar(
                    x + offset,
                    df_u[col],
                    width=width,
                    label=col_to_label[col],
                    color=col_to_color[col],
                )

            ax.set_title(
                f'Meaning Probabilities for "The weather is {utt}."', fontsize=18
            )
            ax.set_xticks(list(state_mapping.keys()))
            ax.set_ylim(0, 1.05)
            # add grid lines
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            if utt == non_empty_utterances[-1]:
                ax.set_xticklabels(
                    [state_mapping[i] for i in state_mapping.keys()],
                    rotation=30,
                    fontsize=16,
                )

            # add legend to each plot
            ax.legend(
                loc="best",
                # ncol=len(col_to_label),
                fontsize=14,
            )
            # ax.set_ylabel("Probability", fontsize=16)
            # ax.set_xlabel("State", fontsize=16)

        legend = None
        # handles, labels = axs[0].get_legend_handles_labels()
        # legend = fig.legend(
        #     handles,
        #     labels,
        #     loc="lower center",
        #     ncol=len(col_to_label),
        #     bbox_to_anchor=(0.5, -0.01),
        #     fontsize=20,
        # )

        fig.supxlabel("State", fontsize=16)
        fig.supylabel("Probability", fontsize=16)
        fig.suptitle(
            f"Meaning Probabilities for Weather Context {contexts_map[context]}\n",
            fontsize=18,
        )

        fig.tight_layout()

        figs[context] = (fig, legend)
    return figs


# @log_run
def run_exps(config: dict) -> None:
    ### Run (RSA)^2 experiments ###
    run_rsa_two_exps(
        config=config,
    )
    ### Run affect-aware RSA experiments ###
    run_affect_aware_rsa_exps(
        config=config,
    )
    ### Compute metrics ###
    for split in ["train", "test"]:
        df_human = pd.read_csv(
            REPO_PATH / config["output_dir"] / f"human_dist_{split}.csv"
        )
        affect_aware_rsa = pd.read_csv(
            REPO_PATH
            / config["output_dir"]
            / "affect_aware_rsa.csv"  # Does not change from train to test because everything is computed using prior data which is from another file
        )
        rsa_two = pd.read_csv(
            REPO_PATH
            / config["output_dir"]
            / f"rsa_two_{split}.csv"  # Changes based on the P(r|c,u) given by each participant
        )
        models = ["PL0-AARSA", "PL1-AARSA", "PL0-RSA^2", "PL1-RSA^2"]
        metrics = compute_metrics(
            df_human=df_human,
            human_column="P(s|c,u)",
            model_predictions=dict(
                zip(models, [affect_aware_rsa, affect_aware_rsa, rsa_two, rsa_two])
            ),
            model_names=dict(
                zip(models, ["PL0(s|c,u)", "PL1(s|c,u)", "PL0(s|c,u)", "PL1(s|c,u)"])
            ),
            merge_on=["context", "utterance", "stateRating"],
        )
        metrics.to_csv(
            REPO_PATH / config["output_dir"] / f"metrics_{split}.csv", index=True
        )
    ### Generate plots ###
    # Only do it for the test set
    plots = plot_bar_plots_by_context(
        df_human=pd.read_csv(REPO_PATH / config["output_dir"] / "human_dist_test.csv"),
        df_affect_aware=pd.read_csv(
            REPO_PATH / config["output_dir"] / "affect_aware_rsa.csv"
        ),
        df_rsa_two=pd.read_csv(REPO_PATH / config["output_dir"] / "rsa_two_test.csv"),
        utterances=[
            "terrible",
            "bad",
            "ok",
            "good",
            "amazing",
        ],
    )
    for context, (fig, legend) in plots.items():
        fig.savefig(
            REPO_PATH / config["output_dir"] / f"bar_plot_{context}.png",
            dpi=300,
            bbox_inches="tight",
            # bbox_extra_artists=[legend],
        )


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="weather_utterances_exp",
)
def main(config: dict) -> None:
    run_exps(config=config)


if __name__ == "__main__":
    main()
