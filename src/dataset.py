import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from loguru import logger
from src.pddl import parse_pddl
from src.transforms import get_transforms


pio.kaleido.scope.mathjax = None


def prepare_dataset(
    verbs_from_args: List[str],
    verbs_path: str,
    train_path: str,
    val_path: str,
    pddl_domain_path: str,
    pddl_problem_path: str,
    save_attributes_path: str,
    make_plots: bool = False,
    augment: bool = True,
    factor: float = 1.0,
) -> None:
    """
    Prepares the dataset by filtering it to keep only the verbs we want and
    making some plots on the distribution.

    It creates a `filtered_train.pkl` and `filtered_val.pkl` files in the
    `data` folder to be later used by the model's `DataLoader`.

    Parameters
    ----------
    `verbs_from_args` : `List[str]`
        The list of verbs to keep, passed by CLI arguments.
    `verbs_path` : `str`
        The path to the verbs file.
    `train_path` : `str`
        The path to the training annotations
    `val_path` : `str`
        The path to the validation annotations
    `pddl_domain_path` : `str`
        The path to the PDDL domain file
    `pddl_problem_path` : `str`
        The path to the PDDL problem file
    `save_attributes_path` : `str`
        The path to the file where to save the attributes
    `make_plots` : `bool`, optional
        Whether to make plots or not, by default `False`
    `augment` : `bool`, optional
        Whether to augment the dataset or not, by default `True`
    `factor` : `float`, optional
        The factor to use for augmentation, by default `1.0`.
    """
    logger.info(f"Preparing dataset with verbs: {verbs_from_args}")
    ids, map_ids_verbs, verbs_df = load_verbs(verbs_from_args=verbs_from_args, path=verbs_path)
    train_df = load_dataset(path=train_path)
    val_df = load_dataset(path=val_path)

    if make_plots:
        logger.info("Making plots...")
        # Make plots for all splits
        _make_plots(
            df=train_df,
            split="train",
            verbs_df=verbs_df,
            chosen_verbs_ids=ids,
        )
        _make_plots(
            df=val_df,
            split="validation",
            verbs_df=verbs_df,
            chosen_verbs_ids=ids,
        )
        plot_split_distribution(train_df=train_df, val_df=val_df)

    # Filter the dataset to keep only the verbs we want
    filtered_train_df = train_df[train_df.verb_class.isin(ids)]
    filtered_val_df = val_df[val_df.verb_class.isin(ids)]

    # Load PDDL domain
    actions, attributes = parse_pddl(domain_path=pddl_domain_path, problem_path=pddl_problem_path)
    assert set(map_ids_verbs.values()).issubset(
        set([a.name for a in actions])
    ), f"Some actions are not in the list of verbs: {set(map_ids_verbs.values()) - set([a.name for a in actions])}"
    attributes_df = pd.DataFrame(attributes, columns=["attribute"])
    attributes_df.to_csv(save_attributes_path, index=False)

    # Extend the datasets with vectorized pre- and post-conditions for each action
    vectors = {
        action.name: {
            "precs": [str(p) for p in action.preconditions],
            "posts": [str(p) for p in action.postconditions],
            "precs_vec": action.vectorize(attributes)[0],
            "posts_vec": action.vectorize(attributes)[1],
        }
        for action in actions
    }

    logger.warning(f"Vectors: {vectors}")
    filtered_train_df = extend_data(
        df=filtered_train_df,
        map_ids_verbs=map_ids_verbs,
        vectors=vectors,
    )
    filtered_val_df = extend_data(
        df=filtered_val_df,
        map_ids_verbs=map_ids_verbs,
        vectors=vectors,
    )

    if augment:
        logger.warning("Augmenting dataset to balance selected classes...")
        transforms = get_transforms(p=1.0)
        logger.debug(f"Before:\n{filtered_train_df.verb_class.value_counts()=}")
        logger.debug(f"Before:\n{filtered_val_df.verb_class.value_counts()=}")
        filtered_train_df = augment_data(
            df=filtered_train_df,
            transforms=transforms,
            factor=3.0,
        )
        filtered_val_df = augment_data(
            df=filtered_val_df,
            transforms=transforms,
            factor=3.0,
        )
        logger.success("Done augmenting dataset.")
        logger.debug(f"After:\n{filtered_train_df.verb_class.value_counts()=}")
        logger.debug(f"After:\n{filtered_val_df.verb_class.value_counts()=}")

    # Save the filtered datasets
    filtered_train_df.to_pickle(os.path.join(os.path.dirname(train_path), "filtered_train.pkl"))
    filtered_val_df.to_pickle(os.path.join(os.path.dirname(val_path), "filtered_val.pkl"))
    logger.success("Dataset prepared!")


def load_verbs(
    verbs_from_args: List[str],
    path: str,
) -> Tuple[List[int], Dict[str, int], pd.DataFrame]:
    """
    Checks that the selected verbs are in the list of actual verbs and returns both the IDs and
    the verbs, as well as all the verbs DataFrame.

    Parameters
    ----------
    `verbs_from_args` : `List[str]`
        The list of verbs to keep, passed by CLI arguments.

    `path` : `str`
        The path to the verbs file.

    Returns
    -------
    `Tuple[List[int], Dict[str, int], pd.DataFrame]`
        The IDs of the chosen verbs, the mapping between verbs and verb IDs and all
        the verbs as a `pd.DataFrame` object.
    """
    # Load the verbs from the file
    verbs_df = pd.read_csv(path, header=0, index_col=0)

    # Sort them alphabetically
    verbs = sorted(verbs_df.key.unique())

    assert set(verbs_from_args).issubset(
        verbs
    ), f"Some verb classes are not in the list of verbs: {verbs_from_args - set(verbs)}"

    # Get all the IDs corresponding to the verb classes we want to keep
    verbs_from_args_ids = verbs_df[verbs_df.key.isin(verbs_from_args)].index.to_list()

    for i in verbs_from_args_ids:
        logger.debug(f"{i}:{verbs_df.loc[i].key}")

    map_ids = {i: verbs_df.loc[i].key for i in verbs_from_args_ids}

    return verbs_from_args_ids, map_ids, verbs_df


def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the dataset from the given path and adds a column with the duration of each video.

    Parameters
    ----------
    `path` : `str`
        The path to the dataset.

    Returns
    -------
    `pd.DataFrame`
        The dataset with the duration of each video.
    """
    df = pd.read_pickle(path)
    df["start_ts_td"] = pd.to_timedelta(df["start_timestamp"])
    df["stop_ts_td"] = pd.to_timedelta(df["stop_timestamp"])
    df["duration_in_s"] = (df["stop_ts_td"] - df["start_ts_td"]).dt.total_seconds()
    del df["start_ts_td"]
    del df["stop_ts_td"]

    return df


def _make_plots(
    df: pd.DataFrame,
    split: str,
    verbs_df: pd.DataFrame,
    chosen_verbs_ids: List[int],
    top_n: int = 30,
    width: int = 1200,
    height: int = 600,
) -> None:
    """
    Makes plots for the given split: the distribution of the top N most
    present verb classes and the interesting verb classes.

    The plots will be saved in the `res` folder as PDF files to be easily
    embeddable in LaTeX documents.

    Parameters
    ----------
    `df` : `pd.DataFrame`
        The dataset
    `split` : `str`
        The split to make the plots for
    `verbs_df` : `pd.DataFrame`
        The DataFrame containing all the verbs
    `chosen_verbs` : `List[str]`
        The list of verbs to keep
    `top_n` : `int`, optional
        The top `n` verbs , by default `30`
    `width` : `int`, optional
        Width of the output PDF files, by default `1200`
    `height` : `int`, optional
        Width of the output PDF files, by default `600`
    """
    os.makedirs("res/dataset", exist_ok=True)
    agg_df = df.groupby("verb_class").agg({"duration_in_s": "sum"}).sort_values("duration_in_s", ascending=False)
    agg_df["verb_class_name"] = agg_df.index.map(lambda x: verbs_df.loc[x].key)
    # Plot the top N most present verb classes
    fig1 = px.bar(
        agg_df[:top_n],
        y="duration_in_s",
        x="verb_class_name",
        labels={
            "verb_class_name": "Verb class",
            "duration_in_s": "Total aggregated duration (in s)",
        },
        title=f"Top {top_n} most present verb classes in the {split} dataset",
        text="duration_in_s",
        text_auto=True,
        color_discrete_sequence=["green"] if split != "train" else None,
        height=height,
        width=width,
    )
    fig1.update_layout(
        xaxis=dict(tickfont=dict(family="Courrier New, monospace", size=15)),
        yaxis=dict(tickformat=".2s"),
        font=dict(family="CMU Serif", size=19),
    )
    fig1.write_image(f"res/dataset/top_{top_n}_verb_classes_{split}.pdf")

    # Verbs distribution for the chosen verbs
    fig2 = px.bar(
        agg_df[agg_df.index.isin(chosen_verbs_ids)].sort_values("duration_in_s", ascending=False),
        x="verb_class_name",
        y="duration_in_s",
        labels={
            "verb_class_name": "Verb class",
            "duration_in_s": "Total aggregated duration (in s)",
        },
        title=f"Chosen verb classes in the {split} dataset",
        text="duration_in_s",
        text_auto=True,
        color_discrete_sequence=["green"] if split != "train" else None,
        height=height,
        width=width,
    )
    fig2.update_layout(
        xaxis=dict(tickfont=dict(family="Courrier New, monospace", size=15)),
        yaxis=dict(tickformat=".2s"),
        font=dict(family="CMU Serif", size=19),
    )
    fig2.write_image(f"res/dataset/interesting_verb_classes_{split}.pdf")


def plot_split_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """
    Plots the distribution of the train and validation splits.

    Parameters
    ----------
    `train_df` : `pd.DataFrame`
        The training dataset
    `val_df` : `pd.DataFrame`
        The validation dataset
    """
    fig = px.bar(
        x=["Train", "Validation"],
        y=[len(train_df), len(val_df)],
        text=[len(train_df), len(val_df)],
        text_auto=True,
        color_discrete_sequence=["salmon"],
        labels={"x": "Split", "y": "Number of data points"},
        width=500,
    )

    fig.update_layout(
        yaxis={
            "tickformat": ".2s",  # 10k instead of 10,000
        },
        font={"family": "CMU Serif"},
    )
    fig.write_image("res/dataset/split_distribution.pdf")


def extend_data(
    df: pd.DataFrame,
    map_ids_verbs: Dict[int, str],
    vectors: Dict[str, Any],
) -> pd.DataFrame:
    """
    Extends dataset with vectorized pre- and post-conditions for each action.

    Parameters
    ----------
    `df` : `pd.DataFrame`
        The dataset
    `map_ids_verbs` : `Dict[int, str]`
        The mapping between verb IDs and verb classes
    `vectors` : `Dict[str, Any]`
        The vectors for each action

    Returns
    -------
    `pd.DataFrame`
        The extended dataset
    """
    copy_df = df.copy()
    verb_classes = copy_df["verb_class"].map(map_ids_verbs)
    copy_df.loc[:, "precs"] = verb_classes.map(lambda vc: vectors[vc]["precs"])
    copy_df.loc[:, "posts"] = verb_classes.map(lambda vc: vectors[vc]["posts"])
    copy_df.loc[:, "precs_vec"] = verb_classes.map(lambda vc: vectors[vc]["precs_vec"])
    copy_df.loc[:, "posts_vec"] = verb_classes.map(lambda vc: vectors[vc]["posts_vec"])

    return copy_df


def augment_data(
    df: pd.DataFrame,
    transforms: List[Callable],
    factor: float = 1.0,
) -> pd.DataFrame:
    """
    Augment dataset with transformation on audio segments

    Parameters
    ----------
    `df` : `pd.DataFrame`
        Dataset dataframe
    `transforms` : `List[Callable]`
        List of possible transforms
    `factor` : `float`. Defaults to 1.0.
        Multiplicative factor for dataset size. The final
        size will be the number of samples from the most
        populated class, multiplied by `factor`.

    Returns
    -------
    `pd.DataFrame`
        The resulting augmented dataframe.
    """
    verb_counts = pd.DataFrame(df["verb_class"].value_counts())
    verb_counts_dict = verb_counts.to_dict()["count"]

    max_verb_class = max(verb_counts_dict, key=verb_counts_dict.get)
    max_value = factor * verb_counts_dict[max_verb_class]

    # Compute how many augmented samples we need to balance classes
    verb_counts["to_augment"] = max_value - verb_counts["count"]
    verb_counts["t_per_sample"] = verb_counts["to_augment"] / verb_counts["count"]
    t_by_class = verb_counts.to_dict(orient="index")

    df["transformation"] = "none"

    augmented_rows = []
    for _, row in df.iterrows():
        rows = [row.to_dict()]
        c = row["verb_class"]
        t_per_row = t_by_class[c]["t_per_sample"]

        if 0 < t_per_row <= 1:
            # Transform the current row with a certain probability
            augment = np.random.binomial(n=1, p=t_per_row)

            if augment:
                transformation = np.random.choice(list(transforms.keys()))
                aug_row = row.copy()
                aug_row["transformation"] = transformation

                rows.append(aug_row)

        elif t_per_row > 1:
            # If we need strictly more than 1 transform per row,
            # we then select a random transformation for each of the
            # augmentations
            for _ in range(round(t_per_row)):
                transformation = np.random.choice(list(transforms.keys()))
                aug_row = row.copy()
                aug_row["transformation"] = transformation

                rows.append(aug_row)

        augmented_rows.extend(rows)

    augmented_df = pd.DataFrame(augmented_rows)
    logger.debug(f"Augmented dataset class counts:\n{augmented_df.verb_class.value_counts()}")

    return augmented_df
