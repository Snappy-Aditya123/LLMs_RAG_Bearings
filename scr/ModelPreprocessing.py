import torch as t 
import pandas as pd 
import numpy as np 
import math 
import random



# First we will work on vibration channel as it is most important data 

# model arch follows this rules 
# -- model for vibration channel is 1D CNN
# -- model for other charge channels are two independent 1D CNNs too 
# -- A simple MLP is enough for other low frequnevcy channels such as temp and torue 
# finally a cross attention block will be applied before classification still workin on this part 

# preprocessing the Vibration channel 
# Vibration channel is mostly standard 64Hz frequency but some data poins are either lessor more based on the eda 
# analysis so padding and cutting unnecessary data will be performed first this preporcessing can be applied to other high feequency channels too


def pad_high_channels(
    signal,
    target_length,
    crop_mode="start",
    padding_type="zero",
    pad_side="right",
):
    signal = np.asarray(signal)

    signal_length = len(signal)
    crop_start = 0

    # Crop: keep one continuous interval.
    if signal_length > target_length:
        excess = signal_length - target_length

        if crop_mode == "center":
            crop_start = excess // 2
        elif crop_mode == "random":
            crop_start = random.randint(0, excess)

        processed = signal[crop_start:crop_start + target_length].copy()
        mask = np.ones(target_length, dtype=bool)

    # Pad: mark all added samples as invalid.
    elif signal_length < target_length:
        missing = target_length - signal_length

        if pad_side == "right":
            left, right = 0, missing
        elif pad_side == "left":
            left, right = missing, 0
        else:
            left = missing // 2
            right = missing - left

        if padding_type == "zero":
            processed = np.pad(
                signal, (left, right),
                mode="constant", constant_values=0,
            )
        elif padding_type == "reflect":
            if signal_length < 2:
                raise ValueError("Reflection requires at least two samples.")
            processed = np.pad(signal, (left, right), mode="reflect")
        else:
            processed = np.pad(signal, (left, right), mode="edge")

        mask = np.pad(
            np.ones(signal_length, dtype=bool),
            (left, right),
            mode="constant",
            constant_values=False,
        )

    else:
        processed = signal.copy()
        mask = np.ones(target_length, dtype=bool)

    return processed, mask



def normalize_signals(signal: np.array, type):
    eps = 1e-8
    if type == "min_max":
        normalixed_signal = signal - min(signal) / (max(signal) - min(signal))

    if type == "Z_score":
        normalixed_signal = (signal - signal.mean()) / signal.std()
    return normalixed_signal


def divide_dataset(
    metadata: pd.DataFrame,
    signal_df: pd.DataFrame,
    sigal_processed: np.ndarray,
    ratio: tuple[float, float, float],
    channel: str,
    *,
    random_state: int = 42,
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Return each split as (metadata, processed signals)."""
    ratio_test, ratio_val, ratio_train = ratio

    channel_names = (
    metadata["signal_id"]
    .astype("string")
    .str.extract(r"/([^/]+)$", expand=False)
)
    selected = metadata.loc[channel_names.str.fullmatch(channel, na=False)]

    join_keys = ["signal_id"]
    if "dataset" in metadata.columns and "dataset" in signal_df.columns:
        join_keys.insert(0, "dataset")

    signal_lookup = signal_df[join_keys].assign(
        _signal_position=np.arange(len(signal_df))
    )
    matched = selected.merge(
        signal_lookup, on=join_keys, how="left", validate="one_to_one"
    )

    recording_keys = ["recording_id"]
    if "dataset" in matched.columns:
        recording_keys.insert(0, "dataset")


    recording_codes = matched.groupby(recording_keys, sort=False).ngroup().to_numpy()
    recordings = np.random.default_rng(random_state).permutation(
        len(np.unique(recording_codes))
    )

    train_end = round(len(recordings) * ratio_train)
    val_end = train_end + round(len(recordings) * ratio_val)
    if ratio_test == 0:
        val_end = len(recordings)

    recording_splits = {
        "train": recordings[:train_end],
        "val": recordings[train_end:val_end],
        "test": recordings[val_end:],
    }

    splits = {}
    for name, recording_ids in recording_splits.items():
        rows = matched.loc[np.isin(recording_codes, recording_ids)]
        positions = rows["_signal_position"].to_numpy(dtype=np.intp)
        split_metadata = rows.drop(columns="_signal_position").reset_index(drop=True)
        splits[name] = (split_metadata, sigal_processed[positions])

    return splits