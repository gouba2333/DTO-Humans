"""
Code adapted for timm >= 1.0.0 from the original mivolo implementation.

Original mivolo license:
# Modifications and additions for mivolo by / Copyright 2023, Irina Tolstykh, Maxim Kuprashevich
"""

import os
from typing import Any, Dict, List, Optional

import timm
import torch
from torch import nn

# register new models from the mivolo project
# This import is essential as it registers custom mivolo models with the timm registry.
from mivolo.model.mivolo_model import * # noqa: F403, F401

# For timm >= 1.0.0, these helpers are generally in _helpers but it's
# always best to rely on the main public API (timm.create_model) when possible.
# We still need load_state_dict for our custom checkpoint loading logic.
from timm.models._helpers import load_state_dict


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    use_ema: bool = True,
    strict: bool = True,
    filter_keys: Optional[List[str]] = None,
    state_dict_map: Optional[Dict[str, str]] = None,
):
    """
    Loads a checkpoint with custom filtering and key mapping.
    This function is preserved to handle the special `filter_keys` and `state_dict_map` logic.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    # This handles loading numpy checkpoints for specific models, a pattern from older timm.
    # It's safe to keep for compatibility with the original project's models.
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        if hasattr(model, "load_pretrained"):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model does not support loading a numpy checkpoint.")
        return

    state_dict = load_state_dict(checkpoint_path, use_ema)

    # Custom logic to filter out keys from the state_dict (e.g., remove classifier weights)
    if filter_keys:
        for sd_key in list(state_dict.keys()):
            if any(filter_key in sd_key for filter_key in filter_keys):
                if sd_key in state_dict:
                    del state_dict[sd_key]

    # Custom logic to rename keys in the state_dict
    if state_dict_map:
        new_state_dict = {}
        for state_k, state_v in state_dict.items():
            new_key = state_k
            for target_v, target_k in state_dict_map.items():
                if target_v in new_key:
                    new_key = new_key.replace(target_v, target_k)
            new_state_dict[new_key] = state_v
        state_dict = new_state_dict

    # Determine strictness: if we filtered keys, we can't be fully strict.
    is_strict = strict and filter_keys is None
    model.load_state_dict(state_dict, strict=is_strict)


def create_model(
    model_name: str,
    pretrained: bool = False,
    checkpoint_path: str = "",
    filter_keys: Optional[List[str]] = None,
    state_dict_map: Optional[Dict[str, str]] = None,
    **kwargs: Any,
):
    """
    Creates a model, with support for custom local checkpoints with key filtering/mapping.

    Args:
        model_name (str): Name of model to instantiate.
        pretrained (bool): Load pretrained weights from timm's default source.
        checkpoint_path (str): Path to a local checkpoint to load.
        filter_keys (list, optional): A list of substrings. Keys in the checkpoint's
            state_dict containing any of these substrings will be removed.
        state_dict_map (dict, optional): A dictionary to map substrings in keys.
            E.g., `{'old.prefix.': 'new.prefix.'}`.
        **kwargs: Other args passed to `timm.create_model`.
    """
    # The modern `timm.create_model` handles name parsing, Hugging Face Hub models
    # (e.g., 'hf-hub:org/model'), pretrained configs, and more automatically.
    # We remove any None kwargs to avoid overriding model defaults.
    model_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # If a local checkpoint is provided, we might not want to download pretrained weights.
    # The common pattern is to create the model un-initialized and then load the local checkpoint.
    # However, to match the original code's behavior, we allow loading both, where the
    # local checkpoint will overwrite the pretrained weights.
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        **model_kwargs,
    )

    # If a local checkpoint path is provided, use our custom loader.
    # This is done *after* model creation.
    if checkpoint_path:
        load_checkpoint(
            model,
            checkpoint_path,
            filter_keys=filter_keys,
            state_dict_map=state_dict_map
        )

    return model