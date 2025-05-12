# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference code: https://github.com/iboing/CorDA/blob/main/cordalib/decomposition.py
# Reference paper: https://arxiv.org/abs/2406.05223

import os
from collections.abc import Iterable
from typing import Any, Callable, Optional

import torch
import numpy as np
import torch.nn as nn
from attr import dataclass
from tqdm import tqdm


from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel
from peft.utils.other import get_pattern_key


@dataclass
class CordaEigens:
    S_WC: torch.Tensor
    U_WC: torch.Tensor
    V_WC: torch.Tensor


def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
    """
    Iterate over CorDA target name and modules of a model. A module is a target if its name is in
    `config.target_modules` and is `nn.Linear`.
    """
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module


def get_model_device(model: nn.Module) -> str:
    if hasattr(model, "module"):  # Handle DeepSpeed/DataParallel
        model = model.module
    return next(iter(model.parameters())).device.type


@torch.no_grad()
def preprocess_corda(
    model: nn.Module,
    lora_config: LoraConfig,
    run_model: Optional[Callable[[], None]] = None,
    hooked_model: Optional[nn.Module] = None,
):
    """
    Build necessary CorDA fields for a model.

    For each `M * N` linear layer, a `M * M` covariance matrix will be built temporarily during the preprocessing
    process, consuming roughly another `2 * MODEL_SIZE` memory for typical LLMs if model weight is FP16 and covariance
    is FP32. If that's too much, consider specifying `use_float16_for_covariance` in `lora_config.corda_config`.

    Args:
        model (`nn.Module`):
            Model to preprocess.
        lora_config (`LoraConfig`):
            Lora configuration of the model. `lora_config.corda_config` should be set.
        run_model (`Optional[Callable[[], None]]`):
            Callback to run the model when building covariance. Typically you should run model inference on your sample
            dataset in this callback. Experiments have shown that when token count per sample is 2048, hidden dimension
            is 4096, collecting 256 distinct samples is enough. If you collect too few or too repetitive samples, the
            covariance matrix may be low-ranked and unstabilize preprocessing. You can estimate sample count as
            `HIDDEN_DIM / TOKEN_PER_SAMPLE * 128`. `run_model` can be `None` only if covariance file in
            `lora_config.corda_config` is already created.
        hooked_model (`Optional[nn.Module]`):
            Model to hook when building covariance. If none, original model will be hooked. This is only useful when
            you want to hook a different model than the one you are training, typically you should leave this `None`.

    Upon completion, the following fields are set for each target module:
        eigens.S_WC (`torch.Tensor`):
            Singular values of the weight matrix.
        eigens.U_WC (`torch.Tensor`):
            Left singular vectors of the weight matrix.
        eigens.V_WC (`torch.Tensor`):
            Right singular vectors of the weight matrix, multiplied by inverse of covariance matrix.
    """
    cache_file = lora_config.corda_config.cache_file
    covariance_file = lora_config.corda_config.covariance_file
    corda_method = lora_config.corda_config.corda_method
    verbose = lora_config.corda_config.verbose
    prune_temporary_fields = lora_config.corda_config.prune_temporary_fields

    # If cache exists, skip building
    if cache_file is not None and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cache = torch.load(cache_file, map_location=get_model_device(model))
        r_with_layer = cache.get("r_with_layer", None)
        model._corda_r_with_layer = r_with_layer
        for name, module in target_modules(model, lora_config):
            module.eigens = CordaEigens(
                S_WC=cache[f"{name}.eigens.S_WC"],
                U_WC=cache[f"{name}.eigens.U_WC"],
                V_WC=cache[f"{name}.eigens.V_WC"],
            )
        crop_corda_eigens(model, lora_config, r_with_layer)
    else:
        # Specify CorDA method for each layer
        if corda_method is None:
            raise ValueError("corda_method is required when cache_file is not provided.")
        for name, module in target_modules(model, lora_config):
            module.corda_method = corda_method

        # Calculate covariance matrix
        calib_cov_distribution(model, lora_config, run_model, hooked_model, covariance_file)

        # Calculate eigens
        collect_eigens(model, lora_config, verbose)

        # --- Begin: Rank Strategy Integration ---
        # Collect statistics for rank search
        total_W_size = {}
        total_S_cov = {}
        total_S_WC = {}
        my_layers_keys = []
        for name, module in target_modules(model, lora_config):
            w_size = getattr(module, '_corda_w_size', None)
            S_cov = getattr(module, '_corda_S_cov', None)
            S_WC = getattr(module, '_corda_S_WC', None)
            total_W_size[name] = w_size
            total_S_cov[name] = S_cov
            total_S_WC[name] = S_WC
            my_layers_keys.append(name)
        # Get param_ratio and val_adjust from config if present, else use defaults
        param_ratio = getattr(lora_config.corda_config, 'param_ratio', None)
        val_adjust = getattr(lora_config.corda_config, 'val_adjust', False)
        if param_ratio is not None:
            r_with_layer, pruned_params, total_params = rank_search_mintomax(
                total_W_size, total_S_cov, total_S_WC, param_ratio, my_layers_keys, val_adjust
            )
            model._corda_r_with_layer = r_with_layer  # Store mapping on model
            for name, module in target_modules(model, lora_config):
                module.rank = r_with_layer.get(name, lora_config.r)
        else:
            # Fallback: use global rank
            for name, module in target_modules(model, lora_config):
                r_key = get_pattern_key(lora_config.rank_pattern.keys(), name)
                module.rank = lora_config.rank_pattern.get(r_key, lora_config.r)
            model._corda_r_with_layer = {name: module.rank for name, module in target_modules(model, lora_config)}

        crop_corda_eigens(model, lora_config, model._corda_r_with_layer)

        # Remove redundant fields if exist
        if prune_temporary_fields:
            for name, module in target_modules(model, lora_config):
                if hasattr(module, "sample_count"):
                    del module.sample_count
                if hasattr(module, "covariance_matrix"):
                    del module.covariance_matrix
                if hasattr(module, "corda_method"):
                    del module.corda_method
                if hasattr(module, "rank"):
                    del module.rank

        # Save cache to disk
        if cache_file is not None:
            cache: dict[str, Any] = {}
            for name, module in target_modules(model, lora_config):
                cache[f"{name}.eigens.S_WC"] = module.eigens.S_WC
                cache[f"{name}.eigens.U_WC"] = module.eigens.U_WC
                cache[f"{name}.eigens.V_WC"] = module.eigens.V_WC
            cache["r_with_layer"] = model._corda_r_with_layer
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save(cache, cache_file)


@torch.no_grad()
def calib_cov_distribution(
    model: nn.Module,
    config: LoraConfig,
    run_model: Optional[Callable[[], None]],
    hooked_model: Optional[nn.Module],
    covariance_file: Optional[str],
):
    if covariance_file is not None and os.path.exists(covariance_file) and os.path.getsize(covariance_file) > 0:
        all_covariance_matrix = torch.load(covariance_file, map_location=get_model_device(model))
        for name, module in target_modules(model, config):
            module.covariance_matrix = all_covariance_matrix[name]
        return

    if run_model is None:
        raise ValueError("run_model must be specified when covariance file and cache file aren't built.")
    if hooked_model is None:
        hooked_model = model
    hooked_model.eval()

    def hook(module, input, output):
        input = input[0].detach().squeeze(0).data  ## (context_length = 2048, dim)
        if not config.corda_config.use_float16_for_covariance:
            input = input.float()
        input = input / torch.max(input).abs()

        # check if input is valid
        if torch.isnan(input).any() or torch.isinf(input).any():
            raise ValueError("Invalid value found in input, please check your input data.")

        # calculate covariance and check if it's valid
        covariance = input.t().matmul(input)
        if torch.isnan(covariance).any() or torch.isinf(covariance).any():
            raise ValueError(
                "Invalid value found in covariance. Please file an issue at https://github.com/huggingface/peft/issues."
            )

        # add to module
        module.sample_count += 1
        module.covariance_matrix += covariance

        # free memory
        del covariance, input

    handles = []
    for name, module in target_modules(hooked_model, config):
        module.sample_count = 0
        module.covariance_matrix = 0
        handles.append(module.register_forward_hook(hook))

    run_model()

    # Clear the hooks
    for handle in handles:
        handle.remove()

    # In some edge cases you might need to hook a model different from the model to add adapters,
    # this case you would specify `hooked_model` and set it to a different model from `model`.
    if hooked_model is not model:
        targets = {}
        for name, module in target_modules(model, config):
            targets[name] = module
        for name, module in target_modules(hooked_model, config):
            # There can be modules used only in inference, but not training
            # Exclude modules not in target model to prevent KeyError in this case
            if name in targets:
                targets[name].sample_count = module.sample_count
                targets[name].covariance_matrix = module.covariance_matrix

    # Divide by sample count
    for name, module in target_modules(model, config):
        module.covariance_matrix /= module.sample_count

    # Save covariance to disk
    if covariance_file is not None:
        all_covariance_matrix = {}
        for name, module in target_modules(model, config):
            all_covariance_matrix[name] = module.covariance_matrix
        os.makedirs(os.path.dirname(covariance_file), exist_ok=True)
        torch.save(all_covariance_matrix, covariance_file)


@torch.no_grad()
def collect_eigens(
    model: nn.Module,
    config: LoraConfig,
    verbose: bool,
):
    """Call collect_eigens_for_layer and store result in key `eigens` of each layer."""
    linear_modules = []
    for name, module in target_modules(model, config):
        linear_modules.append((name, module))
    if verbose:
        linear_modules = tqdm(linear_modules, desc="Collecting eigens")
    for name, module in linear_modules:
        module.eigens = collect_eigens_for_layer(module, config)


@torch.no_grad()
def collect_eigens_for_layer(
    linear: nn.Linear,
    config: LoraConfig,
) -> CordaEigens:
    w = linear.weight.data.float()
    out_dim = w.size(0)
    in_dim = w.size(1)
    min_dim = min(in_dim, out_dim)

    if not hasattr(linear, "covariance_matrix"):
        raise ValueError(
            "Covariance matrix not found in linear module. Please do not call this function directly, "
            "instead call `preprocess_corda`. If your usage is correct but this error still encounters, "
            "please file an issue at https://github.com/huggingface/peft/issues."
        )
    covariance_matrix = linear.covariance_matrix.float()

    damp = 0.01
    while True:
        compensate = torch.diag(
            torch.ones(covariance_matrix.size(0)).to(covariance_matrix.device)
            * torch.mean(torch.diag(covariance_matrix))
            * damp
        )
        fix_covariance_matrix = covariance_matrix + compensate
        cov_inv = torch.linalg.inv(fix_covariance_matrix)
        inv_error = torch.dist(
            fix_covariance_matrix @ cov_inv, torch.eye(covariance_matrix.size(0)).to(get_model_device(linear))
        ).item()
        if inv_error < 0.05:
            break
        else:
            damp = damp * 2
    w = w @ fix_covariance_matrix  ## w: out_dim, in_dim; covariance_matrix: in_dim, in_dim

    # SVD of covariance
    _, S_cov, _ = torch.linalg.svd(fix_covariance_matrix, full_matrices=False)
    # SVD of weight
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    V = (Vh @ cov_inv).transpose(0, 1)
    norm_of_VandCovinv = torch.sqrt((V ** 2).sum(dim=0))
    S_adjusted = S * norm_of_VandCovinv

    # Store for rank search
    linear._corda_w_size = (out_dim, in_dim)
    linear._corda_S_cov = S_cov.cpu().numpy()
    linear._corda_S_WC = S_adjusted.cpu().numpy()

    # Offload U and V to CPU, they consume too much memory
    U = U.cpu()
    V = V.cpu()
    return CordaEigens(
        S_WC=S,
        U_WC=U,
        V_WC=V,
    )


@torch.no_grad()
def crop_corda_eigens(model: nn.Module, config: LoraConfig, r_with_layer: dict):
    for name, module in target_modules(model, config):
        r = r_with_layer[name]
        if module.corda_method == "ipm":
            module.eigens.S_WC = module.eigens.S_WC[: r].clone()
            module.eigens.U_WC = module.eigens.U_WC[:, : r].clone().to(get_model_device(model))
            module.eigens.V_WC = module.eigens.V_WC[:, : r].clone().to(get_model_device(model))
        elif module.corda_method == "kpm":
            module.eigens.S_WC = module.eigens.S_WC[-r :].clone()
            module.eigens.U_WC = module.eigens.U_WC[:, -r :].clone().to(get_model_device(model))
            module.eigens.V_WC = module.eigens.V_WC[:, -r :].clone().to(get_model_device(model))
        else:
            raise ValueError(f"Invalid corda_method found: {module.corda_method}, it should be 'ipm' or 'kpm'.")

        # Sanity check
        if module.eigens.S_WC.size(0) != r:
            raise ValueError(
                f"rank mismatch: {module.eigens.S_WC.size(0)} vs. {r},"
                "please file an issue at https://github.com/huggingface/peft/issues."
            )
        if module.eigens.U_WC.size(0) != module.weight.size(0):
            raise ValueError(
                f"U size mismatch: {module.eigens.U_WC.size(0)} vs. {module.weight.size(0)},"
                "please file an issue at https://github.com/huggingface/peft/issues."
            )
        if module.eigens.U_WC.size(1) != r:
            raise ValueError(
                f"U size mismatch: {module.eigens.U_WC.size(1)} vs. {r},"
                "please file an issue at https://github.com/huggingface/peft/issues."
            )
        if module.eigens.V_WC.size(0) != module.weight.size(1):
            raise ValueError(
                f"V size mismatch: {module.eigens.V_WC.size(0)} vs. {module.weight.size(1)},"
                "please file an issue at https://github.com/huggingface/peft/issues."
            )
        if module.eigens.V_WC.size(1) != r:
            raise ValueError(
                f"V size mismatch: {module.eigens.V_WC.size(1)} vs. {r},"
                "please file an issue at https://github.com/huggingface/peft/issues."
            )

def rank_search_mintomax(total_W_size, total_S_cov, total_S_WC, param_ratio, my_layers_keys, val_adjust=False):
    total_params = 0
    keep_with_layer = []
    win_add_wout = []
    WC_cumsum = {}
    for layername in my_layers_keys:
        w_out, w_in = total_W_size[layername][0], total_W_size[layername][1]
        total_params += w_out * w_in
        keep_with_layer.append(min(w_in, w_out))
        win_add_wout.append(w_in + w_out)
        WC_cumsum[layername] = np.cumsum(total_S_WC[layername])
    quota_params = total_params * param_ratio

    win_add_wout = np.array(win_add_wout)
    keep_with_layer = np.array(keep_with_layer)

    last_ratio = np.floor(((keep_with_layer * win_add_wout).sum() / total_params) * 10) / 10
    print("start ratio:", (keep_with_layer * win_add_wout).sum() / total_params)
    if val_adjust:
        print("\n --- value for rank will be adjusted by coefficients ---\n")
        max_coefficient = 0
        coefficient = {}
        for layer_id in range(len(my_layers_keys)):
            layername = my_layers_keys[layer_id]
            S_cov = total_S_cov[layername]
            current_coefficient = np.sqrt(total_W_size[layername][1] * S_cov[0]) / S_cov[-1]
            coefficient[layername] = current_coefficient
            if current_coefficient > max_coefficient:
                max_coefficient = current_coefficient
            print("coefficient", layername, current_coefficient)
        print("max coefficient:", max_coefficient)
    while (keep_with_layer * win_add_wout).sum() / total_params > param_ratio + (1e-4):
        current_ratio = (keep_with_layer * win_add_wout).sum() / total_params
        if np.floor(current_ratio * 10) / 10 < last_ratio:
            print("current ratio:", current_ratio)
            print(keep_with_layer)
            print("min_val:", min_val)
            last_ratio -= 0.1
        min_val = 1e8
        min_id = 0
        max_current_val = 0
        for layer_id in range(len(my_layers_keys)):
            layername = my_layers_keys[layer_id]
            S = total_S_WC[layername]
            current_val = S[keep_with_layer[layer_id]-1] / WC_cumsum[layername][keep_with_layer[layer_id]-2]
            if current_val > max_current_val:
                max_current_val = current_val
        for layer_id in range(len(my_layers_keys)):
            layername = my_layers_keys[layer_id]
            S = total_S_WC[layername]
            S_cov = total_S_cov[layername]
            current_val = S[keep_with_layer[layer_id]-1] / WC_cumsum[layername][keep_with_layer[layer_id]-2]
            if val_adjust:
                current_val = (current_val / max_current_val) * np.log(coefficient[layername]) / np.log(max_coefficient)
            if current_val < min_val:
                min_val = current_val
                min_id = layer_id
        keep_with_layer[min_id] = keep_with_layer[min_id] - 1
    print("final")
    print(keep_with_layer)
    pruned_params = (keep_with_layer * win_add_wout).sum()
    r_with_layer = {}
    for layer_id in range(len(my_layers_keys)):
        layername = my_layers_keys[layer_id]
        total_rank = min(total_W_size[layername])
        r_with_layer[layername] = total_rank - keep_with_layer[layer_id]
    return r_with_layer, pruned_params, total_params 