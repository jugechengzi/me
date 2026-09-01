from copy import deepcopy
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.layer_stats import get_cov
from util import nethook
from util.generate import generate_fast

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .memit_joint import (
    get_hidden_size,
    get_mi_cache_entry,
    load_cached_context_templates,
    load_cached_mi,
    save_cached_mi,
)
from omegaconf import DictConfig

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
covs = []


def load_cov(cfg, model, tok):
    """Load each covariance once and retain it on CPU."""
    covs.clear()
    for layer in cfg.llms.layers:
        cov = get_cov(
            cfg,
            model,
            tok,
            layer,
            cfg.llms.mom2_dataset,
            cfg.llms.mom2_n_samples,
            cfg.llms.mom2_dtype,
            force_recompute=False,
        )
        if cfg.cov_mode == "random":
            print("Using random covariance matrix!")
            cov = torch.randn_like(cov)
        if cfg.cov_mode == "identity":
            print("Using identity covariance matrix!")
            cov = torch.eye(cov.shape[0])
        covs.append(cov.cpu())

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig
):
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        requests[i]["target_new"] = " " + request["target_new"]
    if not requests:
        raise ValueError("Original MEMIT requires at least one edit")

    print("Running original closed-form MEMIT")
    layers = cfg.llms.layers
    load_cov(cfg, model, tok)
    cache_c = [
        torch.zeros_like(cov, device="cpu")
        if cfg.algs.add_old_keys
        else None
        for cov in covs
    ]

    z_layer = layers[-1]
    all_cache_entries = [
        get_mi_cache_entry(cfg, model, tok, request, z_layer)
        for request in requests
    ]
    context_templates = load_cached_context_templates(all_cache_entries)
    if context_templates is None:
        context_templates = get_context_templates(model, tok)

    for requests_chunk in chunks(requests, cfg.bs):
        batch_edit(
            cfg,
            model,
            tok,
            requests_chunk,
            device,
            cache_c,
            context_templates,
        )
    return model


def batch_edit(
    cfg, model, tok, requests, device, cache_c, context_templates
):
    # Retrieve weights that user desires to change
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }
    # Compute z for final layer
    z_layer = cfg.llms.layers[-1]
    z_list = [None] * len(requests)
    cache_entries = [
        get_mi_cache_entry(cfg, model, tok, request, z_layer)
        for request in requests
    ]
    missing_indices = []
    start_time = time.time()
    for index, (cache_path, metadata) in enumerate(cache_entries):
        cached_result = load_cached_mi(
            cache_path=cache_path,
            expected_metadata=metadata,
            hidden_size=get_hidden_size(model),
            force_recompute=cfg.algs.mi_cache_force_recompute,
        )
        if cached_result is None:
            missing_indices.append(index)
        else:
            cached_target, _ = cached_result
            z_list[index] = cached_target

    for index in missing_indices:
        cur_z, delta = compute_z(
            model,
            tok,
            requests[index],
            cfg,
            z_layer,
            context_templates,
        )
        target = cur_z.detach().to(device="cpu", dtype=torch.float32)
        z_list[index] = target
        cache_path, metadata = cache_entries[index]
        save_cached_mi(
            cache_path=cache_path,
            target=target,
            delta=delta,
            metadata={
                **metadata,
                "context_templates": context_templates,
            },
        )

    print(
        f"Original MEMIT m_i cache: "
        f"{len(requests) - len(missing_indices)} hit(s), "
        f"{len(missing_indices)} miss(es); batch prepared in "
        f"{time.time() - start_time:.2f} seconds"
    )
    zs = torch.stack(z_list, dim=1).to(device)  # [hidden, batch]

    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        if cfg.negetive_prompt_test:
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["negetive_prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        else:
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        targets = zs - cur_zs#[dim,bs]
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        layer_ks, targets = (
            layer_ks.double(),
            targets.double()
        )
        resid = targets / (len(cfg.llms.layers) - i)  # Distribute residual across layers

        cov = covs[i].to(device=device, dtype=torch.float64)

        start_time = time.time()
        coef = cfg.llms.mom2_update_weight[i]
        key_gram = layer_ks @ layer_ks.T
        system_matrix = key_gram + coef * cov
        if cache_c[i] is not None:
            system_matrix = system_matrix + cache_c[i].to(
                device=device, dtype=torch.float64
            )
        if cfg.llms.memit_ori.L2:
            system_matrix = system_matrix + cfg.llms.memit_ori.L2 * torch.eye(
                layer_ks.shape[0],
                device=device,
                dtype=torch.float64,
            )
        upd_matrix = torch.linalg.solve(
            system_matrix,
            layer_ks @ resid.T,
        )
        end_time = time.time()
        print(f"Solved for update matrix in {end_time - start_time:.2f} seconds")
        if cache_c[i] is not None:
            cache_c[i].add_(
                key_gram.detach().to(
                    device="cpu", dtype=cache_c[i].dtype
                )
            )
        # Adjust update matrix shape
        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
