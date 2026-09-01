"""Cross-layer joint-gradient MEMIT implementation."""

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.layer_stats import get_cov
from util import nethook
from util.generate import generate_fast
from util.utility import ensure_file_directory

from .compute_z import compute_z, find_fact_lookup_idx
from omegaconf import DictConfig

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
covs=[]#将K0K0T先从文件读取到cpu上，之后不用再读文件，可以显著加快速度（空间换时间），尤其是批次batch size小的时候。
def load_cov(cfg,model,tok):
    covs.clear()
    layers=cfg.llms.layers
    for i, layer in enumerate(layers):
        cov=get_cov(
            cfg,
            model,
            tok,
            layer,
            cfg.llms.mom2_dataset,
            cfg.llms.mom2_n_samples,
            cfg.llms.mom2_dtype,
            force_recompute=False,
        )
        covs.append(cov)

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def get_fc_dim(model,cfg):
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim=W_out.shape[0] if W_out.shape[0]>W_out.shape[1] else W_out.shape[1]
    return fc_dim

def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig
):
    """
    Returns a model with the desired changes.
    :return: (1) the updated model
    """

    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    #查看KKT是否已经计算好。
    for i, layer in enumerate(layers):
        Cpathi = cfg.cache_dir + "/stats/"+ cfg.llms.alias.replace("/","-") + "/layer-" + str(layer) + ".npz"
        ensure_file_directory(Cpathi)
        if not os.path.exists(Cpathi):#then compute
            print("The key matrix of old memory K0K0T for model {} layer {} "
                  "does not exist and now calculate.".format(cfg.llms.alias, layer))
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
            #这个内部会自动保存，我们不需要再额外管。
    load_cov(cfg,model,tok)
    # Treat every requested edit as part of one global training set. ``cfg.bs``
    # is only the micro-batch size; updates are merged into the model once,
    # after all epochs have completed.
    batch_edit(cfg,model,tok,requests,device)
    return model

def batch_edit(cfg, model, tok, requests, device):
    """
    Jointly optimize all rewrite-layer updates through model forward passes.

    The base model remains frozen while differentiable full-rank update
    matrices are injected into every rewrite module. ``target_loss_mode``
    selects one of two independent target objectives:

        hidden_mse:
        mean_(i,c) ||m_i,c(W + Delta W) - (m_i,c(W) + delta_i)||_2^2

        end_to_end_ce:
        mean_i sum_c w_c mean_t CE(
            p_(W + Delta W)(y_t | x_i,c,y_<t), y_t
        )

    Both modes add

        + alpha * sum_l tr(Delta W_l C_l Delta W_l.T).

    Each edit is expanded with every context template. In ``hidden_mse``,
    contexts belonging to one edit share its optimized displacement
    ``delta_i`` while retaining their own frozen-base-model representation
    ``m_i,c(W)``. In ``end_to_end_ce``, every expanded prompt is trained
    directly against the requested target tokens and ``compute_z`` is not
    called. E2E context weights sum to one per edit. By default they are
    uniform; ``original_prompt_loss_share`` can reserve a larger fraction for
    the unprefixed/original prompt while all generated contexts split the
    remainder.

    AdamW applies decoupled weight decay directly to every ``Delta W_l``;
    weight decay is deliberately not included as an explicit loss term.

    ``C_l`` is the non-centered second moment of old keys at layer ``l``.
    """
    if not requests:
        raise ValueError("Joint MEMIT training requires at least one edit")

    joint_hparams = cfg.llms.memit_joint
    target_loss_mode = str(
        joint_hparams.get("target_loss_mode", "hidden_mse")
    ).lower()
    if target_loss_mode not in {"hidden_mse", "end_to_end_ce"}:
        raise ValueError(
            "target_loss_mode must be one of: hidden_mse, end_to_end_ce"
        )

    # Reuse a stable template set across processes. In hidden-MSE mode these
    # templates are also used by compute_z; in end-to-end CE mode they only
    # define the expanded teacher-forcing training set.
    z_layer = cfg.llms.layers[-1]
    cache_entries = [
        get_mi_cache_entry(cfg, model, tok, request, z_layer)
        for request in requests
    ]
    if cfg.algs.joint_context_training:
        context_templates = get_stable_joint_context_templates(
            cfg=cfg,
            model=model,
            tok=tok,
            cache_entries=cache_entries,
        )
    else:
        context_templates = [["{}"]]

    edit_deltas = None
    if target_loss_mode == "hidden_mse":
        delta_list = [None] * len(requests)
        missing_indices = []

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
                _, cached_delta = cached_result
                delta_list[index] = cached_delta

        print(
            f"m_i cache: {len(requests) - len(missing_indices)} hit(s), "
            f"{len(missing_indices)} miss(es)"
        )

        if missing_indices:
            for completed, index in enumerate(missing_indices, start=1):
                cur_z, delta = compute_z(
                    model,
                    tok,
                    requests[index],
                    cfg,
                    z_layer,
                    context_templates,
                )
                target = cur_z.detach().to(
                    device="cpu", dtype=torch.float32
                )
                delta = delta.detach().to(
                    device="cpu", dtype=torch.float32
                )
                delta_list[index] = delta
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
                    f"computed and cached m_i {completed}/"
                    f"{len(missing_indices)}"
                )

        # The canonical targets remain in the per-edit cache for original
        # closed-form MEMIT. Joint training only needs each edit displacement.
        edit_deltas = torch.stack(delta_list, dim=0)
        del delta_list
    else:
        print(
            "Joint target loss: end_to_end_ce; skipping compute_z and "
            "m_i/delta caches"
        )

    training_samples = build_joint_training_samples(
        requests=requests,
        context_templates=context_templates,
        use_context_templates=cfg.algs.joint_context_training,
    )
    num_edits = len(requests)
    num_contexts = len(training_samples) // len(requests)
    print(
        f"joint training expansion: {len(requests)} edit(s) * "
        f"{num_contexts} context(s) = {len(training_samples)} sample(s)"
    )
    original_prompt_loss_share = joint_hparams.get(
        "original_prompt_loss_share", None
    )
    if (
        target_loss_mode != "end_to_end_ce"
        and original_prompt_loss_share is not None
    ):
        raise ValueError(
            "original_prompt_loss_share is currently supported only for "
            "target_loss_mode=end_to_end_ce"
        )
    sample_loss_weights = torch.tensor(
        build_joint_sample_loss_weights(
            training_samples=training_samples,
            num_edits=num_edits,
            original_prompt_loss_share=(
                original_prompt_loss_share
                if target_loss_mode == "end_to_end_ce"
                else None
            ),
        ),
        dtype=torch.float64,
    )
    effective_original_share = sum(
        weight.item()
        for weight, sample in zip(sample_loss_weights, training_samples)
        if sample["is_original_prompt"]
    ) / num_edits
    print(
        "Joint sample-loss weighting: original prompt "
        f"{effective_original_share:.2%}, all context prompts "
        f"{1.0 - effective_original_share:.2%} per edit"
    )

    nethook.set_requires_grad(False, model)
    base_context_ms = None
    if target_loss_mode == "hidden_mse":
        base_context_ms = compute_joint_base_context_vectors(
            model=model,
            tok=tok,
            training_samples=training_samples,
            fact_token_strategy=cfg.llms.fact_token,
            z_layer=z_layer,
            layer_module_tmp=cfg.llms.layer_module_tmp,
            device=device,
            batch_size=cfg.bs,
        )
    weights = []
    upd_matrices = []
    hook_handles = []

    for layer in cfg.llms.layers:
        module_name = cfg.llms.rewrite_module_tmp.format(layer)
        module = nethook.get_module(model, module_name)
        weight = nethook.get_parameter(model, f"{module_name}.weight")
        if weight is None:
            raise ValueError(f"Rewrite module {module_name} has no weight")

        # Keep full updates in FP32 even when the frozen base model is BF16.
        update = torch.nn.Parameter(
            torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
        )
        weights.append(weight)
        upd_matrices.append(update)
        hook_handles.append(
            module.register_forward_hook(make_update_hook(update, module_name))
        )

    layer_norm_mode = str(joint_hparams.layer_norm_mode).lower()
    if layer_norm_mode not in {"none", "balance", "cap"}:
        raise ValueError(
            "layer_norm_mode must be one of: none, balance, cap"
        )
    if joint_hparams.layer_norm_beta < 0:
        raise ValueError("layer_norm_beta must be non-negative")
    if (
        layer_norm_mode == "cap"
        and joint_hparams.layer_update_max_ratio <= 0
    ):
        raise ValueError(
            "layer_update_max_ratio must be positive in cap mode"
        )
    if joint_hparams.gd_weight_decay < 0:
        raise ValueError("AdamW weight decay must be non-negative")
    base_weight_norms = [
        torch.linalg.vector_norm(
            weight.detach(), dtype=torch.float32
        ).item()
        for weight in weights
    ]
    if any(
        not np.isfinite(norm) or norm <= 0 for norm in base_weight_norms
    ):
        raise FloatingPointError(
            "All edited base weights must have finite positive norms"
        )
    decay_factor = 1.0 - joint_hparams.gd_lr * joint_hparams.gd_weight_decay
    if decay_factor <= 0:
        raise ValueError(
            "AdamW requires gd_lr * gd_weight_decay < 1 for a positive "
            "per-step decay factor"
        )
    optimizer = torch.optim.AdamW(
        upd_matrices,
        lr=joint_hparams.gd_lr,
        weight_decay=joint_hparams.gd_weight_decay,
    )
    print(
        f"Joint optimizer AdamW: lr={joint_hparams.gd_lr:.3e}, "
        f"weight_decay={joint_hparams.gd_weight_decay:.3e}, "
        f"per-step decay factor={decay_factor:.8f}, "
        f"planned decay-only factor="
        f"{decay_factor ** joint_hparams.joint_epochs:.6f}"
    )
    print(
        f"Joint target objective: mode={target_loss_mode}, "
        f"loss_layer={max(cfg.llms.v_loss_layer, z_layer)}"
    )
    print(
        f"Layer norm regularizer: mode={layer_norm_mode}, "
        f"beta={joint_hparams.layer_norm_beta:.3e}, "
        f"max_relative_norm="
        f"{joint_hparams.layer_update_max_ratio:.3e}"
    )
    previous_epoch_loss = None
    num_training_samples = len(training_samples)
    checkpoint_path, checkpoint_metadata = get_joint_checkpoint_entry(
        cfg=cfg,
        requests=requests,
        edit_deltas=edit_deltas,
        upd_matrices=upd_matrices,
        training_samples=training_samples,
        target_loss_mode=target_loss_mode,
        original_prompt_loss_share=original_prompt_loss_share,
    )
    start_epoch, previous_epoch_loss = load_joint_checkpoint(
        checkpoint_path=checkpoint_path,
        expected_metadata=checkpoint_metadata,
        upd_matrices=upd_matrices,
        optimizer=optimizer,
        resume=cfg.algs.joint_checkpoint_resume,
    )
    previous_update_norm = get_global_tensor_norm(
        [update.detach() for update in upd_matrices],
        name="update",
    )

    try:
        for epoch in range(start_epoch, joint_hparams.joint_epochs):
            permutation = torch.randperm(num_training_samples).tolist()
            epoch_target_sum = 0.0
            per_edit_loss_sums = torch.zeros(num_edits, dtype=torch.float64)
            per_edit_loss_weights = torch.zeros(
                num_edits, dtype=torch.float64
            )
            per_edit_loss_counts = torch.zeros(num_edits, dtype=torch.int64)
            num_micro_batches = (
                num_training_samples + cfg.bs - 1
            ) // cfg.bs

            # Accumulate the exact gradient of the target-loss mean over all
            # edits. Parameters remain unchanged throughout the epoch.
            optimizer.zero_grad(set_to_none=True)

            for micro_step, batch_indices in enumerate(
                chunks(permutation, cfg.bs), start=1
            ):
                batch_samples = [
                    training_samples[index] for index in batch_indices
                ]
                batch_request_indices = [
                    sample["request_index"] for sample in batch_samples
                ]
                if target_loss_mode == "hidden_mse":
                    # Every context retains its own frozen-model
                    # representation; only the optimized edit displacement
                    # is shared.
                    batch_targets = (
                        base_context_ms[batch_indices]
                        + edit_deltas[batch_request_indices]
                    ).to(device)
                    model_inputs, lookup_idxs = prepare_joint_batch(
                        tok=tok,
                        requests=batch_samples,
                        fact_token_strategy=cfg.llms.fact_token,
                        device=device,
                    )
                    current_ms = get_joint_target_vectors(
                        model=model,
                        model_inputs=model_inputs,
                        lookup_idxs=lookup_idxs,
                        z_layer=z_layer,
                        layer_module_tmp=cfg.llms.layer_module_tmp,
                    )
                    batch_edit_losses = (
                        (current_ms.float() - batch_targets.float())
                        .square()
                        .sum(dim=1)
                    )
                else:
                    # Each expanded sample receives its own teacher-forcing
                    # CE. The result is one token-mean loss per sample, so
                    # target token length does not change that sample's
                    # configured contribution to the objective.
                    model_inputs, target_positions, target_ids = (
                        prepare_joint_ce_batch(
                            tok=tok,
                            samples=batch_samples,
                            device=device,
                        )
                    )
                    batch_edit_losses = get_joint_ce_losses(
                        model=model,
                        model_inputs=model_inputs,
                        target_positions=target_positions,
                        target_ids=target_ids,
                        loss_layer=max(cfg.llms.v_loss_layer, z_layer),
                        layer_module_tmp=cfg.llms.layer_module_tmp,
                        ln_f_module=cfg.llms.ln_f_module,
                        lm_head_module=cfg.llms.lm_head_module,
                    )
                if not torch.isfinite(batch_edit_losses).all():
                    raise FloatingPointError(
                        f"Non-finite target loss at epoch {epoch + 1}, "
                        f"micro-batch {micro_step}; updates were not merged"
                    )
                batch_size = len(batch_indices)
                batch_loss_weights = sample_loss_weights[
                    batch_indices
                ].to(
                    device=batch_edit_losses.device,
                    dtype=batch_edit_losses.dtype,
                )
                weighted_batch_target_sum = (
                    batch_edit_losses * batch_loss_weights
                ).sum()
                # Sample weights sum to one within every edit. Dividing each
                # micro-batch contribution by num_edits therefore accumulates
                # the exact weighted full-training-set mean before the single
                # optimizer step at the end of the epoch.
                scaled_target_loss = weighted_batch_target_sum / num_edits
                scaled_target_loss.backward()
                epoch_target_sum += weighted_batch_target_sum.detach().item()
                # Report a locally normalized weighted mean for readable
                # micro-batch logs; this value is not used for gradients.
                target_loss = (
                    weighted_batch_target_sum / batch_loss_weights.sum()
                )
                request_index_tensor = torch.tensor(
                    batch_request_indices, dtype=torch.int64
                )
                detached_batch_losses = batch_edit_losses.detach().to(
                    device="cpu", dtype=torch.float64
                )
                # Use the canonical CPU/FP64 weights for accounting instead
                # of weights rounded to the model-loss dtype on the GPU.
                detached_batch_weights = sample_loss_weights[batch_indices]
                per_edit_loss_sums.index_add_(
                    0,
                    request_index_tensor,
                    detached_batch_losses * detached_batch_weights,
                )
                per_edit_loss_weights.index_add_(
                    0,
                    request_index_tensor,
                    detached_batch_weights,
                )
                per_edit_loss_counts.index_add_(
                    0,
                    request_index_tensor,
                    torch.ones(batch_size, dtype=torch.int64),
                )

                if (
                    micro_step == 1
                    or micro_step % cfg.algs.gd_log_interval == 0
                    or micro_step == num_micro_batches
                ):
                    print(
                        f"joint epoch {epoch + 1}/"
                        f"{joint_hparams.joint_epochs}, micro-batch "
                        f"{micro_step}/"
                        f"{num_micro_batches}: target "
                        f"{target_loss.detach().item():.6e} "
                        f"(gradient accumulated)"
                    )

            # The preservation term is independent of edit micro-batches, so
            # compute its gradient exactly once per full-batch update. Layers
            # are processed separately to avoid keeping every large cov in
            # GPU memory and in one autograd graph simultaneously.
            preserve_loss_value = 0.0
            if joint_hparams.alpha:
                for update, cov in zip(upd_matrices, covs):
                    covariance = cov.to(
                        device=update.device, dtype=update.dtype
                    )
                    layer_preserve_loss = trace_preservation_loss(
                        update, covariance
                    )
                    if not torch.isfinite(layer_preserve_loss):
                        raise FloatingPointError(
                            f"Non-finite preservation loss at epoch "
                            f"{epoch + 1}; updates were not merged"
                        )
                    (joint_hparams.alpha * layer_preserve_loss).backward()
                    preserve_loss_value += (
                        layer_preserve_loss.detach().item()
                    )
                    del covariance, layer_preserve_loss

            # The layer-norm regularizer also depends only on the update
            # matrices, so add its gradient once per full-batch update.  It
            # uses relative Frobenius norms ||Delta W_l||_F / ||W_l||_F to
            # remain comparable across layers and model families.
            layer_norm_loss, _ = layer_update_norm_regularization(
                upd_matrices=upd_matrices,
                base_weight_norms=base_weight_norms,
                mode=layer_norm_mode,
                max_relative_norm=(
                    joint_hparams.layer_update_max_ratio
                ),
            )
            if not torch.isfinite(layer_norm_loss):
                raise FloatingPointError(
                    f"Non-finite layer norm loss at epoch {epoch + 1}; "
                    "updates were not merged"
                )
            layer_norm_loss_value = layer_norm_loss.detach().item()
            if joint_hparams.layer_norm_beta and layer_norm_mode != "none":
                (
                    joint_hparams.layer_norm_beta * layer_norm_loss
                ).backward()

            epoch_target_loss = epoch_target_sum / num_edits
            epoch_loss = (
                epoch_target_loss
                + joint_hparams.alpha * preserve_loss_value
                + joint_hparams.layer_norm_beta * layer_norm_loss_value
            )
            if not torch.isfinite(torch.tensor(epoch_loss)):
                raise FloatingPointError(
                    f"Non-finite total loss at epoch {epoch + 1}; "
                    f"updates were not merged"
                )

            grad_norm = get_global_tensor_norm(
                [update.grad for update in upd_matrices],
                name="gradient",
            )
            clipped = False
            if (
                joint_hparams.gd_max_grad_norm > 0
                and grad_norm > joint_hparams.gd_max_grad_norm
            ):
                scale = joint_hparams.gd_max_grad_norm / max(
                    grad_norm, torch.finfo(torch.float32).eps
                )
                with torch.no_grad():
                    for update in upd_matrices:
                        if update.grad is not None:
                            update.grad.mul_(scale)
                clipped = True

            # Exactly one parameter update is performed per epoch, after all
            # target, preservation, and layer-norm gradients are accumulated.
            optimizer.step()
            _, relative_layer_norms = layer_update_norm_regularization(
                upd_matrices=upd_matrices,
                base_weight_norms=base_weight_norms,
                mode="none",
                max_relative_norm=(
                    joint_hparams.layer_update_max_ratio
                ),
            )
            relative_layer_norms = relative_layer_norms.detach()
            relative_layer_norm_mean = relative_layer_norms.mean().item()
            relative_layer_norm_std = relative_layer_norms.std(
                unbiased=False
            ).item()
            relative_layer_norm_cv = relative_layer_norm_std / max(
                relative_layer_norm_mean,
                torch.finfo(torch.float32).eps,
            )
            relative_layer_norm_min = relative_layer_norms.min().item()
            relative_layer_norm_max = relative_layer_norms.max().item()
            update_norm = get_global_tensor_norm(
                [update.detach() for update in upd_matrices],
                name="update",
            )
            relative_update_norm_change = abs(
                update_norm - previous_update_norm
            ) / max(
                previous_update_norm,
                torch.finfo(torch.float32).eps,
            )
            if not torch.all(per_edit_loss_counts == num_contexts):
                raise RuntimeError(
                    "Each edit must contribute exactly one sample per context"
                )
            if not torch.allclose(
                per_edit_loss_weights,
                torch.ones_like(per_edit_loss_weights),
                atol=1e-8,
                rtol=1e-8,
            ):
                raise RuntimeError(
                    "Per-edit target-loss weights must sum to one"
                )
            edit_loss_values = (
                per_edit_loss_sums / per_edit_loss_weights
            ).to(torch.float32)
            median_loss = torch.quantile(edit_loss_values, 0.5).item()
            p90_loss = torch.quantile(edit_loss_values, 0.9).item()
            max_loss = edit_loss_values.max().item()
            layer_norm_objective = ""
            if layer_norm_mode != "none":
                layer_norm_objective = (
                    f" + beta * layer_norm[{layer_norm_mode}] "
                    f"{joint_hparams.layer_norm_beta:.3e} * "
                    f"{layer_norm_loss_value:.6e}"
                )
            print(
                f"joint epoch {epoch + 1} complete: "
                f"loss {epoch_loss:.6e} = target "
                f"{epoch_target_loss:.6e} + alpha * preserve "
                f"{joint_hparams.alpha:.3e} * {preserve_loss_value:.6e}"
                f"{layer_norm_objective}; layer relative norm "
                f"min/mean/max/cv {relative_layer_norm_min:.6e}/"
                f"{relative_layer_norm_mean:.6e}/"
                f"{relative_layer_norm_max:.6e}/"
                f"{relative_layer_norm_cv:.6e}; "
                f"AdamW weight decay {joint_hparams.gd_weight_decay:.3e} "
                f"(factor {decay_factor:.8f}); "
                f"gradient norm {grad_norm:.6e}"
                f"{' (clipped)' if clipped else ''}; "
                f"update norm {update_norm:.6e} "
                f"(relative norm change "
                f"{relative_update_norm_change:.3e}); per-edit target "
                f"median/p90/max {median_loss:.6e}/"
                f"{p90_loss:.6e}/{max_loss:.6e}"
            )
            print(
                f"joint epoch {epoch + 1} layer relative update norms: "
                + ", ".join(
                    f"{layer}={relative_norm:.6e}"
                    for layer, relative_norm in zip(
                        cfg.llms.layers,
                        relative_layer_norms.cpu().tolist(),
                    )
                )
            )

            should_save_snapshot = (
                (epoch + 1) % cfg.algs.joint_checkpoint_interval == 0
                or epoch + 1 == joint_hparams.joint_epochs
            )
            if cfg.algs.joint_checkpoint_enabled and should_save_snapshot:
                # Keep a stable latest checkpoint for automatic resume.
                save_joint_checkpoint(
                    checkpoint_path=checkpoint_path,
                    metadata=checkpoint_metadata,
                    completed_epochs=epoch + 1,
                    previous_epoch_loss=epoch_loss,
                    upd_matrices=upd_matrices,
                    optimizer=optimizer,
                )
                # Save a complete edited-weight snapshot in exactly the same
                # format and directory convention used by load_model().
                save_epoch_model_weights(
                    cfg=cfg,
                    weights=weights,
                    upd_matrices=upd_matrices,
                    epoch=epoch + 1,
                )

            if previous_epoch_loss is not None:
                relative_change = abs(
                    previous_epoch_loss - epoch_loss
                ) / max(
                    abs(previous_epoch_loss),
                    torch.finfo(torch.float32).eps,
                )
                if (
                    relative_change <= cfg.algs.gd_tolerance
                    and relative_update_norm_change
                    <= cfg.algs.gd_tolerance
                ):
                    print(
                        f"joint training converged after epoch {epoch + 1}: "
                        f"relative epoch-loss change {relative_change:.3e}, "
                        f"relative update-norm change "
                        f"{relative_update_norm_change:.3e}"
                    )
                    break
            previous_epoch_loss = epoch_loss
            previous_update_norm = update_norm
    finally:
        for handle in hook_handles:
            handle.remove()

    # Merge all jointly optimized updates only after optimization finishes.
    with torch.no_grad():
        for layer, weight, update, base_weight_norm in zip(
            cfg.llms.layers,
            weights,
            upd_matrices,
            base_weight_norms,
        ):
            update_norm_value = torch.linalg.norm(update).item()
            print(
                f"LAYER {layer}: orig norm {torch.linalg.norm(weight)}, "
                f"upd norm {update_norm_value}, relative upd norm "
                f"{update_norm_value / base_weight_norm:.6e}"
            )
            weight.add_(update.to(device=weight.device, dtype=weight.dtype))


def layer_update_norm_regularization(
    upd_matrices, base_weight_norms, mode, max_relative_norm
):
    """Return a dimensionless layer-norm loss and relative layer norms.

    ``balance`` minimizes the squared coefficient of variation, encouraging
    all layers to have similar relative Frobenius norms. ``cap`` penalizes
    only the normalized amount by which a layer exceeds ``max_relative_norm``.
    ``none`` returns a zero loss while still providing diagnostics.
    """
    if len(upd_matrices) != len(base_weight_norms) or not upd_matrices:
        raise ValueError(
            "Update matrices and base weight norms must be non-empty and "
            "have matching lengths"
        )
    if mode not in {"none", "balance", "cap"}:
        raise ValueError(f"Unsupported layer norm mode: {mode}")

    reference_device = upd_matrices[0].device
    relative_norms = torch.stack(
        [
            (
                torch.linalg.vector_norm(update)
                / max(float(weight_norm), torch.finfo(torch.float32).eps)
            ).to(reference_device)
            for update, weight_norm in zip(
                upd_matrices, base_weight_norms
            )
        ]
    )
    if mode == "none":
        return relative_norms.new_zeros(()), relative_norms

    if mode == "balance":
        mean_relative_norm = relative_norms.mean()
        normalizer = mean_relative_norm.detach().clamp_min(
            torch.finfo(relative_norms.dtype).eps
        )
        loss = (
            (relative_norms - mean_relative_norm) / normalizer
        ).square().mean()
        return loss, relative_norms

    if max_relative_norm <= 0:
        raise ValueError("max_relative_norm must be positive in cap mode")
    violations = torch.relu(
        (relative_norms - max_relative_norm) / max_relative_norm
    )
    return violations.square().mean(), relative_norms


def get_global_tensor_norm(tensors, name):
    """Return a cross-device FP32 L2 norm and reject non-finite tensors."""
    squared_norm = 0.0
    found_tensor = False
    for tensor in tensors:
        if tensor is None:
            continue
        found_tensor = True
        tensor_norm = torch.linalg.vector_norm(
            tensor.detach().float()
        ).item()
        if not np.isfinite(tensor_norm):
            raise FloatingPointError(
                f"Non-finite {name} detected; updates were not merged"
            )
        squared_norm += tensor_norm * tensor_norm
    if not found_tensor:
        raise RuntimeError(f"No {name} tensors were produced")
    norm = float(np.sqrt(squared_norm))
    if not np.isfinite(norm):
        raise FloatingPointError(
            f"Non-finite {name} norm detected; updates were not merged"
        )
    return norm


def get_joint_checkpoint_entry(
    cfg,
    requests,
    edit_deltas,
    upd_matrices,
    training_samples,
    target_loss_mode,
    original_prompt_loss_share,
):
    """Build a stable checkpoint path for one joint-edit training run."""
    request_payload = [
        {
            "prompt": request["prompt"],
            "subject": request["subject"],
            "target_new": request["target_new"],
        }
        for request in requests
    ]
    training_payload = [
        {
            "request_index": int(sample["request_index"]),
            "prompt": sample["prompt"],
            "subject": sample["subject"],
        }
        for sample in training_samples
    ]
    metadata = {
        "model_name": cfg.llms.alias,
        "layers": [int(layer) for layer in cfg.llms.layers],
        "update_shapes": [list(update.shape) for update in upd_matrices],
        "request_digest": hashlib.sha256(
            json.dumps(
                request_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "num_requests": len(requests),
        "num_training_samples": len(training_samples),
        "joint_context_training": bool(cfg.algs.joint_context_training),
        "training_sample_digest": hashlib.sha256(
            json.dumps(
                training_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "optimizer": "AdamW",
        "gd_lr": float(cfg.llms.memit_joint.gd_lr),
        "gd_weight_decay": float(cfg.llms.memit_joint.gd_weight_decay),
        "gd_max_grad_norm": float(cfg.llms.memit_joint.gd_max_grad_norm),
        "alpha": float(cfg.llms.memit_joint.alpha),
        "layer_norm_mode": str(
            cfg.llms.memit_joint.layer_norm_mode
        ).lower(),
        "layer_norm_beta": float(
            cfg.llms.memit_joint.layer_norm_beta
        ),
        "layer_update_max_ratio": float(
            cfg.llms.memit_joint.layer_update_max_ratio
        ),
    }
    if target_loss_mode == "hidden_mse":
        if edit_deltas is None:
            raise ValueError("hidden_mse checkpoint requires edit deltas")
        # Keep the original metadata byte-for-byte compatible so existing
        # hidden-MSE resume checkpoints retain their stable hash.
        metadata = {
            "checkpoint_format_version": 5,
            "target_construction": "context-base-plus-shared-delta-v1",
            **metadata,
            "delta_digest": hashlib.sha256(
                edit_deltas.contiguous().numpy().tobytes()
            ).hexdigest(),
        }
    elif target_loss_mode == "end_to_end_ce":
        if original_prompt_loss_share is None:
            # Preserve compatibility with existing uniformly weighted E2E
            # resume checkpoints.
            metadata = {
                "checkpoint_format_version": 6,
                "target_construction": "context-teacher-forcing-ce-v1",
                "target_loss_mode": target_loss_mode,
                "loss_layer": int(
                    max(cfg.llms.v_loss_layer, cfg.llms.layers[-1])
                ),
                **metadata,
            }
        else:
            metadata = {
                "checkpoint_format_version": 7,
                "target_construction": (
                    "context-teacher-forcing-weighted-ce-v2"
                ),
                "target_loss_mode": target_loss_mode,
                "original_prompt_loss_share": float(
                    original_prompt_loss_share
                ),
                "loss_layer": int(
                    max(cfg.llms.v_loss_layer, cfg.llms.layers[-1])
                ),
                **metadata,
            }
    else:
        raise ValueError(f"Unsupported target loss mode: {target_loss_mode}")
    serialized = json.dumps(metadata, sort_keys=True).encode("utf-8")
    checkpoint_key = hashlib.sha256(serialized).hexdigest()
    model_dir = cfg.llms.alias.replace("/", "-")
    checkpoint_path = (
        Path(cfg.cache_dir)
        / "memit_joint_checkpoints"
        / model_dir
        / f"{checkpoint_key}.pt"
    )
    return checkpoint_path, metadata


def load_joint_checkpoint(
    checkpoint_path,
    expected_metadata,
    upd_matrices,
    optimizer,
    resume,
):
    """Restore full-rank deltas and optimizer state from a safe checkpoint."""
    if not resume or not checkpoint_path.is_file():
        return 0, None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["metadata"] != expected_metadata:
            print(f"Ignoring incompatible joint checkpoint: {checkpoint_path}")
            return 0, None
        saved_updates = checkpoint["upd_matrices"]
        if len(saved_updates) != len(upd_matrices):
            raise ValueError("wrong number of update matrices")
        for saved_update, update in zip(saved_updates, upd_matrices):
            if saved_update.shape != update.shape:
                raise ValueError("update matrix shape mismatch")
            if not torch.isfinite(saved_update).all():
                raise ValueError("non-finite update matrix")
        with torch.no_grad():
            for saved_update, update in zip(saved_updates, upd_matrices):
                update.copy_(
                    saved_update.to(
                        device=update.device, dtype=update.dtype
                    )
                )
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_epochs = int(checkpoint["completed_epochs"])
        previous_epoch_loss = checkpoint.get("previous_epoch_loss")
        print(
            f"Resumed joint optimization from {checkpoint_path} "
            f"after {completed_epochs} completed epoch(s)"
        )
        return completed_epochs, previous_epoch_loss
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Ignoring unreadable joint checkpoint {checkpoint_path}: {error}")
        return 0, None


def save_joint_checkpoint(
    checkpoint_path,
    metadata,
    completed_epochs,
    previous_epoch_loss,
    upd_matrices,
    optimizer,
):
    """Atomically save resumable joint-optimization state on CPU."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(
            {
                "metadata": metadata,
                "completed_epochs": completed_epochs,
                "previous_epoch_loss": previous_epoch_loss,
                "upd_matrices": [
                    update.detach().to(device="cpu", dtype=torch.float32)
                    for update in upd_matrices
                ],
                "optimizer": optimizer.state_dict(),
            },
            temporary_path,
        )
        os.replace(temporary_path, checkpoint_path)
        print(f"Saved joint checkpoint: {checkpoint_path}")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_epoch_model_weights(cfg, weights, upd_matrices, epoch):
    """Save an epoch in the exact full-weight format consumed by load_model."""
    model_name = cfg.llms.alias.replace("/", "-")
    epoch_load_name = f"{cfg.save_name}-epoch-{epoch:04d}"
    weights_path = (
        Path(cfg.cache_dir)
        / "saved_weights"
        / cfg.algs.name
        / f"{cfg.data}-{epoch_load_name}-{model_name}.pt"
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Match the final merge semantics: cast each FP32 update to the original
    # model dtype, then add it to the frozen base weight.
    edited_weights = {}
    with torch.no_grad():
        for layer, weight, update in zip(
            cfg.llms.layers, weights, upd_matrices
        ):
            weight_name = (
                f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
            )
            edited_weights[weight_name] = (
                weight.detach().cpu()
                + update.detach().to(device="cpu", dtype=weight.dtype)
            )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=weights_path.parent,
            prefix=f".{weights_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(edited_weights, temporary_path)
        os.replace(temporary_path, weights_path)
        print(
            f"Saved evaluation-compatible epoch checkpoint: "
            f"{weights_path}; use load_name={epoch_load_name}"
        )
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def get_hidden_size(model) -> int:
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    if hasattr(model.config, "n_embd"):
        return model.config.n_embd
    raise ValueError("Cannot determine model hidden size for m_i cache")


def get_mi_cache_entry(cfg, model, tok, request, z_layer):
    """Return a stable per-request cache path and its validation metadata."""
    metadata = {
        "cache_format_version": 2,
        "model_name": cfg.llms.alias,
        "model_class": type(model).__name__,
        "tokenizer_name": getattr(tok, "name_or_path", type(tok).__name__),
        "z_layer": int(z_layer),
        "prompt": request["prompt"],
        "subject": request["subject"],
        "target_new": request["target_new"],
        "fact_token": cfg.llms.fact_token,
        "v_num_grad_steps": int(cfg.llms.v_num_grad_steps),
        "v_lr": float(cfg.llms.v_lr),
        "v_loss_layer": int(cfg.llms.v_loss_layer),
        "v_weight_decay": float(cfg.llms.v_weight_decay),
        "kl_factor": float(cfg.llms.kl_factor),
        "clamp_norm_factor": float(cfg.llms.clamp_norm_factor),
        # get_context_templates currently uses one plain template and five
        # sampled length-10 prefixes. The generated strings are saved as
        # metadata but deliberately excluded from the stable cache key.
        "context_template_scheme": "plain-plus-generated-10x5-v1",
    }
    if not cfg.algs.mi_cache_enabled:
        return None, metadata

    # Preserve the version-1 path so its generated context templates can be
    # reused during migration. The version-1 payload itself is rejected by
    # ``load_cached_mi`` because it has no delta and is then overwritten
    # atomically in format version 2.
    cache_key_metadata = {**metadata, "cache_format_version": 1}
    serialized = json.dumps(
        cache_key_metadata, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    cache_key = hashlib.sha256(serialized).hexdigest()
    model_dir = cfg.llms.alias.replace("/", "-")
    cache_path = (
        Path(cfg.cache_dir)
        / "memit_targets"
        / model_dir
        / f"layer-{z_layer}"
        / f"{cache_key}.pt"
    )
    return cache_path, metadata


def load_cached_mi(
    cache_path,
    expected_metadata,
    hidden_size,
    force_recompute=False,
):
    """Load and validate cached ``(target, delta)`` for one edit."""
    if cache_path is None or force_recompute or not cache_path.is_file():
        return None

    try:
        cached = torch.load(cache_path, map_location="cpu")
        target = cached.get("target")
        delta = cached.get("delta")
        metadata = cached["metadata"]
        if any(
            metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            return None
        if (
            not isinstance(target, torch.Tensor)
            or target.ndim != 1
            or target.shape[0] != hidden_size
            or not torch.isfinite(target).all()
            or not isinstance(delta, torch.Tensor)
            or delta.ndim != 1
            or delta.shape[0] != hidden_size
            or not torch.isfinite(delta).all()
        ):
            print(f"Ignoring invalid m_i cache file: {cache_path}")
            return None
        print(f"Loaded cached m_i: {cache_path}")
        return (
            target.detach().to(dtype=torch.float32),
            delta.detach().to(dtype=torch.float32),
        )
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Ignoring unreadable m_i cache {cache_path}: {error}")
        return None


def load_cached_context_templates(cache_entries):
    """Reuse the exact templates saved with an existing m_i cache entry."""
    for cache_path, expected_metadata in cache_entries:
        if cache_path is None or not cache_path.is_file():
            continue
        try:
            cached = torch.load(cache_path, map_location="cpu")
            metadata = cached["metadata"]
            # Context strings are independent of the target/delta cache
            # payload, so a version-1 entry may supply them during migration.
            if any(
                metadata.get(key) != value
                for key, value in expected_metadata.items()
                if key != "cache_format_version"
            ):
                continue
            context_templates = metadata.get("context_templates")
            if (
                isinstance(context_templates, list)
                and context_templates
                and all(
                    isinstance(group, list)
                    and group
                    and all(isinstance(template, str) for template in group)
                    for group in context_templates
                )
            ):
                print(
                    f"Reusing context templates from m_i cache: "
                    f"{cache_path}"
                )
                return context_templates
        except (
            EOFError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            continue
    return None


def get_stable_joint_context_templates(cfg, model, tok, cache_entries):
    """Load or atomically cache the stochastic joint context templates."""
    context_templates = load_cached_context_templates(cache_entries)
    if context_templates is not None:
        return context_templates

    metadata = {
        "cache_format_version": 1,
        "model_name": cfg.llms.alias,
        "model_class": type(model).__name__,
        "tokenizer_name": getattr(tok, "name_or_path", type(tok).__name__),
        "context_template_scheme": "plain-plus-generated-10x5-v1",
    }
    cache_key = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    cache_path = (
        Path(cfg.cache_dir)
        / "memit_context_templates"
        / cfg.llms.alias.replace("/", "-")
        / f"{cache_key}.pt"
    )
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, map_location="cpu")
            cached_templates = cached["context_templates"]
            if (
                cached.get("metadata") == metadata
                and valid_context_templates(cached_templates)
            ):
                print(f"Loaded joint context templates: {cache_path}")
                return cached_templates
        except (
            EOFError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            print(
                f"Ignoring unreadable joint context-template cache "
                f"{cache_path}: {error}"
            )

    context_templates = get_context_templates(model, tok)
    if not valid_context_templates(context_templates):
        raise ValueError("Generated joint context templates are invalid")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(
            {
                "metadata": metadata,
                "context_templates": context_templates,
            },
            temporary_path,
        )
        os.replace(temporary_path, cache_path)
        print(f"Saved joint context templates: {cache_path}")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return context_templates


def valid_context_templates(context_templates):
    """Return whether a nested context-template list is well formed."""
    return (
        isinstance(context_templates, list)
        and bool(context_templates)
        and all(
            isinstance(group, list)
            and bool(group)
            and all(isinstance(template, str) for template in group)
            for group in context_templates
        )
    )


def save_cached_mi(cache_path, target, delta, metadata):
    """Atomically save one canonical target and its edit displacement."""
    if cache_path is None:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(
            {
                "target": target.detach().to(
                    device="cpu", dtype=torch.float32
                ),
                "delta": delta.detach().to(
                    device="cpu", dtype=torch.float32
                ),
                "metadata": metadata,
            },
            temporary_path,
        )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_joint_training_samples(
    requests, context_templates, use_context_templates=True
):
    """Expand every edit into canonical and context-prefixed samples."""
    if use_context_templates:
        templates = [
            context
            for context_group in context_templates
            for context in context_group
        ]
    else:
        templates = ["{}"]
    if not templates:
        raise ValueError("Joint context training requires at least one template")

    samples = []
    for request_index, request in enumerate(requests):
        for context_index, context in enumerate(templates):
            samples.append(
                {
                    "request_index": request_index,
                    "context_index": context_index,
                    "is_original_prompt": context == "{}",
                    "prompt": context.format(request["prompt"]),
                    "subject": request["subject"],
                    "target_new": request["target_new"],
                }
            )
    return samples


def build_joint_sample_loss_weights(
    training_samples, num_edits, original_prompt_loss_share=None
):
    """Return sample weights that sum to one independently for each edit.

    With ``original_prompt_loss_share=None``, every expanded prompt belonging
    to an edit receives equal weight, exactly matching the previous objective.
    Otherwise the unique canonical/original prompt receives the configured
    share and all context-prefixed prompts split the remainder equally.
    """
    if num_edits <= 0:
        raise ValueError("num_edits must be positive")
    if not training_samples:
        raise ValueError("Joint training samples must not be empty")
    if original_prompt_loss_share is not None:
        original_prompt_loss_share = float(original_prompt_loss_share)
        if not 0.0 < original_prompt_loss_share < 1.0:
            raise ValueError(
                "original_prompt_loss_share must be strictly between 0 and 1"
            )

    samples_by_edit = [[] for _ in range(num_edits)]
    for sample_index, sample in enumerate(training_samples):
        request_index = int(sample["request_index"])
        if request_index < 0 or request_index >= num_edits:
            raise ValueError(
                f"Invalid request_index {request_index} in training samples"
            )
        samples_by_edit[request_index].append(sample_index)

    weights = [0.0] * len(training_samples)
    for request_index, sample_indices in enumerate(samples_by_edit):
        if not sample_indices:
            raise ValueError(
                f"Edit {request_index} has no joint training samples"
            )
        if original_prompt_loss_share is None or len(sample_indices) == 1:
            uniform_weight = 1.0 / len(sample_indices)
            for sample_index in sample_indices:
                weights[sample_index] = uniform_weight
            continue

        original_indices = [
            sample_index
            for sample_index in sample_indices
            if training_samples[sample_index]["is_original_prompt"]
        ]
        if len(original_indices) != 1:
            raise ValueError(
                "Weighted context training requires exactly one '{}' "
                f"original template per edit; edit {request_index} has "
                f"{len(original_indices)}"
            )
        context_indices = [
            sample_index
            for sample_index in sample_indices
            if sample_index != original_indices[0]
        ]
        context_weight = (
            1.0 - original_prompt_loss_share
        ) / len(context_indices)
        weights[original_indices[0]] = original_prompt_loss_share
        for sample_index in context_indices:
            weights[sample_index] = context_weight

    return weights


def prepare_joint_batch(tok, requests, fact_token_strategy, device):
    """Tokenize one shuffled micro-batch and locate its fact positions."""
    prompt_templates = [request["prompt"] for request in requests]
    subjects = [request["subject"] for request in requests]
    prompts = [
        prompt.format(subject)
        for prompt, subject in zip(prompt_templates, subjects)
    ]
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt,
            subject,
            tok,
            fact_token_strategy,
            verbose=False,
        )
        for prompt, subject in zip(prompt_templates, subjects)
    ]
    model_inputs = tok(
        prompts, return_tensors="pt", padding=True
    ).to(device)
    return model_inputs, lookup_idxs


def prepare_joint_ce_batch(tok, samples, device):
    """Build teacher-forcing inputs and target positions for joint CE.

    As in ``compute_z``, a target with token IDs ``y_1 ... y_T`` is
    represented by appending ``y_1 ... y_(T-1)`` to the rewrite prompt. The
    final ``T`` non-padding hidden positions therefore predict all ``T``
    target tokens. Positions are derived from the attention mask, so both
    left- and right-padding tokenizers are supported.
    """
    if not samples:
        raise ValueError("Joint CE requires a non-empty micro-batch")

    target_ids = []
    teacher_forcing_prompts = []
    removable_prefix_ids = {
        token_id
        for token_id in (tok.bos_token_id, tok.unk_token_id)
        if token_id is not None
    }
    for sample in samples:
        ids = tok(
            sample["target_new"], return_tensors="pt"
        )["input_ids"][0]
        if ids.numel() and ids[0].item() in removable_prefix_ids:
            ids = ids[1:]
        if not ids.numel():
            raise ValueError(
                f"Target tokenization is empty for request "
                f"{sample['request_index']}"
            )
        target_ids.append(ids.detach().to(device=device))
        target_prefix = tok.decode(ids[:-1])
        teacher_forcing_prompts.append(
            (sample["prompt"] + target_prefix).format(sample["subject"])
        )

    model_inputs = tok(
        teacher_forcing_prompts, return_tensors="pt", padding=True
    ).to(device)
    attention_mask = model_inputs["attention_mask"]
    target_positions = []
    for sample_index, ids in enumerate(target_ids):
        non_padding_positions = torch.nonzero(
            attention_mask[sample_index], as_tuple=False
        ).flatten()
        if non_padding_positions.numel() < ids.numel():
            raise ValueError(
                "Teacher-forcing input is shorter than its target at "
                f"micro-batch row {sample_index}"
            )
        target_positions.append(non_padding_positions[-ids.numel() :])
    return model_inputs, target_positions, target_ids


def get_joint_ce_losses(
    model,
    model_inputs,
    target_positions,
    target_ids,
    loss_layer,
    layer_module_tmp,
    ln_f_module,
    lm_head_module,
):
    """Return differentiable per-sample, per-token-mean CE losses."""
    layer_name = layer_module_tmp.format(loss_layer)
    with nethook.Trace(
        module=model,
        layer=layer_name,
        retain_output=True,
        stop=True,
    ) as trace:
        model(**model_inputs, use_cache=False)

    output = trace.output[0] if isinstance(trace.output, tuple) else trace.output
    expected_batch, expected_sequence = model_inputs["input_ids"].shape
    if output.ndim != 3:
        raise ValueError(
            f"Loss-layer output must be rank 3, got {tuple(output.shape)}"
        )
    if output.shape[:2] == (expected_batch, expected_sequence):
        batch_first_output = output
    elif output.shape[:2] == (expected_sequence, expected_batch):
        batch_first_output = output.transpose(0, 1)
    else:
        raise ValueError(
            f"Loss-layer output shape {tuple(output.shape)} is incompatible "
            f"with token shape {(expected_batch, expected_sequence)}"
        )

    ln_f = nethook.get_module(model, ln_f_module)
    lm_head = nethook.get_module(model, lm_head_module)
    target_lengths = [ids.numel() for ids in target_ids]
    selected_hidden = torch.cat(
        [
            batch_first_output[
                sample_index,
                positions.to(device=batch_first_output.device),
                :,
            ]
            for sample_index, positions in enumerate(target_positions)
        ],
        dim=0,
    )
    logits = lm_head(ln_f(selected_hidden))
    flat_target_ids = torch.cat(
        [ids.to(device=logits.device) for ids in target_ids], dim=0
    )
    token_losses = F.cross_entropy(
        logits.float(), flat_target_ids, reduction="none"
    )
    return torch.stack(
        [losses.mean() for losses in token_losses.split(target_lengths)]
    )


def make_update_hook(update: torch.Tensor, module_name: str):
    """Inject a differentiable full-rank update into a rewrite module."""

    def inject_update(module, inputs, output):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError(f"{module_name} does not have a tensor input")

        layer_input = inputs[0]
        layer_output = output[0] if isinstance(output, tuple) else output
        update_for_forward = update.to(dtype=layer_input.dtype)

        if (
            update.shape[1] == layer_input.shape[-1]
            and update.shape[0] == layer_output.shape[-1]
        ):
            # torch.nn.Linear stores [out_features, in_features].
            correction = F.linear(layer_input, update_for_forward)
        elif (
            update.shape[0] == layer_input.shape[-1]
            and update.shape[1] == layer_output.shape[-1]
        ):
            # GPT-2 Conv1D stores its weight transposed as [in, out].
            correction = layer_input @ update_for_forward
        else:
            raise ValueError(
                f"Update shape {tuple(update.shape)} does not match input "
                f"{tuple(layer_input.shape)} and output "
                f"{tuple(layer_output.shape)} for {module_name}"
            )

        updated_output = layer_output + correction
        if isinstance(output, tuple):
            return (updated_output, *output[1:])
        return updated_output

    return inject_update


def get_joint_target_vectors(
    model,
    model_inputs,
    lookup_idxs,
    z_layer,
    layer_module_tmp,
):
    """Return differentiable z-layer vectors at the edited fact positions."""
    layer_name = layer_module_tmp.format(z_layer)
    with nethook.Trace(
        module=model,
        layer=layer_name,
        retain_output=True,
        stop=True,
    ) as trace:
        model(**model_inputs)

    output = trace.output[0] if isinstance(trace.output, tuple) else trace.output
    if output.shape[0] != len(lookup_idxs):
        output = output.transpose(0, 1)
    return torch.stack(
        [output[i, lookup_idx, :] for i, lookup_idx in enumerate(lookup_idxs)],
        dim=0,
    )


def compute_joint_base_context_vectors(
    model,
    tok,
    training_samples,
    fact_token_strategy,
    z_layer,
    layer_module_tmp,
    device,
    batch_size,
):
    """Precompute frozen-model ``h_i,c`` for every expanded sample."""
    if batch_size <= 0:
        raise ValueError("Joint context baseline batch size must be positive")

    num_samples = len(training_samples)
    base_context_ms = torch.empty(
        (num_samples, get_hidden_size(model)),
        device="cpu",
        dtype=torch.float32,
    )
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(
        f"Precomputing {num_samples} frozen context representation(s) "
        f"in {num_batches} batch(es)"
    )

    with torch.no_grad():
        for batch_number, batch_indices in enumerate(
            chunks(list(range(num_samples)), batch_size), start=1
        ):
            batch_samples = [
                training_samples[index] for index in batch_indices
            ]
            model_inputs, lookup_idxs = prepare_joint_batch(
                tok=tok,
                requests=batch_samples,
                fact_token_strategy=fact_token_strategy,
                device=device,
            )
            batch_base_ms = get_joint_target_vectors(
                model=model,
                model_inputs=model_inputs,
                lookup_idxs=lookup_idxs,
                z_layer=z_layer,
                layer_module_tmp=layer_module_tmp,
            )
            if not torch.isfinite(batch_base_ms).all():
                raise FloatingPointError(
                    "Non-finite frozen context representation detected"
                )
            base_context_ms[batch_indices] = batch_base_ms.detach().to(
                device="cpu", dtype=torch.float32
            )
            if batch_number == 1 or batch_number == num_batches:
                print(
                    f"frozen context representations: "
                    f"batch {batch_number}/{num_batches}"
                )

    return base_context_ms


def trace_preservation_loss(
    update: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    """Compute tr(Delta W C Delta W.T) for either weight convention."""
    if update.shape[1] == covariance.shape[0]:
        # Linear convention: Delta W is [out, in].
        return torch.sum((update @ covariance) * update)
    if update.shape[0] == covariance.shape[0]:
        # GPT-2 Conv1D convention: stored update is [in, out].
        return torch.sum((covariance @ update) * update)
    raise ValueError(
        f"Covariance shape {tuple(covariance.shape)} is incompatible with "
        f"update shape {tuple(update.shape)}"
    )


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
