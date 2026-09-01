from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.repr_tools  import *
from util import nethook
from omegaconf import DictConfig
from .context import get_edit_prefix,get_edit_target


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    cfg: DictConfig,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the canonical value target and its context-trained displacement.

    ``target`` is the canonical prompt representation plus ``delta``.  The
    same ``delta`` is injected into every rewriting context while it is
    optimized, so callers that train on those contexts can construct the
    context-specific target ``h_i,c + delta_i``.
    """
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{cfg.llms.lm_head_module}").weight.T,
        nethook.get_module(model, cfg.llms.ln_f_module),
    )
    lm_b = nethook.get_parameter(model, f"{cfg.llms.lm_head_module}.bias")
    if lm_b is None:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    # print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"], return_tensors="pt").to(device)[
        "input_ids"
    ][0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    if cfg.negetive_prompt_test:
        rewriting_prompts, kl_prompts = [
            context.format(request["negetive_prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ], ["{} is a".format(request["subject"])]
        all_prompts = rewriting_prompts + kl_prompts
        input_tok = tok(
            all_prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)
    else:
        # Compile list of rewriting and KL x/y pairs
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ], ["{} is a"]
        all_prompts = rewriting_prompts + kl_prompts

        input_tok = tok(
            [prompt.format(request["subject"]) for prompt in all_prompts],
            return_tensors="pt",
            padding=True,
        ).to(device)

    target_idxs_start = []
    target_idxs_end = []
    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
        target_idxs_start.append(ex_len - len(target_ids))
        target_idxs_end.append(ex_len)

    for i in range(len(kl_prompts)):
        ex_len = input_tok["attention_mask"][i+len(rewriting_prompts)].sum()
        target_idxs_start.append(ex_len - 1)
        target_idxs_end.append(ex_len)

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, cfg.llms.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    if cfg.lti:
        context_kl_log_probs_tar, context_kl_log_probs_mid = get_edit_target(
            model=model,
            tok=tok,
            request=request,
            cfg=cfg.llms,
            context_prefix=get_edit_prefix(
                request=request
            ),
            rewriting_prompts=rewriting_prompts,
            loc_prompts=kl_prompts,
            target_idxs_start=target_idxs_start,
            target_idxs_end=target_idxs_end,
            lookup_idxs=lookup_idxs,
        )

    trace_layers_mid = [
        cfg.llms.layer_module_tmp.format(layer) for layer in cfg.llms.midlayers
    ]
    trace_layers_final = [
        cfg.llms.layer_module_tmp.format(layer+1) for layer in cfg.llms.midlayers
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(cfg.llms.v_loss_layer, layer)
    # print(f"Rewrite layer is {layer}")
    # print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=device)
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=device)
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None
    midlayer_vec, subject_vec, last_vec = None, None, None
    out_tuple=None
    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        nonlocal out_tuple
        nonlocal target_init, midlayer_vec, subject_vec, last_vec
        out_tuple=type(cur_out)==tuple
        if not out_tuple:#正常情况下，cur_out[0]是该层输出，cur_out[1]是该层注意力权重。
            cur_out = (cur_out,)#但是有一些大模型例如llama3-8b不输出注意力权重，只有该层输出，是一个tensor。
        if cur_layer in trace_layers_mid:
            tmp_repr = cur_out[0]
            subject_vec = torch.stack(
                [
                    tmp_repr[i, idx, :]
                    for i, idx in enumerate(lookup_idxs)
                ],
                dim=0
            ).unsqueeze(1)

        if cur_layer in trace_layers_final:
            tmp_repr = cur_out[0]
            last_vec = torch.cat(
                [
                    tmp_repr[i, idxst:idxed, :]
                    for i, (idxst, idxed) in enumerate(zip(target_idxs_start, target_idxs_end))
                ],
                dim=0
            ).unsqueeze(1)
        if cfg.llms.constr_pos == "subject":
            midlayer_vec = subject_vec
        elif cfg.llms.constr_pos == "last":
            midlayer_vec = last_vec
        elif cfg.llms.constr_pos == "all":
            if last_vec is not None:
                midlayer_vec = torch.cat([subject_vec, last_vec], dim=0)
        else:
            raise ValueError(
                f"Unsupported constr_pos: {cfg.llms.constr_pos}")
        
        if cur_layer == cfg.llms.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=len(cur_out[0]):#batch size在seqlen后的情况，一般不是。
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta

        if not out_tuple:
            cur_out = cur_out[0]
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=cfg.llms.v_lr)
    nethook.set_requires_grad(False, model)

    trace_layers = [
        cfg.llms.layer_module_tmp.format(loss_layer),
        cfg.llms.layer_module_tmp.format(layer),
    ]
    if cfg.lti:
        trace_layers += trace_layers_mid
        trace_layers += trace_layers_final
    # Execute optimization
    for it in range(cfg.llms.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=trace_layers,
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits#[bs,seqlen,vocab_size]

            if cfg.lti:
                kl_logits = torch.cat(
                    [
                        logits[i, idxst:idxed, :]
                        for i, (idxst, idxed) in enumerate(zip(target_idxs_start, target_idxs_end))
                    ],
                    dim=0,
                )
            else:
                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_prompts), idx, :]
                        for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()
        if cfg.lti:
            midlayer_logits = ln_f(
                midlayer_vec) @ lm_w.to(midlayer_vec.device) + lm_b.to(midlayer_vec.device)

            # kl_logits=torch.cat([midlayer_logits.squeeze(1),kl_logits],dim=0)
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            mid_log_probs = torch.nn.functional.log_softmax(
                midlayer_logits.squeeze(1), dim=1)

            # Compute loss on rewriting targets
            log_probs = torch.log_softmax(logits, dim=2)

            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100,
                            rewriting_targets, 0).unsqueeze(2),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()*cfg.llms.nll_factor
            # nll_loss = nll_loss_each.mean()

            kl_loss = cfg.llms.last_kl_factor * torch.nn.functional.kl_div(
                context_kl_log_probs_tar, kl_log_probs, log_target=True, reduction="batchmean"
            )

            mid_kl_loss = cfg.llms.mid_kl_factor * torch.nn.functional.kl_div(
                context_kl_log_probs_mid, mid_log_probs, log_target=True, reduction="batchmean"
            )

            weight_decay = cfg.llms.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )

            device = torch.device("cuda:0")
            loss = kl_loss.to(device) + mid_kl_loss.to(device) + \
                weight_decay.to(device) + nll_loss.to(device)
        else:
            # Compute loss on rewriting targets
            if not out_tuple:
                output=tr[cfg.llms.layer_module_tmp.format(loss_layer)].output
            else:
                output=tr[cfg.llms.layer_module_tmp.format(loss_layer)].output[0]
            if output.shape[1]!=rewriting_targets.shape[1]:
                output=torch.transpose(output, 0, 1)
            full_repr = output[:len(rewriting_prompts)]

            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = cfg.llms.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            weight_decay = cfg.llms.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            # weight_decay = cfg.llms.v_weight_decay * torch.norm(delta) ** 2
            loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == cfg.llms.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = cfg.llms.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target, delta


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        l_input, l_output = get_reprs_at_word_tokens(
            track="both", subtoken=None, **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    # sentence = prompt.format(subject)
    # if verbose:
    #     print(
    #         f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
    #         tok.decode(tok(sentence)["input_ids"][ret]),
    #     )

    return ret
