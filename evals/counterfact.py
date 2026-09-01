
"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.generate import generate_fast
from util.perplexity import perplexity
from omegaconf import DictConfig
from evals.lweval import lw_eval, true_false_probs, test_inversion_tf,test_unlearning_ab
from evals.lbqeval import lbq_eval

target_true_logits = []
target_new_logits = []

def compute_probs_correct(cfg,model,tok,prob_prompts,which_correct,
                          target_new,target_true,keys):
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        cfg,
        model,
        tok,
        list(chain(*prob_prompts)),#这个其实就是扁平化，[[1,2],[3]]变成[1,2,3]
        list(chain(*which_correct)),
        target_new,
        target_true,
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1]: cutoffs[i]] for i in range(1, len(cutoffs))
    ]

    ret_probs=summarize(ret_probs, which_correct)
    # Structure the restuls as a dictionary.
    ret = {
              f"{key}_probs": ret_probs[i]
              for i, key in enumerate(keys)
          } | {
              f"{key}_correct": ret_corrects[i]
              for i, key in enumerate(keys)
          }
    metrics=replace_tf_with_acc(ret)
    return metrics


def replace_tf_with_acc(metrics):
    for key in metrics:
        metrics[key]=np.mean(metrics[key]).item()
    return metrics

def summarize(kinds,labels):
    all_preds=[]
    for i in range(len(kinds)):
        kind=kinds[i]
        preds=[]
        for j in range(len(kind)):
            result=kind[j]
            label=labels[i][j]
            if label==0:
                eval_pred=result["target_new"]<result["target_true"]
            else:
                eval_pred=result["target_new"]>result["target_true"]
            preds.append(eval_pred)
        all_preds.append(preds)
    return all_preds

def eval_counterfact(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record

    :return: Dictionary containing rewriting metrics
    """
    if cfg.unlearning_ab:
        return test_unlearning_ab(record,cfg,model,tok)
    if cfg.tf_props:
        true_false_probs(record,cfg,model,tok)
        return
    if cfg.inversion_tf:
        return test_inversion_tf(record,cfg,model,tok)

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record[x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["prompt"].format(subject)]
    if cfg.negetive_prompt_test:
        rewrite_prompts = [record["negetive_prompt"]]
    if cfg.subject_output_relation_test1:
        rewrite_prompts = [record["subject"] + " is"]
    if cfg.subject_output_relation_test2:
        import random
        random_prefixs = ["the best friend of ", "the hometown of "]
        random_prefix = random_prefixs[random.randint(0,1)]
        rewrite_prompts = [random_prefix + record["subject"] + " is"]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    keys=["rewrite_prompts","paraphrase_prompts","neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    if cfg.neighborhood_logits:
        prob_prompts = [
            # rewrite_prompts,
            # paraphrase_prompts,
            neighborhood_prompts,
        ]
        keys=["neighborhood_prompts"]
    else:
        prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
        ]
    if cfg.target_true_test:
        which_correct = [
            [1 for _ in range(len(rewrite_prompts))],
            [1 for _ in range(len(paraphrase_prompts))],
            [1 for _ in range(len(neighborhood_prompts))],
        ]
    elif cfg.neighborhood_logits:
        which_correct = [
            [1 for _ in range(len(neighborhood_prompts))],
        ]
    else:
        which_correct = [
            [0 for _ in range(len(rewrite_prompts))],
            [0 for _ in range(len(paraphrase_prompts))],
            [1 for _ in range(len(neighborhood_prompts))],
        ]
    if cfg.negetive_prompt_test:
        keys[0] = "rewrite_negetive_prompts"
    metrics=compute_probs_correct(cfg,model,tok,prob_prompts,which_correct,target_new,target_true
                                   ,keys=keys)
    if cfg.lbq_eval:
        metrics_lbq = lbq_eval(record, cfg, model, tok,q_test=True)
        metrics = metrics | metrics_lbq
    if cfg.lw_eval:
        metrics_lw=lw_eval(record,cfg,model,tok)
        metrics=metrics | metrics_lw
    return metrics


def test_batch_prediction(
    cfg,
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """
    global target_true_logits
    global target_new_logits
    device=next(model.parameters()).device
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"#注意到这里是加了空格的，就是输入输出拼接。
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)

    a_tok, b_tok = (tok(f" {n}",add_special_tokens=False)["input_ids"] for n in [target_new, target_true])
    #有些模型会在开头加上特殊token，所以上面这样进行处理。
    # if 'llama' in model.config._name_or_path.lower():
    #     # a_tok = a_tok[1:]
    #     # b_tok = b_tok[1:]
    #     prefix_lens = [lengths -1 for lengths in prefix_lens]

    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    with torch.no_grad():
        logits = model(**prompt_tok).logits

    # if 'llama' in model.config._name_or_path.lower():
    #     logits = logits[:, 1:, :]

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        cur_logits_list = []

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
            if cfg.neighborhood_logits and j == 0:
                cur_logits_list.append(
                    logits[i, prefix_lens[i // 2] + j - 1, :].cpu()
                )
        if i % 2 == 0 and cfg.neighborhood_logits:
            target_new_logits.append(
                cur_logits_list
            )
        elif i % 2 == 1 and cfg.neighborhood_logits:
            target_true_logits.append(
                cur_logits_list
            )
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct





