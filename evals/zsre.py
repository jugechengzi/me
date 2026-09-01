"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from evals.lweval import lw_eval, true_false_probs
from evals.lbqeval import lbq_eval

target_true_logits = []
target_new_logits = []

def get_prompt_target_pairs(tok,model,prompt,target):#这里有待检查一下。
    prompts, targets=[],[]
    target_tok = tok(" " + target,add_special_tokens=False)["input_ids"]#编辑目标，这涉及到多个token的精确匹配。
    # if 'llama' in model.config._name_or_path.lower():
    #     target_tok = target_tok[1:]
    #然后又要恢复回去，decode。
    for i in range(len(target_tok)):
        prompts.append(prompt + tok.decode(target_tok[:i]))
        targets.append(tok.decode(target_tok[i]))
    return prompts,targets

def eval_zsre(
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
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if cfg.tf_props:
        true_false_probs(record,cfg,model,tok)
        return

    # First, unpack rewrite evaluation record.
    subject, target_new, neighborhood_answer = (
        record[x] for x in ["subject", "target_new","neighborhood_prompts_answers"]
    )
    rewrite_prompts = [record["prompt"].format(subject)]
    if cfg.negetive_prompt_test:
        rewrite_prompts = [record["negetive_prompt"]]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    rewrite_prompts,rewrite_targets=get_prompt_target_pairs(tok,model,rewrite_prompts[0],target_new)
    paraphrase_prompts,paraphrase_targets=get_prompt_target_pairs(tok,model,paraphrase_prompts[0],target_new)
    neighborhood_prompts,neighborhood_targets=get_prompt_target_pairs(tok,model,neighborhood_prompts[0],neighborhood_answer[0])

    if cfg.neighborhood_logits:
        prompts=neighborhood_prompts
        targets=neighborhood_targets
        keys=["neighborhood_prompts_correct"]
        strict_keys=["neighborhood_strict_correct"]
        cut_offs=np.cumsum([0,len(neighborhood_targets)])
    else:
        prompts=rewrite_prompts+paraphrase_prompts+neighborhood_prompts
        targets=rewrite_targets+paraphrase_targets+neighborhood_targets
        keys=["rewrite_prompts_correct","paraphrase_prompts_correct","neighborhood_prompts_correct"]
        strict_keys=["rewrite_strict_correct","paraphrase_strict_correct","neighborhood_strict_correct"]
        cut_offs=np.cumsum([0,len(rewrite_targets),len(paraphrase_targets),len(neighborhood_targets)])
    metrics_detail = test_batch_prediction_acc(model, tok, prompts, targets)#对于llama3-8b,float32,zsre，需要39/40GB的评估，注意。
    if cfg.negetive_prompt_test:
        keys[0] = "rewrite_negetive_prompts"
    metrics={}
    for i in range(len(keys)):
        start=cut_offs[i]
        end=cut_offs[i+1]
        metrics[keys[i]]=np.mean(metrics_detail[start:end]).item()
        metrics[strict_keys[i]]= np.min(metrics_detail[start:end]).astype(np.int32).item()

    if cfg.lbq_eval:
        metrics_lbq = lbq_eval(record, cfg, model, tok,q_test=False)
        metrics = metrics | metrics_lbq

    if cfg.lw_eval:
        metrics_lw=lw_eval(record,cfg,model,tok)
        metrics=metrics | metrics_lw

    return metrics


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    global target_true_logits
    device=next(model.parameters()).device
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**prompt_tok).logits#[bs,seqlen,vocab_size]
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1#[bs]最后一个有效token的位置。
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)#[bs,1,vocab_size]
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)#[bs,vocab_size]取出下一个单词的预测概率
        ans = torch.argmax(gathered, dim=1)#[bs]查看预测概率最大的那个token。

        # Locality compares only the full-vocabulary distribution that predicts
        # the first token of the original neighborhood answer.
        target_true_logits.append(gathered[:1].cpu())
        correct_id = tok(target, padding=True, return_tensors="pt",
                         add_special_tokens=False).to(device)["input_ids"]#这个和ans对应，ans是预测的，而correct_id是正确的。
        # Temporary hack to deal with foreign characters.
        # if 'llama' in model.config._name_or_path.lower():
        #     correct_id = correct_id[:, 1].squeeze()#第0个是空格。
        # else:
        correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()
