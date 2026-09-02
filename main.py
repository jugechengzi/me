import os
import torch
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from algs.wise import apply_wise_to_model
from evals.evaluation import eval_one_edit
from evals.neighborhood_locality import compute_neighborhood_locality
from algs.alphaedit import apply_alphaedit_to_model
from algs.memit import (
    apply_memit_lora_to_model,
    apply_memit_ori_to_model,
    apply_memit_to_model,
)
from algs.rome import apply_rome_to_model
from algs.emmet import apply_emmet_to_model
from algs.rect import apply_rect_to_model
from algs.namet import apply_namet_to_model
from algs.prune import apply_prune_to_model
from algs.pmet import apply_pmet_to_model
from algs.adaedit import apply_adaedit_to_model
from algs.wise import apply_wise_to_model
from algs.ft import apply_ft_to_model
from algs.rledit import apply_rledit_to_model
from datetime import datetime

from load import load_model,load_data,save_model
from util.utility import ensure_file_directory
import numpy as np
import time

ALG_DICT = {
    "alphaedit":  apply_alphaedit_to_model,
    "memit": apply_memit_to_model,
    "memit_lora": apply_memit_lora_to_model,
    "memit_ori": apply_memit_ori_to_model,
    "rome": apply_rome_to_model,
    "emmet": apply_emmet_to_model,
    "namet": apply_namet_to_model,
    "namet_nok0": apply_namet_to_model,
    "rect": apply_rect_to_model,
    "rect_nok0": apply_rect_to_model,
    "prune": apply_prune_to_model,
    "prune_nok0": apply_prune_to_model,
    "pmet": apply_pmet_to_model,
    "pmet_nok0": apply_pmet_to_model,
    "adaedit": apply_adaedit_to_model,
    "adaedit_nok0": apply_adaedit_to_model,
    "wise": apply_wise_to_model,
    "ft": apply_ft_to_model,
    "rledit": apply_rledit_to_model,
}


def set_random_seed(seed=42):
    torch.manual_seed(seed)  # torch的cpu随机性
    torch.cuda.manual_seed_all(seed)  # torch的gpu随机性
    torch.backends.cudnn.benchmark = False  # 保证gpu每次都选择相同的算法，但是不保证该算法是deterministic的。
    torch.backends.cudnn.deterministic = True  # 紧接着上面，保证算法是deterministic的。
    np.random.seed(seed)  # np的随机性。
    random.seed(seed)  # python的随机性。
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置python哈希种子

import hydra
from omegaconf import DictConfig, OmegaConf


def print_dict(dict):
    for key, value in dict.items():
        print(key, value)

def eval_algo(cfg,model,tok,data):
    all_metrics = {}
    # filtered_data = []
    #一个一个评估。
    for edit in data:#不支持不同的样本有不同的key评估。
        metrics=eval_one_edit(cfg,model,tok,edit)
        if metrics is None:
            continue
        # if metrics.get('rewrite_prompts_correct', 0) == 1:
        #     filtered_data.append(edit)
        if len(all_metrics)==0:
            for key,value in metrics.items():
                all_metrics[key]=[value]
        else:
            for key,value in metrics.items():
                all_metrics[key].append(value)
    # #保存过滤后的数据集。
    # file=cfg.data_dir+"/"+f"{cfg.data}_{cfg.llms.name.replace('/','-')}"+"_filtered.json"
    # ensure_file_directory(file)
    # with open(file, "w", encoding="utf-8") as f:
    #     json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    #对所有指标进行总结。
    avg_metrics={}
    for key,value in all_metrics.items():
        avg_metrics[key]=np.round(np.mean(value),3).item()
    return avg_metrics


def get_neighborhood_logit_buffers(data_name):
    if "zsre_mend_eval" in data_name:
        from evals.zsre import target_true_logits, target_new_logits
    elif "counterfact" in data_name:
        from evals.counterfact import target_true_logits, target_new_logits
    else:
        raise ValueError(
            "Synchronous neighborhood KL/top-k evaluation currently supports "
            "CounterFact and ZSRE datasets only."
        )
    return target_true_logits, target_new_logits

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_random_seed(cfg.seed)
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    print("Start Loading model")
    model_name_or_path=cfg.llms.name
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=cfg.model_dtype,trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
    print("Loading model successfully")
    tok.pad_token = tok.eos_token

    apply_algo = ALG_DICT[cfg.algs.name]
    data=load_data(cfg)

    if cfg.test_only:
        from evals.lweval import predicts
        from evals.lweval import abcd_orders
        pre_eval_name = cfg.pre_eval_name or cfg.save_name
        pre_results_file = cfg.results_dir + "/{}/{}/{}".format(
            cfg.data,
            cfg.llms.alias.replace("/", "-"),
            pre_eval_name,
        )
        pre_logits_file = pre_results_file + "_neighborhood_target_logits.pt"
        ensure_file_directory(pre_results_file)
        need_pre_evaluation = not os.path.exists(pre_results_file)
        if cfg.neighborhood_logits and not os.path.exists(pre_logits_file):
            need_pre_evaluation = True
            print("Original neighborhood logits are missing; recomputing original evaluation.")
        if need_pre_evaluation:
            start_time = time.time()
            print("Start Evaluating the Original Model")
            if cfg.neighborhood_logits:
                target_true_logits, target_new_logits = get_neighborhood_logit_buffers(
                    cfg.data
                )
                target_true_logits.clear()
                target_new_logits.clear()
            pre_metrics=eval_algo(cfg, model, tok, data)#不是每一次都有必要进行这个。
            end_time = time.time()
            hours = np.round((end_time - start_time) / 3600, 3)
            with open(pre_results_file, "w", encoding="utf-8") as f:
                f.write("\n\nEvaluation Took {} Hours".format(hours))
                f.write("\n\n")
                f.write("The Evaluation Results before Editing:")
                f.write("\n\n")
                json.dump(pre_metrics, f, ensure_ascii=False, indent=2)
                f.write("\n\n")
            if cfg.neighborhood_logits:
                logits_dict = {
                    "target_true_logits": target_true_logits,
                    "target_new_logits": target_new_logits
                }
                torch.save(logits_dict, pre_logits_file)
                target_true_logits.clear()
                target_new_logits.clear()
            print("End Evaluating the Original Model")
            if cfg.lw_eval:
                file=cfg.results_dir+"/lw_eval/"+cfg.llms.alias.replace("/", "-")+"/"+cfg.data+"/pred_lw_eval.npy"
                file_orders=cfg.results_dir+"/lw_eval/"+cfg.llms.alias.replace("/", "-")+"/"+cfg.data+"/abcd_orders.npy"
                ensure_file_directory(file)
                ensure_file_directory(file_orders)
                np.save(file,np.array(predicts))
                np.save(file_orders,np.array(abcd_orders))
                abcd_orders.clear()
                predicts.clear()
        edited_model=load_model(model,cfg)
        print("Start Evaluating the Edited Model")
        if cfg.neighborhood_logits:
            target_true_logits, target_new_logits = get_neighborhood_logit_buffers(
                cfg.data
            )
            target_true_logits.clear()
            target_new_logits.clear()
        post_metrics = eval_algo(cfg, edited_model, tok, data)
        # formatted_time = datetime.now().strftime("%d_%H_%M_%S")
        # post_results_file = cfg.results_dir + "/{}/{}/{}-{}-{}".format(cfg.data, cfg.llms.name.replace("/","-"),cfg.algs.name, cfg.num_edits,formatted_time)
        post_results_file = cfg.results_dir + "/{}/{}/{}-{}".format(cfg.data, cfg.llms.alias.replace("/","-"),cfg.algs.name, cfg.save_name)
        ensure_file_directory(post_results_file)

        if cfg.neighborhood_logits:
            # Edited logits are consumed immediately by the synchronous
            # KL/top-k calculation, so keeping another full-vocabulary cache
            # on disk only wastes space. The reusable original-model cache is
            # deliberately preserved.
            post_logits_file = (
                post_results_file + "_neighborhood_target_logits.pt"
            )
            original_logits = None
            try:
                original_logits = torch.load(
                    pre_logits_file, map_location="cpu"
                )["target_true_logits"]
                locality_metrics = compute_neighborhood_locality(
                    original_logits,
                    target_true_logits,
                    topks=cfg.neighborhood_locality_topks,
                    batch_size=cfg.neighborhood_locality_batch_size,
                    device=device,
                )
                post_metrics.update(locality_metrics)
                print("Neighborhood locality metrics:")
                print_dict(locality_metrics)
            finally:
                target_true_logits.clear()
                target_new_logits.clear()
                del original_logits
                if os.path.exists(post_logits_file):
                    os.remove(post_logits_file)
                    print(
                        "Removed edited neighborhood logits cache: "
                        f"{post_logits_file}"
                    )

        with open(post_results_file, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg) + "\n\n")  # 写入字符串，加空行分隔
            f.write("The Evaluation Results after Editing:")
            f.write("\n\n")
            json.dump(post_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n\n")
        print("End Evaluating the Edited Model")
        if cfg.lw_eval:
            file = cfg.results_dir+"/lw_eval/"+cfg.llms.alias.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/pred_lw_eval.npy"
            file_orders = cfg.results_dir+"/lw_eval/"+cfg.llms.alias.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/abcd_orders.npy"
            ensure_file_directory(file)
            ensure_file_directory(file_orders)
            np.save(file, np.array(predicts))
            np.save(file_orders,np.array(abcd_orders))
    elif not cfg.tf_props and not cfg.inversion_tf and not cfg.unlearning_ab:
        if cfg.debug_mode:
            pre_metrics = eval_algo(cfg, model, tok, data)  # 不是每一次都有必要进行这个。

        edited_model = apply_algo(model,tok,data,cfg)

        if cfg.debug_mode:
            post_metrics = eval_algo(cfg, edited_model, tok, data)
            print("The Evaluation Results before Editing:")
            print_dict(pre_metrics)
            print("\n\n")
            print("The Evaluation Results after Editing:")
            print_dict(post_metrics)
        else:
            save_model(edited_model,cfg)

if __name__ == "__main__":
    main()
