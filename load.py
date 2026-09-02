import json
import torch
from util import nethook
from util.utility import ensure_file_directory
from algs.wise import WISE


MEMIT_LORA_CHECKPOINT_FORMAT = "memit_lora_v1"


def _merge_memit_lora_checkpoint(model, checkpoint):
    if checkpoint.get("__format__") != MEMIT_LORA_CHECKPOINT_FORMAT:
        raise ValueError(
            "Unsupported MEMIT-LoRA checkpoint format: "
            f"{checkpoint.get('__format__')!r}"
        )
    adapters = checkpoint.get("adapters")
    if not isinstance(adapters, dict) or not adapters:
        raise ValueError("MEMIT-LoRA checkpoint has no adapters")
    with torch.no_grad():
        for module_name, state in adapters.items():
            weight = nethook.get_parameter(model, f"{module_name}.weight")
            a = state["a"].to(device=weight.device, dtype=torch.float32)
            b = state["b"].to(device=weight.device, dtype=torch.float32)
            delta = (b @ a) * float(state["scale"])
            if bool(state.get("transpose_for_weight", False)):
                delta = delta.T
            if tuple(delta.shape) != tuple(weight.shape):
                raise ValueError(
                    f"LoRA delta for {module_name} has shape {tuple(delta.shape)}, "
                    f"expected {tuple(weight.shape)}"
                )
            if not torch.isfinite(delta).all():
                raise FloatingPointError(
                    f"LoRA delta for {module_name} contains non-finite values"
                )
            weight.add_(delta.to(dtype=weight.dtype))
    print(f"Merged {len(adapters)} MEMIT-LoRA adapters into the base model")
    return model


def load_data(cfg):
    if cfg.data.endswith(".json"):
        data_file=cfg.data_dir+"/"+cfg.data
    else:
        data_file = cfg.data_dir+"/"+ cfg.data+".json"
    if cfg.negetive_prompt_test and not cfg.target_true_test:
        if cfg.data=="multi_counterfact_20877":
            data_file = cfg.data_dir+"/mcf_negetive_prompt.json"
            print("load mcf_negetive_prompt.json......")
        elif cfg.data=="wiki_cf_2266":
            data_file = cfg.data_dir+"/wiki_negetive_prompt.json"
            print("load wiki_negetive_prompt.json......")
        elif cfg.data=="mquake_cf_9218":
            data_file = cfg.data_dir+"/mq_negetive_prompt.json"
            print("load mq_negetive_prompt.json......")
        elif cfg.data=="zsre_mend_eval_19086":
            data_file = cfg.data_dir+"/zsre_negetive_prompt.json"
            print("load zsre_negetive_prompt.json......")
        else:
            raise ValueError("{} has no negetive prompt.".format(cfg.data))
    if cfg.target_true_test:
        data_file = cfg.data_dir + "/" + cfg.data + cfg.target_true_file_suffix + ".json"
    with open(data_file, "r") as f:
        data = json.load(f)
    if cfg.algs.name == "wise":
        num_data = len(data)
        if cfg.num_edits*2 > num_data:
            print("编辑数量超过了数据集容量的一半，由于WISE需要同样多的无关数据做训练，因此将使用mcf中的数据做无关数据")
            with open(cfg.data_dir+"/multi_counterfact_20877.json",'r') as f:
                loc_data = json.load(f)
            loc_data = loc_data[-cfg.num_edits:]
        else:
            loc_data = data[-cfg.num_edits:]
        data = data[:cfg.num_edits]
        for i in range(cfg.num_edits):
            loc_data_i = loc_data[i]
            data[i]["loc_prompt"] = loc_data_i["prompt"].format(loc_data_i["subject"]) + ' ' + loc_data_i["target_true"] + '.'
    else:
        data=data[:cfg.num_edits]
    if cfg.negetive_prompt_test:
        for i in range(len(data)):
            negetive_prompt = data[i]["negetive_prompt"]
            subject = data[i]["subject"]
            if subject not in negetive_prompt:
                data[i]["negetive_prompt"] += f" {subject} "
                print(f"第{i+1}个样本negetive_prompt中没有subject[{subject}]，将subject接在后面.....")
    return data

def load_model(model,cfg):
    weights_dir = cfg.cache_dir + "/saved_weights"
    # if cfg.algs.name == 'wise':
    #     device = f'cuda:{cfg.gpu}'
    #     editor = WISE.WISE(model=model, config=cfg.llms, device=device)
    #     editor.load(f"{weights_dir}/{cfg.algs.name}/{cfg.data}-{cfg.load_name}-{cfg.llms.alias.replace("/", "-")}.pt")
    #     return editor
    weights_file = weights_dir + "/{}/{}-{}-{}.pt".format(cfg.algs.name, cfg.data, cfg.load_name,
                                                          cfg.llms.alias.replace("/", "-"))
    device=torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    weights=torch.load(weights_file, map_location="cpu")
    if isinstance(weights, dict) and weights.get("__format__") == MEMIT_LORA_CHECKPOINT_FORMAT:
        return _merge_memit_lora_checkpoint(model, weights)
    with torch.no_grad():
        for key, value in weights.items():
            weight=nethook.get_parameter(model,key)
            weight[...]=value.to(device)
    return model

def save_model(model,cfg):
    weights_dir = cfg.cache_dir + "/saved_weights"
    # if cfg.algs.name == 'wise':
    #     model.save(f"{weights_dir}/{cfg.algs.name}/{cfg.data}-{cfg.save_name}-{cfg.llms.alias.replace("/", "-")}.pt")
    #     return
    if cfg.algs.name == "memit_lora":
        weights = getattr(model, "_memit_lora_payload", None)
        if weights is None:
            raise RuntimeError(
                "MEMIT-LoRA finished without a compact adapter payload"
            )
        if weights.get("__format__") != MEMIT_LORA_CHECKPOINT_FORMAT:
            raise RuntimeError("Invalid compact MEMIT-LoRA adapter payload")
    elif cfg.algs.name == 'rledit':
        weights = {
            f"{edite_module}.weight": nethook.get_parameter(
                model, f"{edite_module}.weight"
            ).cpu()
            for edite_module in cfg.llms.edit_modules
        }
    else:
        weights = {
            f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
                model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
            ).cpu()
            for layer in cfg.llms.layers
        }
    weights_file = weights_dir + "/{}/{}-{}-{}.pt".format(cfg.algs.name, cfg.data, cfg.save_name,
                                                          cfg.llms.alias.replace("/", "-"))
    ensure_file_directory(weights_file)
    torch.save(weights, weights_file)  # 好像就完成了，只要保存好这个东西就可以了。
