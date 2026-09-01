import numpy
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import struct
import os
from util.nethook import Trace, set_requires_grad
from util.runningstats import (
    CombinedStat,
    Mean,
    NormMean,
    SecondMoment,
    WeightedSecondMoment,
    tally,
)
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

null_numpy_value = numpy.array(
    struct.unpack(">d", struct.pack(">Q", 0xFFF8000000000002))[0], dtype=numpy.float64
)
global_load_cache_enabled = True
STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def _next_token_probabilities(logits, input_ids):
    """Return p(input_ids[t + 1] | input_ids[:t + 1]) for every position t."""
    next_token_logits = logits[:, :-1, :]
    next_token_ids = input_ids[:, 1:].unsqueeze(-1)
    target_logits = next_token_logits.gather(-1, next_token_ids).squeeze(-1)
    return torch.exp(target_logits - torch.logsumexp(next_token_logits, dim=-1))


def _smooth_probability_weights(probabilities, epsilon, alpha):
    """Floor and flatten next-token probability weights."""
    if not 0 <= epsilon <= 1:
        raise ValueError("prob_weight_epsilon must be between 0 and 1")
    if alpha <= 0:
        raise ValueError("prob_weight_alpha must be greater than 0")
    return epsilon + (1 - epsilon) * probabilities.pow(alpha)


def _probability_weighted_cache_suffix(cache_filename_suffix, epsilon, alpha):
    epsilon_text = format(epsilon, ".6g").replace(".", "p")
    alpha_text = format(alpha, ".6g").replace(".", "p")
    weighted_suffix = f"next-token-prob-eps{epsilon_text}-alpha{alpha_text}"
    return "-".join(filter(None, [cache_filename_suffix, weighted_suffix]))


def layer_stats(
        cfg,
        model,
        tokenizer,
        layer,
        ds_name,
        to_collect,
        sample_size=None,
        precision=None,
        batch_tokens=None,
        progress=tqdm,
        force_recompute=False,
        cache_filename_suffix="",
        random_sample=1,
        probability_weighted=False,
        prob_weight_epsilon=0.1,
        prob_weight_alpha=0.5,
):
    """
    Function to load or compute cached stats.
    """
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    def get_ds(maxlen):
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('data/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        if ds_name in ["wikipedia", "wikitext"]:
                # dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name]
            raw_ds = load_dataset(
                ds_name,
                dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name]
            )
        else:
            raw_ds = load_dataset("json", data_files={"train": ds_name})
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    assert hasattr(model.config, 'max_position_embeddings') or hasattr(model.config, 'seq_length'),\
        ("the max sequence length can not be obtained by model.config.max_position_embeddings or model.config.seq_length,"
         "Please obtain it on your own and specify it via mom2_maxseqlen in directory configs/llms")
    if hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    else:
        npos = model.config.seq_length
    if batch_tokens is not None and batch_tokens < npos:
        npos = batch_tokens
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    # stats_dir = Path(stats_dir)
    if probability_weighted:
        if set(to_collect) != {"mom2"}:
            raise ValueError("Probability weighting currently supports only mom2")
        cache_filename_suffix = _probability_weighted_cache_suffix(
            cache_filename_suffix, prob_weight_epsilon, prob_weight_alpha
        )
    filename=cfg.cache_dir+"/stats/"+cfg.llms.alias.replace("/","-") + "/layer-" + str(layer) +("-" if cache_filename_suffix !="" else "")+ cache_filename_suffix + ".npz"
    if cache_filename_suffix == "local":
        sample_indices_filename=cfg.cache_dir+"/stats/"+cfg.llms.alias.replace("/","-") + "/layer-" + str(layer) + "-local-sample-indices.npz"
    # file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    # filename = stats_dir / file_extension

    # print(f"Computing Cov locally....")

    # A forced recomputation still needs the source dataset even when a cache
    # file already exists. Without this condition tally() receives None and
    # fails while constructing its DataLoader.
    ds = get_ds(npos) if force_recompute or not os.path.exists(filename) else None
    if progress is None:
        progress = lambda x: x

    if probability_weighted:
        stat = CombinedStat(mom2=WeightedSecondMoment())
    else:
        stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    tally_loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=random_sample,
        num_workers=4,
    )
    if not isinstance(tally_loader, tuple):
        loader = tally_loader
    else:
        loader = tally_loader[0]
        ori_loader = tally_loader[1]
        if cache_filename_suffix == "local" and not os.path.exists(sample_indices_filename) and ds is not None:
            # save the sampled indices
            numpy.savez(sample_indices_filename, sample_indices=numpy.array(list(ori_loader.sampler)))
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                if probability_weighted:
                    with Trace(
                            model,
                            cfg.llms.rewrite_module_tmp.format(layer),
                            retain_input=True,
                            retain_output=False,
                            stop=False,
                    ) as tr:
                        outputs = model(**batch, use_cache=False)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    probabilities = _next_token_probabilities(logits, batch["input_ids"])
                    # Logits are much larger than the scalar probabilities; release
                    # them before accumulating this batch and starting the next one.
                    del outputs, logits
                    next_token_mask = (
                        batch["attention_mask"][:, :-1].bool()
                        & batch["attention_mask"][:, 1:].bool()
                    )
                    feats = flatten_masked_batch(
                        tr.input[:, 1:], next_token_mask
                    )
                    weights = _smooth_probability_weights(
                        probabilities[next_token_mask],
                        prob_weight_epsilon,
                        prob_weight_alpha,
                    )
                else:
                    with Trace(
                            model,
                            cfg.llms.rewrite_module_tmp.format(layer),
                            retain_input=True,
                            retain_output=False,
                            stop=True,
                    ) as tr:
                        model(**batch)
                    feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                if "normlize" in cache_filename_suffix:
                    import torch.nn.functional as F
                    feats = F.normalize(feats, p=2, dim=-1)
                feats = feats.to(dtype=dtype)
                if probability_weighted:
                    stat.add(feats, weights.to(dtype=dtype))
                else:
                    stat.add(feats)
    return stat

def get_cov(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    cache_filename_suffix="",
    random_sample=1,
    probability_weighted=None,
    prob_weight_epsilon=None,
    prob_weight_alpha=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    model_name = cfg.llms.alias.replace("/", "-")
    if probability_weighted is None:
        probability_weighted = cfg.get("cov_probability_weighted", False)
    if prob_weight_epsilon is None:
        prob_weight_epsilon = cfg.get("cov_prob_weight_epsilon", 0.1)
    if prob_weight_alpha is None:
        prob_weight_alpha = cfg.get("cov_prob_weight_alpha", 0.5)
    # key = (model_name, cfg.llms.rewrite_module_tmp.format(layer))

    print(f"Retrieving covariance statistics for {model_name} @ layer {cfg.llms.rewrite_module_tmp.format(layer)}.")
    # if key not in COV_CACHE or force_recompute:
    stat = layer_stats(
        cfg,
        model,
        tok,
        layer,
        mom2_dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
        batch_tokens=cfg.llms.mom2_maxseqlen,
        force_recompute=force_recompute,
        cache_filename_suffix=cache_filename_suffix,
        random_sample=random_sample,
        probability_weighted=probability_weighted,
        prob_weight_epsilon=prob_weight_epsilon,
        prob_weight_alpha=prob_weight_alpha,
    )
    cov=stat.mom2.moment()
    return (
        torch.inverse(cov) if inv else cov
    )


def get_probability_weighted_cov(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    cache_filename_suffix="",
    random_sample=1,
    prob_weight_epsilon: float = 0.1,
    prob_weight_alpha: float = 0.5,
) -> torch.Tensor:
    """Compute E[w h h^T] / E[w] using smoothed next-token probabilities."""
    return get_cov(
        cfg,
        model,
        tok,
        layer,
        mom2_dataset,
        mom2_n_samples,
        mom2_dtype,
        inv=inv,
        force_recompute=force_recompute,
        cache_filename_suffix=cache_filename_suffix,
        random_sample=random_sample,
        probability_weighted=True,
        prob_weight_epsilon=prob_weight_epsilon,
        prob_weight_alpha=prob_weight_alpha,
    )

def is_null_numpy_value(v):
    """
    True if v is a 64-bit float numpy scalar NaN matching null_numpy_value.
    """
    return (
        isinstance(v, numpy.ndarray)
        and numpy.ndim(v) == 0
        and v.dtype == numpy.float64
        and numpy.isnan(v)
        and 0xFFF8000000000002 == struct.unpack(">Q", struct.pack(">d", v))[0]
    )

def unbox_numpy_null(d):
    """
    Reverses box_numpy_null, replacing null_numpy_value with None.
    Recursively descends into a dictionary replacing None values.
    """
    try:
        return {k: unbox_numpy_null(v) for k, v in d.items()}
    except Exception:
        return None if is_null_numpy_value(d) else d

def box_numpy_null(d):
    """
    Replaces None with null_numpy_value, leaving non-None values unchanged.
    Recursively descends into a dictionary replacing None values.
    """
    try:
        return {k: box_numpy_null(v) for k, v in d.items()}
    except Exception:
        return null_numpy_value if d is None else d

def resolve_state_dict(s):
    """
    Resolves a state, which can be a filename or a dict-like object.
    """
    if isinstance(s, str):
        return unbox_numpy_null(numpy.load(s))
    return s

def load_cached_state(cachefile, args, quiet=False, throw=False):
    """
    Resolves a state, which can be a filename or a dict-like object.
    """
    if not global_load_cache_enabled or cachefile is None:
        return None
    try:
        if isinstance(cachefile, dict):
            dat = cachefile
            cachefile = "state"  # for printed messages
        else:
            dat = unbox_numpy_null(numpy.load(cachefile))
        for a, v in args.items():
            if a not in dat or dat[a] != v:
                if not quiet:
                    print("%s %s changed from %s to %s" % (cachefile, a, dat[a], v))
                return None
    except (FileNotFoundError, ValueError) as e:
        if throw:
            raise e
        return None
    else:
        if not quiet:
            print("Loading cached %s" % cachefile)
        return dat


def save_cached_state(cachefile, obj, args):
    """
    Saves the state_dict of the given object in a dict or npz file.
    """
    if cachefile is None:
        return
    dat = obj.state_dict()
    for a, v in args.items():
        if a in dat:
            assert dat[a] == v
        dat[a] = v
    if isinstance(cachefile, dict):
        cachefile.clear()
        cachefile.update(dat)
    else:
        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        numpy.savez(cachefile, **box_numpy_null(dat))


class Stat:
    """
    Abstract base class for a running pytorch statistic.
    """

    def __init__(self, state):
        """
        By convention, all Stat subclasses can be initialized by passing
        state=; and then they will initialize by calling load_state_dict.
        """
        self.load_state_dict(resolve_state_dict(state))

    def add(self, x, *args, **kwargs):
        """
        Observes a batch of samples to be incorporated into the statistic.
        Dimension 0 should be the batch dimension, and dimension 1 should
        be the feature dimension of the pytorch tensor x.
        """
        pass

    def load_state_dict(self, d):
        """
        Loads this Stat from a dictionary of numpy arrays as saved
        by state_dict.
        """
        pass

    def state_dict(self):
        """
        Saves this Stat as a dictionary of numpy arrays that can be
        stored in an npz or reloaded later using load_state_dict.
        """
        return {}

    def save(self, filename):
        """
        Saves this stat as an npz file containing the state_dict.
        """
        save_cached_state(filename, self, {})

    def load(self, filename):
        """
        Loads this stat from an npz file containing a saved state_dict.
        """
        self.load_state_dict(load_cached_state(filename, {}, quiet=True, throw=True))

    def to_(self, device):
        """
        Moves this Stat to the given device.
        """
        pass

    def cpu_(self):
        """
        Moves this Stat to the cpu device.
        """
        self.to_("cpu")

    def cuda_(self):
        """
        Moves this Stat to the default cuda device.
        """
        self.to_("cuda")

    def _normalize_add_shape(self, x, attr="data_shape"):
        """
        Flattens input data to 2d.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if len(x.shape) < 1:
            x = x.view(-1)
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            data_shape = x.shape[1:]
            setattr(self, attr, data_shape)
        else:
            assert x.shape[1:] == data_shape
        return x.view(x.shape[0], int(numpy.prod(data_shape)))

    def _restore_result_shape(self, x, attr="data_shape"):
        """
        Restores output data to input data shape.
        """
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            return x
        return x.view(data_shape * len(x.shape))


class SecondMoment(Stat):
    """
    Running computation. Use this when the entire non-centered 2nd-moment
    'covariance-like' matrix is needed, and when the whole matrix fits
    in the GPU.
    """

    def __init__(self, split_batch=True, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self.mom2 = None
        self.split_batch = split_batch

    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        # Initial batch reveals the shape of the data.
        if self.count == 0:
            self.mom2 = a.new(a.shape[1], a.shape[1]).zero_()
        batch_count = a.shape[0]
        # Update the covariance using the batch deviation
        self.count += batch_count
        self.mom2 += a.t().mm(a)

    def to_(self, device):
        if self.mom2 is not None:
            self.mom2 = self.mom2.to(device)

    def moment(self):
        return self.mom2 / self.count

    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            mom2=self.mom2.cpu().numpy(),
        )

    def load_state_dict(self, state):
        self.count = int(state["count"])
        self.mom2 = torch.from_numpy(state["mom2"])
