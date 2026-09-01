from typing import Iterable, Iterator, Sequence

import torch


def _iter_first_token_logits(logit_groups: Iterable) -> Iterator[torch.Tensor]:
    """Yield one full-vocabulary logit vector per neighborhood example."""
    for sample_index, group in enumerate(logit_groups):
        if torch.is_tensor(group):
            if group.ndim == 1:
                first_token_logits = group
            elif group.ndim == 2 and group.shape[0] > 0:
                first_token_logits = group[0]
            else:
                raise ValueError(
                    f"Unexpected tensor shape at sample {sample_index}: "
                    f"{tuple(group.shape)}"
                )
        elif isinstance(group, (list, tuple)) and len(group) > 0:
            first_token_logits = group[0]
        else:
            raise ValueError(
                f"Missing first-token logits at sample {sample_index}."
            )

        if not torch.is_tensor(first_token_logits):
            first_token_logits = torch.as_tensor(first_token_logits)
        if first_token_logits.ndim != 1:
            raise ValueError(
                f"Expected a [vocab_size] vector at sample {sample_index}, got "
                f"{tuple(first_token_logits.shape)}."
            )
        yield first_token_logits.detach().cpu()


def _batched(iterator: Iterator[torch.Tensor], batch_size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield torch.stack(batch)
            batch = []
    if batch:
        yield torch.stack(batch)


@torch.inference_mode()
def compute_neighborhood_locality(
    original_logit_groups: Iterable,
    edited_logit_groups: Iterable,
    topks: Sequence[int] = (1, 5, 10),
    batch_size: int = 8,
    device=None,
):
    """
    Compare first-token full-vocabulary distributions on neighborhood prompts.

    KL is KL(original || edited). Top-k overlap is |A intersect B| / k.
    Both metrics are macro-averaged over neighborhood examples.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    topks = tuple(sorted(set(int(k) for k in topks)))
    if not topks or any(k <= 0 for k in topks):
        raise ValueError(f"topks must contain positive integers, got {topks}.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    original_iterator = _iter_first_token_logits(original_logit_groups)
    edited_iterator = _iter_first_token_logits(edited_logit_groups)

    kl_sum = 0.0
    overlap_sums = {k: 0.0 for k in topks}
    sample_count = 0

    original_batches = _batched(original_iterator, batch_size)
    edited_batches = _batched(edited_iterator, batch_size)

    while True:
        original_batch = next(original_batches, None)
        edited_batch = next(edited_batches, None)
        if original_batch is None or edited_batch is None:
            if original_batch is not None or edited_batch is not None:
                raise ValueError(
                    "Original and edited neighborhood logit counts do not match."
                )
            break

        if original_batch.shape != edited_batch.shape:
            raise ValueError(
                "Original and edited logit shapes do not match: "
                f"{tuple(original_batch.shape)} vs {tuple(edited_batch.shape)}."
            )
        if max(topks) > original_batch.shape[-1]:
            raise ValueError(
                f"Requested top-{max(topks)} for vocabulary size "
                f"{original_batch.shape[-1]}."
            )

        original_batch = original_batch.to(device=device, dtype=torch.float32)
        edited_batch = edited_batch.to(device=device, dtype=torch.float32)
        if not torch.isfinite(original_batch).all() or not torch.isfinite(edited_batch).all():
            raise ValueError("Neighborhood logits contain NaN or infinity.")

        original_log_probs = torch.log_softmax(original_batch, dim=-1)
        edited_log_probs = torch.log_softmax(edited_batch, dim=-1)
        original_probs = original_log_probs.exp()
        per_sample_kl = torch.sum(
            original_probs * (original_log_probs - edited_log_probs), dim=-1
        )
        kl_sum += per_sample_kl.sum().item()

        for k in topks:
            original_topk = torch.topk(original_batch, k=k, dim=-1).indices
            edited_topk = torch.topk(edited_batch, k=k, dim=-1).indices
            overlap_count = (
                original_topk.unsqueeze(-1) == edited_topk.unsqueeze(-2)
            ).any(dim=-1).sum(dim=-1)
            overlap_sums[k] += (overlap_count.float() / k).sum().item()

        sample_count += original_batch.shape[0]

    if sample_count == 0:
        raise ValueError("No neighborhood logits were collected.")

    metrics = {
        "neighborhood_kl_original_to_edited": kl_sum / sample_count,
        "neighborhood_locality_num_samples": sample_count,
    }
    metrics.update(
        {
            f"neighborhood_top{k}_overlap": overlap_sums[k] / sample_count
            for k in topks
        }
    )
    return metrics
