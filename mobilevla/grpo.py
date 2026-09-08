"""Shared multimodal probability calculation for current, old and reference policies."""
import io

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100  # PyTorch/NaVILA label ignore index; no model imports required.

PROTOCOL_VERSION = 2


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    return value


def dumps(batch):
    buffer = io.BytesIO()
    torch.save(to_cpu(batch), buffer)
    return buffer.getvalue()


def loads(data):
    batch = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if not isinstance(batch, dict) or batch.get('protocol') != PROTOCOL_VERSION:
        raise ValueError('Incompatible GRPO protocol; restart all three processes with the same code')
    return batch


def validate_batch(batch):
    required = ('prompt_ids', 'images', 'completion_ids', 'completion_mask',
                'advantages', 'old_logps', 'policy_version', 'run_id')
    if batch.get('protocol') != PROTOCOL_VERSION or any(k not in batch for k in required):
        raise ValueError('Incomplete multimodal GRPO group')
    ids, mask = batch['completion_ids'], batch['completion_mask']
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] < 2:
        raise ValueError('A group needs at least two completions and matching masks')
    if batch['prompt_ids'].ndim != 2 or batch['prompt_ids'].shape[0] != 1:
        raise ValueError('Each group must have exactly one prompt')
    if mask.dtype != torch.bool or not mask.any(dim=1).all():
        raise ValueError('Completion masks must be boolean and nonempty')
    if ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError('Completion padding must be on the right')
    if batch['advantages'].shape != (ids.shape[0],):
        raise ValueError('Advantages must be computed once over the complete group')
    if ids.dtype != torch.long or batch['prompt_ids'].dtype != torch.long:
        raise ValueError('Token IDs must be int64 tensors')
    payload = batch['images']
    if not isinstance(payload, dict) or not payload.get('token_types'):
        raise ValueError('Missing multimodal observation')
    if int((batch['prompt_ids'] == -200).sum()) != len(payload['token_types']):
        raise ValueError('Image placeholders and observation counts differ')
    for modality in set(payload['token_types']):
        tensor = payload.get(modality)
        if modality not in ('rgb', 'depth', 'point') or not isinstance(tensor, torch.Tensor):
            raise ValueError('Missing or invalid modality tensor')
        if tensor.shape[0] != payload['token_types'].count(modality) or not torch.isfinite(tensor).all():
            raise ValueError('Modality tensor count/value mismatch')
    if batch['old_logps'].shape != ids.shape:
        raise ValueError('Old probabilities must match completion tokens')
    for name in ('advantages', 'old_logps', 'ref_logps'):
        if name in batch and not torch.isfinite(batch[name]).all():
            raise ValueError(f'Non-finite {name}')
    if 'ref_logps' in batch and batch['ref_logps'].shape != ids.shape:
        raise ValueError('Reference probabilities must match completion tokens')


def group_advantages(rewards):
    rewards = rewards.float()
    if rewards.ndim != 1 or rewards.numel() < 2 or not torch.isfinite(rewards).all():
        raise ValueError('Expected finite rewards for one complete group')
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)


def repeat_media(payload, count, device):
    result = {'token_types': payload['token_types'] * count}
    for key in ('rgb', 'depth', 'point'):
        if payload.get(key) is not None:
            result[key] = torch.cat([payload[key]] * count, dim=0).to(device)
    return result


def completion_logps(model, batch, forward=None):
    """Score generated tokens after visual expansion, including EOS.

    All policies call this function with exactly the recorded prompt, media and
    completion IDs. Labels locate completion tokens after visual token expansion.
    No text retokenization and no hard-coded visual sequence length are used.
    """
    model.eval()  # Disable dropout/BN updates while preserving policy gradients.
    device = model.device
    completions = batch['completion_ids'].to(device)
    mask = batch['completion_mask'].to(device)
    count, width = completions.shape
    prompt = batch['prompt_ids'].to(device).expand(count, -1)
    ids = torch.cat([prompt, completions], dim=1)
    attention = torch.cat([torch.ones_like(prompt, dtype=torch.bool), mask], dim=1)
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[:, prompt.shape[1]:] = completions.masked_fill(~mask, IGNORE_INDEX)
    media = repeat_media(batch['images'], count, device)
    _, positions, expanded_mask, _, embeds, expanded_labels = model.prepare_inputs_labels_for_multimodal(
        ids, None, attention, None, labels, media,
    )
    # A context overflow must not silently train on a truncated answer.
    selected = expanded_labels[:, 1:] != IGNORE_INDEX
    if not torch.equal(selected.sum(dim=1), mask.sum(dim=1)):
        raise ValueError('Completion was truncated during multimodal expansion; reduce context/generation length')
    output = (forward or model)(
        inputs_embeds=embeds, attention_mask=expanded_mask, position_ids=positions,
        use_cache=False, dpo_forward=True,
    )
    return completion_logps_from_logits(output.logits, expanded_labels, mask)


def completion_logps_from_logits(logits, expanded_labels, completion_mask):
    """Causal alignment and padding handling, independent of model execution."""
    logits = logits[:, :-1, :]
    targets = expanded_labels[:, 1:]
    selected = targets != IGNORE_INDEX
    if not torch.equal(selected.sum(dim=1), completion_mask.sum(dim=1)):
        raise ValueError('Completion labels/masks differ after visual expansion')
    width = completion_mask.shape[1]
    rows = []
    for row_logits, row_targets, row_selected in zip(logits, targets, selected):
        token_logits = row_logits[row_selected].float()
        token_ids = row_targets[row_selected]
        logps = token_logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) - token_logits.logsumexp(-1)
        rows.append(F.pad(logps, (0, width - logps.numel())))
    return torch.stack(rows)


def grpo_loss(current, batch, beta=0.04, clip=0.2):
    mask = batch['completion_mask'].to(current.device)
    old = batch['old_logps'].to(current.device).float()
    reference = batch['ref_logps'].to(current.device).float()
    advantage = batch['advantages'].to(current.device).float().unsqueeze(1)
    # Group advantages are immutable; never normalize a microbatch here.
    ratio = torch.exp(current.float() - old)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    log_ratio = reference - current.float()
    kl = torch.expm1(log_ratio) - log_ratio
    per_token = -surrogate + beta * kl
    loss = (per_token.masked_fill(~mask, 0).sum(dim=1) / mask.sum(dim=1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite GRPO loss; inspect log-probabilities instead of silently clipping them')
    return loss
