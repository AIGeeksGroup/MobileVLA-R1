"""Helpers shared by SFT, adapter merging and inference."""
AUX_MODULES = ('depth_tower', 'point_tower', 'depth_bridge', 'point_bridge')


def strip_adapter_prefix(key):
    prefix = 'base_model.model.'
    return key[len(prefix):] if key.startswith(prefix) else key


def auxiliary_buffers(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()
            if any(part in AUX_MODULES for part in name.split('.'))}


def load_non_lora(model, path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = {strip_adapter_prefix(k): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f'Non-LoRA checkpoint contains unrecognized weights: {result.unexpected_keys}')
    return result


def require_auxiliary_checkpoint(model):
    if any(getattr(model, name, None) is not None for name in AUX_MODULES):
        if not getattr(model.config, 'auxiliary_weights_file', None):
            raise ValueError('Export a full SFT checkpoint with depth/point weights before starting GRPO')
