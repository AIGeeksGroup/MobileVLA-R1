"""Merge SFT adapters and export all modalities. Runs only when explicitly invoked."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--lora-path', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    from peft import PeftModel
    from llava.model import LlavaLlamaConfig, LlavaLlamaModel
    from mobilevla.checkpoints import load_non_lora

    config = LlavaLlamaConfig.from_pretrained(args.lora_path)
    # The adapter config may contain checkpoint-relative component paths. Load
    # base component locations while retaining the adapter's modality settings.
    base = LlavaLlamaConfig.from_pretrained(args.base_model)
    for name in ('llm_cfg', 'vision_tower_cfg', 'mm_projector_cfg'):
        setattr(config, name, getattr(base, name))
    config.auxiliary_weights_file = getattr(base, 'auxiliary_weights_file', None)
    model = LlavaLlamaModel.from_pretrained(args.base_model, config=config)
    non_lora = Path(args.lora_path) / 'non_lora_trainables.bin'
    if not non_lora.is_file():
        raise FileNotFoundError(f'Missing SFT non-LoRA parameters: {non_lora}')
    load_non_lora(model, non_lora)
    model = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
    model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
