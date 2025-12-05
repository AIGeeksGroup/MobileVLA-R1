from peft import PeftModel
import os
import torch
from transformers import AutoModelForCausalLM
from llava.model import *
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_image, tokenizer_image_token

# Load base model
model = LlavaLlamaModel.from_pretrained("/root/autodl-tmp/model/finetune")
print("Loaded base model")

# Load non-LoRA weights
lora_path = "./checkpoints/navila-8b-8f-sft-lora/sft_new1"
non_lora_path = os.path.join(lora_path, "non_lora_trainables.bin")
if os.path.exists(non_lora_path):
    non_lora_weights = torch.load(non_lora_path, map_location="cpu")
    model.load_state_dict(non_lora_weights, strict=False)
    print("Loaded non-LoRA trainables")

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "./checkpoints/navila-8b-8f-sft-lora/sft_new1")
print("Loaded LoRA weights")

# Merge LoRA weights into the model
model = model.merge_and_unload()
print("Merged LoRA weights into the model")

# Save the merged full model
model.save_pretrained("/root/autodl-tmp/model/MobileVla-r1-8b")