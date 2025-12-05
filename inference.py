import os
from pathlib import Path
import torch
import json
import argparse
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

from llava.model import *
from llava.data.nav_cot_utils import depth_to_point_cloud, load_depth_map
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_image, tokenizer_image_token
from peft import PeftModel


class NaVILAImageInference:
    def __init__(
        self,
        model_path,
        lora_path=None,
        device="cuda",
        use_flash_attn=True,
        depth_scale: float = 1000.0,
        pointcloud_points: int = 2048,
    ):
        """
        Args:
            model_path: path to the base model
            lora_path: path to LoRA weights
            device: torch device
            use_flash_attn: enable Flash Attention 2 for acceleration
        """
        self.device = device
        self.model_path = model_path
        self.lora_path = lora_path
        self.use_flash_attn = use_flash_attn
        self.depth_scale = depth_scale
        self.pointcloud_points = pointcloud_points
        
        # Load model and tokenizer
        self._load_model()
        self._setup_conversation()
        
    def _load_model(self):
        """Load the NaVILA model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        # Load config from LoRA path if provided, otherwise from base model
        if self.lora_path and os.path.exists(self.lora_path):
            config = LlavaLlamaConfig.from_pretrained(self.lora_path)
        else:
            config = LlavaLlamaConfig.from_pretrained(self.model_path)
        
        config.use_cache = True
        
        if self.use_flash_attn:
            config._attn_implementation = "flash_attention_2"
            print("Using Flash Attention 2.0 for acceleration")
        
        model_kwargs = {
            "config": config,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if self.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        try:
            model = LlavaLlamaModel.from_pretrained(
                self.model_path,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error loading model with Flash Attention: {e}")
            print("Retrying without Flash Attention...")
            model_kwargs.pop("attn_implementation", None)
            if hasattr(config, '_attn_implementation'):
                delattr(config, '_attn_implementation')
            self.use_flash_attn = False
            
            model = LlavaLlamaModel.from_pretrained(
                self.model_path,
                **model_kwargs
            )
        
        model = model.to(torch.bfloat16)
        
        if self.lora_path and os.path.exists(self.lora_path):
            print(f"Loading LoRA weights from {self.lora_path}...")
            
            non_lora_path = os.path.join(self.lora_path, "non_lora_trainables.bin")
            if os.path.exists(non_lora_path):
                non_lora_weights = torch.load(non_lora_path, map_location="cpu")
                model.load_state_dict(non_lora_weights, strict=False)
                print("Loaded non-LoRA trainables")
            
            model = PeftModel.from_pretrained(
                model, 
                self.lora_path,
                torch_dtype=torch.bfloat16
            )
            print("LoRA weights loaded successfully")
        
        model.eval()
        self.model = model
        self.tokenizer = model.tokenizer
        
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            
    def _setup_conversation(self):
        """Configure the conversation template used for inference."""
        self.conv_mode = "llama_3"
        if self.conv_mode not in conv_templates:
            self.conv_mode = "vicuna_v1"

    def _prepare_depth_tensor(self, depth_input):
        if depth_input is None:
            return None
        if isinstance(depth_input, (str, Path)):
            depth_input = [depth_input]
        tensors = []
        for item in depth_input:
            if isinstance(item, torch.Tensor):
                tensor = item
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
            else:
                tensor = load_depth_map(str(item), scale=self.depth_scale)
            tensors.append(tensor)
        depth_tensor = torch.cat(tensors, dim=0)
        return depth_tensor.to(self.device, dtype=torch.bfloat16)

    def _prepare_point_tensor(self, point_input, depth_tensor=None):
        if point_input is None and depth_tensor is None:
            return None
        if point_input == "from_depth" and depth_tensor is not None:
            pcs = []
            for depth_map in depth_tensor:
                pc = depth_to_point_cloud(
                    depth_map.unsqueeze(0).cpu(),
                    max_points=self.pointcloud_points,
                    normalize=True,
                )
                pcs.append(pc)
            point_tensor = torch.stack(pcs)
            return point_tensor.to(self.device, dtype=torch.bfloat16)
        if isinstance(point_input, torch.Tensor):
            return point_input.to(self.device, dtype=torch.bfloat16)
        if isinstance(point_input, (str, Path)):
            data = np.load(str(point_input))
            tensor = torch.from_numpy(data).float()
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            return tensor.to(self.device, dtype=torch.bfloat16)
        if isinstance(point_input, list):
            tensors = []
            for item in point_input:
                tensors.append(self._prepare_point_tensor(item))
            return torch.cat(tensors, dim=0)
        return None

    def _build_image_payload(self, rgb_tensor, depth_tensor=None, point_tensor=None):
        token_types = ["rgb"] * rgb_tensor.shape[0]
        if depth_tensor is not None:
            token_types += ["depth"] * depth_tensor.shape[0]
        if point_tensor is not None:
            token_types += ["point"] * point_tensor.shape[0]
        payload = {"rgb": rgb_tensor, "token_types": token_types}
        if depth_tensor is not None:
            payload["depth"] = depth_tensor
        if point_tensor is not None:
            payload["point"] = point_tensor
        return payload
    
    def load_image_from_pil(self, image: Image.Image):
        """Convert a PIL.Image to tensor."""
        class TempDataArgs:
            def __init__(self):
                self.image_aspect_ratio = "resize"
                self.image_processor = None

        data_args = TempDataArgs()
        data_args.image_processor = self.model.get_vision_tower().image_processor
        image_tensor = process_image(image, data_args, None)
        return image_tensor.unsqueeze(0)
        
    def load_image(self, image_path):
        """Load an image from path and preprocess it."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            image = Image.open(image_path).convert('RGB')
            
            class TempDataArgs:
                def __init__(self):
                    self.image_aspect_ratio = "resize"
                    self.image_processor = None
            
            data_args = TempDataArgs()
            data_args.image_processor = self.model.get_vision_tower().image_processor
            
            image_tensor = process_image(image, data_args, None)
            
            return image_tensor.unsqueeze(0)
            
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
    
    def load_multiple_images(self, image_paths):
        """Load multiple images and stack them."""
        image_tensors = []
        for image_path in image_paths:
            image_tensor = self.load_image(image_path)
            image_tensors.append(image_tensor.squeeze(0))
        
        return torch.stack(image_tensors)
    
    def generate_response(
        self,
        image_input,
        question,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        depth_input=None,
        point_cloud=None,
        return_token_ids=False,
    ):
        """Generate a response with optional depth and point-cloud inputs."""
        
        if isinstance(image_input, str):
            image_tensors = self.load_image(image_input)
            num_images = 1
        elif isinstance(image_input, list):
            image_tensors = self.load_multiple_images(image_input)
            num_images = len(image_input)
        elif isinstance(image_input, torch.Tensor):
            image_tensors = image_input
            num_images = image_tensors.shape[0] if len(image_tensors.shape) == 4 else 1
        else:
            raise ValueError("image_input must be a path, list of paths, or tensor")
        
        conv = conv_templates[self.conv_mode].copy()
        
        depth_tensor = self._prepare_depth_tensor(depth_input)
        point_tensor = self._prepare_point_tensor(point_cloud, depth_tensor)
        payload = self._build_image_payload(image_tensors, depth_tensor, point_tensor)
        token_block = ""
        if len(payload["token_types"]) > 0:
            token_block = "\n".join([DEFAULT_IMAGE_TOKEN for _ in payload["token_types"]]) + "\n"
        question_with_image = f"{token_block}{question}"
        
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        if len(image_tensors.shape) == 3:
            image_tensors = image_tensors.unsqueeze(0)
        image_tensors = image_tensors.to(self.device, dtype=torch.bfloat16)
        payload["rgb"] = image_tensors
        with torch.inference_mode():
            if self.use_flash_attn:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output_ids = self.model.generate(
                        input_ids,
                        images=payload,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p if do_sample else 1.0,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                output_ids = self.model.generate(
                    input_ids,
                    images=payload,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode text output
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(
            output_ids, 
            skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if return_token_ids:
            return outputs, output_ids[0]
        return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the base model")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to the LoRA weights")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the input image")
    parser.add_argument("--question", type=str, default="Describe this image in detail.",
                       help="Question about the image")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_flash_attn", action="store_true",
                       help="Disable Flash Attention")
    
    args = parser.parse_args()
    
    try:
        inferencer = NaVILAImageInference(
            model_path=args.model_path,
            lora_path=args.lora_path,
            use_flash_attn=not args.no_flash_attn
        )
        
        response = inferencer.generate_response(
            image_input=args.image_path,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print(f"Question: {args.question}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()