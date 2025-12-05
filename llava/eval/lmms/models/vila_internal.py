import copy
import json
from typing import List, Optional, Tuple

import accelerate
import torch
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.utils import distributed as dist


@register_model("vila_internal")
class VILA(lmms):
    def __init__(self, model_path: str, conv_mode: str, batch_size: int = 1) -> None:
        super().__init__()
        assert batch_size == 1, "VILA only supports batch size of 1 at the moment."

        devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
        torch.cuda.set_device(devices[0])

        self.model = llava.load(model_path, devices=devices)
        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode].copy()

        self.accelerator = accelerate.Accelerator()
        self.device = torch.device(f"cuda:{devices[0]}")
        self._world_size = dist.size()
        self._rank = dist.rank()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        for request in tqdm(requests, disable=not dist.is_main()):
            contexts, generation_kwargs, doc_to_visual, doc_id, task, split = request.args

            images = doc_to_visual(self.task_dict[task][split][doc_id])
            prompt = images + [contexts]

            generation_config = copy.deepcopy(self.model.generation_config)
            generation_config.update(**generation_kwargs)

            response = self.model.generate_content(prompt, generation_config=generation_config)
            responses.append(response)

            print("Prompt:", prompt)
            print("Response:", response)
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
