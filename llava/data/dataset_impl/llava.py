import os
from typing import Any, Dict, List

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset
from llava.media import Image
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["LLaVADataset"]


class LLaVADataset(BaseDataset):
    def __init__(self, data_path: str, image_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.image_dir = image_dir
        self.instances = []
        for instance in io.load(self.data_path):
            if "image" in instance:
                image_path = os.path.join(self.image_dir, instance.pop("image"))
                if not os.path.exists(image_path):
                    logger.warning(f"Image `{image_path}` not found. Excluded from dataset.")
                    continue
                instance["image_path"] = image_path
            self.instances.append(instance)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = instance["conversations"]
        if "image_path" in instance:
            # Remove the image token from the messages
            for message in instance["conversations"]:
                message["value"] = message["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()

            # Add image to the first message
            image = Image(instance["image_path"])
            messages[0]["value"] = [image, messages[0]["value"]]
        return messages