import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
import PIL.Image
from decord import VideoReader

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN
from llava.media import Image, Video
from llava.train.args import DataArguments, TrainingArguments

__all__ = ["extract_media"]


Config = Union[DataArguments, TrainingArguments]


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        image = PIL.Image.open(image.path)
    return image


def _extract_video(video: Video, config: Config) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    if getattr(config, "fps") != 0:
        raise NotImplementedError("Extracting frames from video with specified FPS is not supported yet")

    if os.path.isdir(video.path):
        frame_paths = sorted(glob.glob(os.path.join(video.path, "*")))
        idx = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        frame_paths = list(np.array(frame_paths)[idx])
        frames = [PIL.Image.open(frame_path) for frame_path in frame_paths]
    else:
        video_reader = VideoReader(uri=video.path)
        idx = np.round(np.linspace(0, len(video_reader) - 1, num_frames)).astype(int)
        frames = video_reader.get_batch(idx).asnumpy()
        frames = [PIL.Image.fromarray(frame) for frame in frames]

    if not frames:
        raise ValueError(f"Video `{video.path}` has no frames")
    return frames


def extract_media(messages: List[Dict[str, Any]], config: Config) -> Dict[str, List[Any]]:
    # TODO(zhijianl): This logic will be moved to model forward function
    if config.mm_use_im_start_end:
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        image_token = DEFAULT_IMAGE_TOKEN

    media = defaultdict(list)
    for message in messages:
        prompt = message["value"]
        if isinstance(prompt, str):
            prompt = [prompt]
        text = ""
        for part in prompt:
            if isinstance(part, str):
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                image = _extract_image(part)
                text += image_token + "\n"
                media["image"].append(image)
            elif isinstance(part, Video):
                video = _extract_video(part, config)
                text += (image_token + "\n") * len(video)
                media["image"].extend(video)
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media
