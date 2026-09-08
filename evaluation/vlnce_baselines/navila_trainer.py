import copy
import gc
import json
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs
from habitat_extensions.utils import generate_video, observations_to_image
from PIL import Image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.data.nav_cot_utils import depth_to_point_cloud
from mobilevla.actions import habitat_action, parse_answer, serialize_answer
from mobilevla.data import observation_question, policy_question


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    frames = copy.deepcopy(images)

    if len(frames) < num_frames:
        padding_frames = num_frames - len(frames)
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    else:
        padding_frames = 0

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    return sampled_frames


def mobilevla_payload(model, rgb_tensor, rgb, observation, config, depth_provider=None):
    payload = {"rgb": rgb_tensor, "token_types": ["rgb"] * rgb_tensor.shape[0]}
    needs_depth = getattr(model, "depth_tower", None) is not None
    needs_point = getattr(model, "point_tower", None) is not None
    if needs_depth or needs_point:
        source = config.MOBILEVLA.DEPTH_SOURCE
        if source == "simulator":
            if "depth" not in observation:
                raise ValueError("Simulator observation does not contain depth")
            depth = torch.as_tensor(observation["depth"]).float().squeeze(-1)
            sensor = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
            if sensor.NORMALIZE_DEPTH:
                depth = depth * (sensor.MAX_DEPTH - sensor.MIN_DEPTH) + sensor.MIN_DEPTH
        elif source == "provider" and depth_provider is not None:
            depth = torch.as_tensor(depth_provider(np.asarray(rgb))).float()
        else:
            raise ValueError("Depth/point policy requires an explicit simulator or RGB depth provider")
        if depth.ndim != 2 or not torch.isfinite(depth).all():
            raise ValueError("Depth provider must return a finite H x W map")
        depth = depth.unsqueeze(0).unsqueeze(0)
        if needs_depth:
            payload["depth"] = depth
            payload["token_types"].append("depth")
        if needs_point:
            payload["point"] = depth_to_point_cloud(depth, config.MOBILEVLA.POINT_COUNT).unsqueeze(0)
            payload["token_types"].append("point")
    return payload


@baseline_registry.register_trainer(name="navila")
class NaVILATrainer(BaseVLNCETrainer):
    def __init__(self, config=None, num_chunks=1, chunk_idx=0):
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

        super().__init__(config)

    def _make_dirs(self) -> None:
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        # build model
        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        tokenizer, model, image_processor, context_len = load_pretrained_model(checkpoint_path, model_name)
        model = model.cuda().eval()
        depth_provider = None
        if self.config.MOBILEVLA.ENABLED and self.config.MOBILEVLA.DEPTH_SOURCE == "provider":
            import importlib
            module, separator, name = self.config.MOBILEVLA.DEPTH_PROVIDER.partition(":")
            if not separator:
                raise ValueError("Set MOBILEVLA.DEPTH_PROVIDER to module:function")
            depth_provider = getattr(importlib.import_module(module), name)

        config = self.config.clone()
        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = self.num_chunks
        config.TASK_CONFIG.DATASET.CHUNK_IDX = self.chunk_idx
        config.RESULTS_DIR = os.path.join(
            config.RESULTS_DIR, model_name, config.TASK_CONFIG.DATASET.TYPE, config.TASK_CONFIG.DATASET.SPLIT
        )
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"{split}_{self.num_chunks}-{self.chunk_idx}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observations = envs.reset()
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_episodes = {}

        past_rgbs = [[] for _ in range(envs.num_envs)]
        rgb_frames = [[] for _ in range(envs.num_envs)]  # this is for visualization, contains text and map

        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_path}]" " [Episodes evaluated: {evaluated}/{total}]" " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        assert envs.num_envs == 1

        queue_actions = []

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:

            current_episodes = envs.current_episodes()

            if len(queue_actions) > 0:
                print(f"using queue...{queue_actions[0]}")
                outputs = envs.step([queue_actions[0]])
                queue_actions.pop(0)
                print(f"queue length after using...{len(queue_actions)}")

            else:
                with torch.no_grad():
                    curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")

                    past_and_current_rgbs = past_rgbs[0] + [curr_rgb]
                    num_video_frames = model.config.num_video_frames

                    past_and_current_rgbs = sample_and_pad_images(past_and_current_rgbs, num_frames=num_video_frames)

                    instruction = current_episodes[0].instruction.instruction_text

                    interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)

                    frame_length = len(past_and_current_rgbs)
                    print(f"input frame length {frame_length}")

                    question = (
                        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                        f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                        f"degree, moving forward a certain distance, or stop if the task is completed."
                    )

                    images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(
                        model.device, dtype=model.dtype
                    )
                    image_payload = images_tensor
                    if config.MOBILEVLA.ENABLED:
                        image_payload = mobilevla_payload(
                            model, images_tensor, curr_rgb, observations[0], config, depth_provider
                        )
                        question = observation_question(
                            policy_question(instruction, "navigation"), image_payload["token_types"]
                        )

                    conv_mode = "llama_3"
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_payload,
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=config.MOBILEVLA.MAX_NEW_TOKENS if config.MOBILEVLA.ENABLED else 32,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()

                    if outputs.endswith(stop_str):
                        outputs = outputs[: -len(stop_str)]
                    outputs = outputs.strip()
                    print(outputs)

                    if config.MOBILEVLA.ENABLED:
                        try:
                            action_id, _ = habitat_action(outputs)
                            # Downstream distance/angle handling sees only the final
                            # answer; reasoning keywords cannot trigger a command.
                            outputs = serialize_answer(parse_answer(outputs))
                            actions = [action_id]
                        except ValueError as exc:
                            logger.warning(f"Invalid MobileVLA navigation answer; stopping episode: {exc}")
                            actions = [0]
                            outputs = "stop"
                    else:
                        patterns = {
                            0: re.compile(r"\bstop\b", re.IGNORECASE),
                            1: re.compile(r"\bis move forward\b", re.IGNORECASE),
                            2: re.compile(r"\bis turn left\b", re.IGNORECASE),
                            3: re.compile(r"\bis turn right\b", re.IGNORECASE),
                        }
                        actions = [next((action for action, pattern in patterns.items() if pattern.search(outputs)), 0)]

                if actions[0] == 1:
                    try:
                        match = re.search(r"move forward (\d+) cm", outputs)
                        distance = int(match.group(1))
                    except:
                        distance = 25
                    if (distance % 25) != 0:
                        distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                    if config.MOBILEVLA.ENABLED:
                        _, repeats = habitat_action(outputs)
                        distance = repeats * 25
                    outputs = envs.step([1])

                    for _ in range(int(distance // 25) - 1):
                        queue_actions.append(1)

                elif actions[0] == 2:
                    try:
                        match = re.search(r"turn left (\d+) degree", outputs)
                        degree = int(match.group(1))
                    except:
                        degree = 15
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    if config.MOBILEVLA.ENABLED:
                        _, repeats = habitat_action(outputs)
                        degree = repeats * 15
                    outputs = envs.step([2])

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(2)
                    print(f"queue length: {len(queue_actions)}")

                elif actions[0] == 3:
                    try:
                        match = re.search(r"turn right (\d+) degree", outputs)
                        degree = int(match.group(1))
                    except:
                        degree = 15
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    if config.MOBILEVLA.ENABLED:
                        _, repeats = habitat_action(outputs)
                        degree = repeats * 15
                    outputs = envs.step([3])

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(3)

                else:  # 0, stop
                    outputs = envs.step(actions)

            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                past_rgbs[i].append(Image.fromarray(batch[0]["rgb"].cpu().numpy()).convert("RGB"))

                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(frame, current_episodes[i].instruction.instruction_text)
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                past_rgbs[i] = []

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx="0",
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (envs, batch, rgb_frames,) = self._pause_envs(
                envs_to_pause,
                envs,
                batch,
                rgb_frames,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        batch,
        rgb_frames=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            batch,
            rgb_frames,
        )

    def eval(self) -> None:
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu")
        )
        if "tensorboard" in self.config.VIDEO_OPTION:
            assert len(self.config.TENSORBOARD_DIR) > 0, "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert len(self.config.VIDEO_DIR) > 0, "Must specify a directory for storing videos on disk"

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            if os.path.isdir(self.config.EVAL_CKPT_PATH_DIR):
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                )
