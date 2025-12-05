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
from gym import Space
from habitat import Config, logger
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


def initialize_policy_from_ckpt(config: Config, ckpt_path: str, observation_space: Space, action_space: Space):

    env = get_env_class(config.ENV_NAME)(config=config)

    baseline_policy = baseline_registry.get_policy(config.MODEL.policy_name)
    policy = baseline_policy.from_config(
        config=config,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt_dict["state_dict"])
    logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

    return policy


def sample_and_pad_images(images, num_frames=50, width=512, height=512):
    # images here should contain the current frame
    frames = copy.deepcopy(images)

    if len(frames) < num_frames:
        # print("Padding frames:", num_frames - len(frames))
        padding_frames = num_frames - len(frames)
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    else:
        padding_frames = 0

    latest_frame = frames[-1]
    # use float intervals, and then round to int
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    return sampled_frames


def save_images_to_combined_image(images, output_path, layout="vertical"):
    if layout not in ["vertical", "horizontal"]:
        raise ValueError("Layout must be 'vertical' or 'horizontal'")

    if layout == "vertical":
        # Determine the width and height of the final image
        total_width = max(image.width for image in images)
        total_height = sum(image.height for image in images)

        # Create a new blank image with the determined size
        combined_image = Image.new("RGB", (total_width, total_height))

        # Paste each image into the combined image
        y_offset = 0
        for image in images:
            combined_image.paste(image, (0, y_offset))
            y_offset += image.height
    else:  # horizontal layout
        # Determine the width and height of the final image
        total_width = sum(image.width for image in images)
        total_height = max(image.height for image in images)

        # Create a new blank image with the determined size
        combined_image = Image.new("RGB", (total_width, total_height))

        # Paste each image into the combined image
        x_offset = 0
        for image in images:
            combined_image.paste(image, (x_offset, 0))
            x_offset += image.width

    # Save the combined image
    combined_image.save(output_path)


@baseline_registry.register_trainer(name="my-cma")
class MyCMATrainer(BaseVLNCETrainer):
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
        model_name = "cma"
        # model_name = os.path.basename(os.path.normpath(checkpoint_path))
        # tokenizer, model, image_processor, context_len = load_pretrained_model(checkpoint_path, model_name)
        # model = model.cuda()

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
        # prev_actions = [[] for _ in range(envs.num_envs)] # thsi is for text input to the model

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

            # breakpoint()
            current_episodes = envs.current_episodes()

            if len(queue_actions) > 0:
                print(f"using queue...{queue_actions[0]}")
                outputs = envs.step([queue_actions[0]])
                queue_actions.pop(0)
                print(f"queue length after using...{len(queue_actions)}")

            else:
                with torch.no_grad():
                    # fucked up... training data us using BGR
                    curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")

                    past_and_current_rgbs = past_rgbs[0] + [curr_rgb]
                    # past_and_current_rgbs = sample_frames(past_and_current_rgbs)
                    num_video_frames = model.config.num_video_frames

                    past_and_current_rgbs = sample_and_pad_images(past_and_current_rgbs, num_frames=num_video_frames)
                    # past_and_current_rgbs = sample_and_pad_images_navila(past_and_current_rgbs, num_frames=num_video_frames)

                    instruction = current_episodes[0].instruction.instruction_text

                    interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)

                    frame_length = len(past_and_current_rgbs)
                    print(f"input frame length {frame_length}")

                    # if frame_length == 1:
                    #     question = (
                    #         f'Imagine you are a robot programmed for navigation tasks. You have been given '
                    #         f'current observation <image>\n. Your assigned task is: \"{instruction}\" '
                    #         f'Analyze this series of images to decide your next action, which could be turning left or right by a specific '
                    #         f'degree, moving forward a certain distance, or stop if the task is completed.'
                    #     )
                    # else:
                    #     question = (
                    #         f'Imagine you are a robot programmed for navigation tasks. You have been given a video '
                    #         f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: \"{instruction}\" '
                    #         f'Analyze this series of images to decide your next action, which could be turning left or right by a specific '
                    #         f'degree, moving forward a certain distance, or stop if the task is completed.'
                    #     )
                    question = (
                        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                        f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                        f"degree, moving forward a certain distance, or stop if the task is completed."
                    )

                    conversation = [{"from": "human", "value": question}]

                    generation_config = model.default_generation_config
                    generation_config.max_new_tokens = 32
                    input_ids = (
                        tokenize_conversation(conversation, tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)
                    )

                    # images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(model.device, dtype=torch.float16)
                    images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).half()

                    # breakpoint()

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids=input_ids, images=images_tensor, generation_config=generation_config
                        )

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

                    # breakpoint()
                    # slow_fast_mask = [True]*(frame_length-1) + [False] # True: fast, False: slow
                    # slow_fast_mask = [False]*(frame_length)
                    # slow_fast_mask_tensor = torch.tensor(slow_fast_mask, dtype=torch.bool)
                    # slow_fast_mask_tensor = slow_fast_mask_tensor.view(frame_length, 1)

                    # output = get_model_output(
                    #         model,
                    #         image_processor,
                    #         tokenizer,
                    #         vpath,
                    #         qs,
                    #         conv_mode=args.conv_mode,
                    #         temperature=args.temperature,
                    #         num_beams=args.num_beams,
                    #         num_video_frames=args.num_video_frames,
                    #     )

                    # conv_mode = config.CONV_TYPE
                    # conv = conv_templates[conv_mode].copy()
                    # conv.append_message(conv.roles[0], question)
                    # conv.append_message(conv.roles[1], None)
                    # prompt = conv.get_prompt()

                    # images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(model.device, dtype=torch.float16)
                    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

                    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    # keywords = [stop_str]
                    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    # conversation = [{"from": "human", "value": prompt}]

                    # with torch.inference_mode():
                    #     output_ids = model.generate(
                    #         input_ids,
                    #         images=images_tensor.half().cuda(),
                    #         # slow_fast_masks=slow_fast_mask_tensor.cuda(),
                    #         do_sample=False,
                    #         temperature=0.,
                    #         max_new_tokens=1024,
                    #         use_cache=True,
                    #         stopping_criteria=[stopping_criteria],
                    #         pad_token_id=tokenizer.eos_token_id,
                    #     )
                    # with torch.inference_mode():
                    #     output_ids = model.generate(
                    #         input_ids,
                    #         images=[
                    #             images_tensor.to(model.device, dtype=torch.float16),
                    #         ],
                    #         do_sample=True,
                    #         temperature=0.2,
                    #         max_new_tokens=128,
                    #         use_cache=True,
                    #         stopping_criteria=[stopping_criteria],
                    #         pad_token_id=tokenizer.eos_token_id,
                    #     )

                    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    # outputs = outputs.strip()

                    # # outputs = "The next action is turn left 30 degree."

                    # if outputs.endswith(stop_str):
                    #     outputs = outputs[: -len(stop_str)]
                    outputs = outputs.strip()
                    print(outputs)

                    # Define the regex patterns for each action
                    patterns = {
                        0: re.compile(r"\bstop\b", re.IGNORECASE),
                        1: re.compile(r"\bis move forward\b", re.IGNORECASE),
                        2: re.compile(r"\bis turn left\b", re.IGNORECASE),
                        3: re.compile(r"\bis turn right\b", re.IGNORECASE),
                    }

                    # Function to map a string to an action integer
                    def map_string_to_action(s):
                        for action, pattern in patterns.items():
                            if pattern.search(s):
                                return action
                        return None  # Return None if no match is found

                    actions = [map_string_to_action(outputs)]
                    print(actions)
                    # save_images_to_combined_image(past_and_current_rgbs, "./sanity.png", layout="horizontal")
                    # breakpoint()
                    # if actions[0] is None:
                    #     breakpoint()
                    #     save_images_to_combined_image(past_and_current_rgbs, "./sanity.png", layout="horizontal")
                    # prev_actions.copy_(actions)

                if actions[0] == 1:
                    match = re.search(r"move forward (\d+) cm", outputs)
                    distance = int(match.group(1))
                    # assert (distance % 25) == 0
                    if (distance % 25) != 0:
                        distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                    outputs = envs.step([1])

                    for _ in range(int(distance // 25) - 1):
                        queue_actions.append(1)

                elif actions[0] == 2:
                    match = re.search(r"turn left (\d+) degree", outputs)
                    degree = int(match.group(1))
                    # assert (degree % 15) == 0
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    outputs = envs.step([2])

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(2)
                    print(f"queue length: {len(queue_actions)}")

                elif actions[0] == 3:
                    match = re.search(r"turn right (\d+) degree", outputs)
                    degree = int(match.group(1))
                    # assert (degree % 15) == 0
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    outputs = envs.step([3])

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(3)

                else:  # 0, stop
                    outputs = envs.step(actions)

            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                # save rgb to past observation
                # breakpoint()  batch[0]["rgb"].cpu().numpy()
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
                # prev_actions[i] = []
                # prev_actions[i] = torch.zeros(1, dtype=torch.long)

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
