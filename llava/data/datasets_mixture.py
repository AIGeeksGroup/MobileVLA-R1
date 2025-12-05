# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS_LEGACY = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS_LEGACY:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS_LEGACY.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():

    video_chatgpt = Dataset(
        dataset_name="video_chatgpt",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/videos",
    )
    add_dataset(video_chatgpt)

    sharegpt_video = Dataset(
        dataset_name="sharegpt_video",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/videos",
    )
    add_dataset(sharegpt_video)

    sharegpt4v_sft = Dataset(
        dataset_name="sharegpt4v_sft",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/videos",
    )
    add_dataset(sharegpt4v_sft)

    envdrop = Dataset(
        dataset_name="envdrop",
        dataset_type="envdrop",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/EnvDrop/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/EnvDrop/videos",
        description="VLN_CE Envdrop.",
    )
    add_dataset(envdrop)

    scanqa = Dataset(
        dataset_name="scanqa",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/ScanQA/annotations/ScanQA_v1.0_train_reformat.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/ScanQA/videos",
        description="ScanQA training set.",
    )
    add_dataset(scanqa)

    r2r = Dataset(
        dataset_name="r2r",
        dataset_type="vlnce",
        data_path="/root/autodl-tmp/dataset/R2R/annotations.json",
        image_path="/root/autodl-tmp/dataset/R2R/train",
        description="350K VLN-CE R2R data. (augmented aith duplicate samples)",
    )
    add_dataset(r2r)

    rxr = Dataset(
        dataset_name="rxr",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/RxR/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/RxR/train",
        description="400K RxR data. (augmented aith duplicate stops only - 5x)",
    )
    add_dataset(rxr)

    human = Dataset(
        dataset_name="human",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/Human/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/Human/raw_frames/",
        description="560K Real augmented, no direction is included. (augmented aith duplicate stops only - 5x)",
    )
    add_dataset(human)

    cot = Dataset(
        dataset_name="cot",
        dataset_type="vlnce",
        data_path="/root/autodl-tmp/dataset/CoT/annotations.json",
        image_path="/root/autodl-tmp/dataset/CoT/train/",
        description="560K Real augmented, no direction is included. (augmented aith duplicate stops only - 5x)",
    )
    add_dataset(cot)

    nav_cot_vln = Dataset(
        dataset_name="nav_cot_vln",
        dataset_type="vlnce",
        data_path="Nav_CoT_FINAL_38K.jsonl",
        image_path=None,
        description="Navigation CoT data built on MP3D scenes with multi-sensor cues.",
    )
    add_dataset(nav_cot_vln)