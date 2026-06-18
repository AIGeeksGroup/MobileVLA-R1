# <img src="./assets/mobilevlar1_logo.png" alt="logo" width="40"/> MobileVLA-R1: Reinforcing Vision-Language-Action for Mobile Robots

This is the official repository for the paper:
> **MobileVLA-R1: Reinforcing Vision-Language-Action for Mobile Robots**
>
> [Ting Huang](https://github.com/Believeht029)\*, [Dongjian Li]()\*, [Rui Yang]()\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, [Zida Yang](), and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> ***ECCV 2026***
> 
> ### [Paper](https://arxiv.org/abs/2511.17889) | [Website](https://aigeeksgroup.github.io/MobileVLA-R1/) | [Data](https://huggingface.co/datasets/AIGeeksGroup/MobileVLA-CoT) | [Models](https://huggingface.co/AIGeeksGroup/MobileVLA-R1) | [HF Paper](https://huggingface.co/papers/2511.17889)



https://github.com/user-attachments/assets/b167ebe6-cd72-470f-9b54-07e6e0989a4e



## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{huang2025mobilevla,
  title={MobileVLA-R1: Reinforcing Vision-Language-Action for Mobile Robots},
  author={Huang, Ting and Li, Dongjian and Yang, Rui and Zhang, Zeyu and Yang, Zida and Tang, Hao},
  journal={arXiv preprint arXiv:2511.17889},
  year={2025}
}
```

---

## 🏃 Intro MobileVLA-R1
MobileVLA-R1 enables robust real-world quadruped control by unifying language reasoning and continuous action through structured CoT alignment and GRPO training.

Grounding natural-language instructions into continuous control for quadruped robots remains a fundamental challenge in vision language action.
Existing methods struggle to bridge high-level semantic reasoning and low-level actuation, leading to unstable grounding and weak generalization in the real-world.
To address these issues, we present MobileVLA-R1, a unified vision–language–action framework that enables explicit reasoning and continuous control for quadruped robots.
We construct MobileVLA-CoT, a large-scale dataset of multi-granularity CoT for embodied trajectories, providing structured reasoning supervision for alignment.
Built upon this foundation, we introduce a two-stage training paradigm that combines supervised CoT alignment with GRPO reinforcement learning to enhance reasoning consistency, control stability, and long-horizon execution.
Extensive evaluations on VLN and VLA tasks demonstrate superior performance over strong baselines, with approximately a 5\% improvement.
Real-world deployment on a quadruped robot validates robust performance in complex environments.

![image](./assets/structure.png)

## 📰 News

<b>2025/12/05:</b> 📣 Our paper has been promoted by <a href="https://mp.weixin.qq.com/s/d9y8Rchx7ZHqfIEIwfmy4A"><b>AI Era</b></a>.

<b>2025/11/30:</b> 🔔 Our paper has been promoted by <a href="https://mp.weixin.qq.com/s/xwNx1-yGbCOwUiVjJ3_IKA"><b>Embodied Intelligent Mind</b></a>.

<b>2025/11/27:</b> 🎉 Our paper has been shared by <a href="https://x.com/_akhaliq/status/1993983918551322807"><b>AK</b></a>.

## TODO List

- [x] Upload our paper to arXiv and build project pages.
- [x] Release MobileVLA-CoT dataset.
- [x] Upload the code.

## 📦 Data Preparation

Our pipeline expects three synchronized modalities per observation: RGB frames (MP3D skybox crops), Depth Anything v2 maps, and point clouds derived from the depth maps. The default dataset used in this repo is `Nav_CoT_FINAL_38K.jsonl`, which augments R2R/RxR trajectories with CoT reasoning.

1. **Download CoT annotations**
   ```bash
   wget https://your-storage/Nav_CoT_FINAL_38K.jsonl -O ./Nav_CoT_FINAL_38K.jsonl
   ```
2. **Extract RGB frames**
   - Clone Matterport3D scans or reuse the official MP3D release.
   - Create a root folder (e.g. `/root/autodl-tmp/dataset/NavCoT/frames`) that mirrors the path structure in the JSONL file. The loader automatically rewrites Windows-style paths via `navcot_image_root`.
3. **Generate Depth Anything v2 maps**
   - Run Depth Anything v2 on each RGB frame and save the outputs (`png`, `npy`, or `pt`) under `/root/autodl-tmp/dataset/NavCoT/depth`.
   - Name each depth file with the same basename as its RGB frame so the loader can resolve it.
4. **(Optional) Pre-compute point clouds**
   - The training code can derive point clouds on the fly from depth maps. If you have higher-quality `.npy` point sets, place them next to your depth files and pass their path through `navcot_use_point`.

Key CLI arguments controlling the data loader live in `llava/train/args.py`, notably:
- `--navcot_image_root`, `--navcot_depth_root`, `--navcot_depth_format`, `--navcot_depth_scale`
- `--navcot_use_depth`, `--navcot_use_point`, `--navcot_pointcloud_points`, `--navcot_depth_frames`

These options also apply to GRPO generation via environment variables (see below).



## ⚙️ Environment Setup

The repo ships with a helper script that creates a compatible Conda environment, installs CUDA/FlashAttention2, links Hugging Face Transformers replacements, and installs all train/eval extras. Run it once per machine:

```bash
# Optional: pass a conda env name, otherwise it assumes you already activated one.
bash environment_setup.sh mobilevla
```

The script will:

1. Create/activate a Python 3.10 environment (if you passed a name).
2. Upgrade `pip`, install `cuda-toolkit`, FlashAttention 2.5.8 (CUDA 12.2, torch 2.3 build), and this project in editable mode with `[train]` / `[eval]` extras.
3. Pull `transformers==4.37.2` from source and copy our patched files from `llava/train/{transformers_replace,deepspeed_replace}` into your site-packages so the long-context + sequence-parallel features work.

If you manage environments manually, replicate the same steps (torch 2.3 + CUDA 12.2, FlashAttention2, transformers 4.37.2 with the provided patches) before launching the SFT/GRPO jobs.


## 🚀 Training

### Stage 1: Supervised CoT Alignment (SFT)

Run the provided LoRA script, which already enables the depth/point encoders and mixes MobileVLA-CoT with the Nav-CoT JSONL:

```bash
WANDB_MODE=offline \
bash scripts/train/sft_8frames.sh \
  --data_mixture cot+nav_cot_vln \
  --navcot_image_root /root/autodl-tmp/dataset/NavCoT/frames \
  --navcot_depth_root /root/autodl-tmp/dataset/NavCoT/depth \
  --navcot_use_depth True \
  --navcot_use_point True
```

Feel free to override any argument defined inside `scripts/train/sft_8frames.sh` for your cluster (batch size, LoRA rank, dataset paths, etc.).

### Stage 2: GRPO Reinforcement Learning

1. **Reference model server**
   ```bash
   python ref_server.py
   ```
2. **Policy fine-tuning with DeepSpeed**
   ```bash
   deepspeed train.py
   ```
3. **Generative worker with multi-modal inputs**
   ```bash
   GEN_DATASET=/root/autodl-tmp/Nav_CoT_FINAL_38K.jsonl \
   NAVCOT_IMAGE_ROOT=/root/autodl-tmp/dataset/NavCoT/frames \
   NAVCOT_DEPTH_ROOT=/root/autodl-tmp/dataset/NavCoT/depth \
   NAVCOT_USE_POINT=1 \
   python gen_worker.py
   ```

The worker instantiates `NaVILAImageInference`, streams RGB/Depth/Point payloads into the policy, scores candidates via the reward calculator, and pushes normalized rewards plus log-probs back to the reference server. `train.py` consumes those batches and performs GRPO updates with periodic model refreshes for the generator.





## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIGeeksGroup/MobileVLA-R1&type=date&legend=top-left)](https://www.star-history.com/#AIGeeksGroup/MobileVLA-R1&type=date&legend=top-left)

## 😘 Acknowledgement
We thank the authors of [Qwen](https://github.com/QwenLM/Qwen), [NaVILA](https://github.com/AnjieCheng/NaVILA) and [DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math) for their open-source code.
