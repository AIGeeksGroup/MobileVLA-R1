from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, time, requests
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from tqdm import tqdm
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Training configuration
model_path = "/root/autodl-tmp/model/finetune/llm"
beta = 0.04
all_steps = 500
train_batch_size = 2
gen_update_steps = 16
save_steps = 2
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"

# Directory used to exchange model checkpoints with the generation worker
model_update_dir = "/root/autodl-tmp/model_updates"
os.makedirs(model_update_dir, exist_ok=True)

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def GRPO_step(batch, engine, tokenizer):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)

    # === Debug 1: raw rewards ===
    print(">>> rewards:", 
          "min =", batch['rewards'].min().item(), 
          "max =", batch['rewards'].max().item(), 
          "mean =", batch['rewards'].float().mean().item())

    # === Debug 2: normalized advantages ===
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False) + 1e-8
    advantages = (advantages - adv_mean) / adv_std
    print(">>> advantages (normalized):", 
          "min =", advantages.min().item(), 
          "max =", advantages.max().item(), 
          "mean =", advantages.mean().item())

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length-1:]

    # === Debug 3: per-token logps ===
    print(">>> per_token_logps:", 
          "min =", per_token_logps.min().item(), 
          "max =", per_token_logps.max().item(), 
          "mean =", per_token_logps.mean().item())

    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    # === Debug 4: ref logps ===
    print(">>> ref_per_token_logps:", 
          "min =", ref_per_token_logps.min().item(), 
          "max =", ref_per_token_logps.max().item(), 
          "mean =", ref_per_token_logps.mean().item())

    per_token_kl = 0.5 * (per_token_logps - ref_per_token_logps) ** 2
    print(">>> per_token_kl:", 
          "min =", per_token_kl.min().item(), 
          "max =", per_token_kl.max().item(), 
          "mean =", per_token_kl.mean().item())

    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    print(">>> completion_mask sum:", completion_mask.sum(dim=1))

    if 'gen_logps' in batch:
        logp_diff = (per_token_logps - batch['gen_logps'].to(engine.device))
        print("logp_diff: min =", logp_diff.min().item(), "max =", logp_diff.max().item(), "mean =", logp_diff.mean().item())
        logp_diff = torch.clamp(logp_diff, -10, 10)  # Prevent exp from blowing up
        ratio = torch.exp(logp_diff)
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False

    # === Debug 5: per-token loss before mask ===
    print(">>> per_token_loss:", 
          "min =", per_token_loss.min().item(), 
          "max =", per_token_loss.max().item(), 
          "mean =", per_token_loss.mean().item())

    per_token_loss = -(per_token_loss - beta * per_token_kl)

    # === Debug 6: per-token loss after KL ===
    print(">>> per_token_loss (after KL):", 
          "min =", per_token_loss.min().item(), 
          "max =", per_token_loss.max().item(), 
          "mean =", per_token_loss.mean().item())

    lengths = completion_mask.sum(dim=1).clamp(min=1)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / lengths).mean()

    print(">>> final loss:", loss.item())
    print("="*60)

    return loss

if __name__ == '__main__':
    deepspeed.init_distributed()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: 
        progress = tqdm(progress)
    
    for step in progress:
        batch = get_batch()
        while batch is None:
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch()

        loss = GRPO_step(batch, engine, tokenizer)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] Saving model for gen_worker ...')
                state_dict = engine.module.state_dict()
                update_file = os.path.join(model_update_dir, "latest_model.pt")
                temp_file = update_file + ".tmp"
                
                # First save to a temporary file
                torch.save(state_dict, temp_file)
                
                # Atomically move the file into place
                if os.path.exists(update_file):
                    os.remove(update_file)
                os.rename(temp_file, update_file)
                
                # Create a flag file to signal that an update is ready
                flag_file = os.path.join(model_update_dir, "update_ready.flag")
                open(flag_file, 'w').close()
                
                print('[TRAINING PROC] Model saved for update')
                del state_dict
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"/root/autodl-tmp/model/step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()