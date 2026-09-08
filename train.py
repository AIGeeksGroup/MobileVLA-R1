"""GRPO over complete multimodal response groups; no work occurs on import."""
import argparse
import json
import os
import time
import uuid
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=os.getenv('MODEL_PATH'), required=not bool(os.getenv('MODEL_PATH')))
    parser.add_argument('--output-dir', default='./checkpoints/grpo')
    parser.add_argument('--updates', default='./model_updates')
    parser.add_argument('--server', default='http://localhost:59875')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--group-size', type=int, default=3)
    parser.add_argument('--gradient-accumulation', type=int, default=2)
    parser.add_argument('--update-every', type=int, default=16)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--beta', type=float, default=0.04)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.gradient_accumulation, args.update_every, args.save_every) < 1 or args.group_size < 2:
        parser.error('Steps/intervals must be positive and group-size must be >= 2')
    return args


def publish(model, directory, run_id, version, model_id):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    snapshot = directory / 'latest_model.pt'
    temporary = directory / 'latest_model.pt.tmp'
    torch.save({'state_dict': state, 'run_id': run_id, 'version': version}, temporary)
    os.replace(temporary, snapshot)
    manifest = directory / 'latest.json.tmp'
    manifest.write_text(json.dumps({'run_id': run_id, 'version': version, 'model_id': model_id}))
    os.replace(manifest, directory / 'latest.json')


def main():
    args = arguments()
    # The shared queue serves one trainer. Multi-rank support needs coordinated
    # dispatch/checkpoint collection and must not silently start independent loops.
    if int(os.getenv('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use deepspeed --num_gpus=1; this group queue supports one trainer rank')
    import requests
    import torch
    import deepspeed
    from inference import NaVILAImageInference
    from mobilevla.checkpoints import require_auxiliary_checkpoint
    from mobilevla.grpo import completion_logps, grpo_loss, loads, validate_batch

    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()
    model = NaVILAImageInference(args.model_path, device=f'cuda:{args.local_rank}').model
    require_auxiliary_checkpoint(model)
    model.requires_grad_(True)
    # Evaluation mode disables dropout and running-stat changes, not gradients.
    model.eval()
    config = {'train_micro_batch_size_per_gpu': args.group_size,
              'gradient_accumulation_steps': args.gradient_accumulation,
              'optimizer': {'type': 'AdamW', 'params': {'lr': args.learning_rate}},
              'bf16': {'enabled': True}, 'gradient_clipping': 1.0,
              'zero_optimization': {'stage': 2, 'offload_optimizer': {'device': 'cpu'}}}
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    run_id, version = uuid.uuid4().hex, 0
    model_id = str(Path(args.model_path).resolve())
    publish(engine.module, args.updates, run_id, version, model_id)
    completed = 0
    while completed < args.steps:
        response = requests.get(args.server + '/get', params={'run_id': run_id, 'version': version}, timeout=120)
        response.raise_for_status()
        if response.status_code == 204:
            time.sleep(1)
            continue
        batch = loads(response.content)
        validate_batch(batch)
        if batch['run_id'] != run_id or batch['policy_version'] != version or batch.get('model_id') != model_id:
            raise ValueError('Received group from a different policy/run')
        if batch['completion_ids'].shape[0] != args.group_size:
            raise ValueError('Trainer and worker group-size must agree')
        current = completion_logps(engine.module, batch, forward=engine)
        loss = grpo_loss(current, batch, args.beta, args.clip)
        engine.backward(loss)
        boundary = engine.is_gradient_accumulation_boundary()
        engine.step()
        if not boundary:
            continue
        completed += 1
        print(f'Optimizer step {completed}: loss={loss.item():.6f}', flush=True)
        if completed % args.update_every == 0:
            version = completed
            publish(engine.module, args.updates, run_id, version, model_id)
        if completed % args.save_every == 0 or completed == args.steps:
            engine.module.save_pretrained(str(Path(args.output_dir) / f'step_{completed}'))


if __name__ == '__main__':
    main()
