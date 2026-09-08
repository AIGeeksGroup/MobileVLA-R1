"""Generate complete, versioned GRPO groups from one multimodal policy."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import requests
import torch
from torch.nn.utils.rnn import pad_sequence

from inference import NaVILAImageInference
from mobilevla.actions import reward
from mobilevla.checkpoints import require_auxiliary_checkpoint
from mobilevla.data import load_records, normalize_path, policy_question, record_target, resolve_depth_path
from mobilevla.grpo import PROTOCOL_VERSION, completion_logps, dumps, group_advantages, to_cpu, validate_batch


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=os.getenv('MODEL_PATH'), required=not bool(os.getenv('MODEL_PATH')))
    parser.add_argument('--dataset', default=os.getenv('GEN_DATASET'), required=not bool(os.getenv('GEN_DATASET')))
    parser.add_argument('--image-root', default=os.getenv('NAVCOT_IMAGE_ROOT'))
    parser.add_argument('--depth-root', default=os.getenv('NAVCOT_DEPTH_ROOT'))
    parser.add_argument('--depth-format', default=os.getenv('NAVCOT_DEPTH_FORMAT', 'png'))
    parser.add_argument('--depth-scale', type=float, default=float(os.getenv('NAVCOT_DEPTH_SCALE', '1000')))
    parser.add_argument('--depth-frames', type=int, default=1)
    parser.add_argument('--point-cloud', action=argparse.BooleanOptionalAction, default=os.getenv('NAVCOT_USE_POINT', '1') == '1')
    parser.add_argument('--point-count', type=int, default=2048)
    parser.add_argument('--group-size', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--server', default='http://localhost:59875')
    parser.add_argument('--updates', default='./model_updates')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    if args.group_size < 2 or args.depth_frames < 1 or args.point_count < 1:
        parser.error('group-size must be >= 2; frame/point counts must be positive')
    return args


def make_samples(args):
    samples = []
    source = Path(args.dataset)
    for index, record in enumerate(load_records(source)):
        try:
            _, target = record_target(record)
            frames = record.get('images', record.get('frames', []))
            if not frames:
                raise ValueError('No RGB frames')
            image_root = args.image_root
            if image_root is None and source.is_dir():
                image_root = str(source / 'train')
            images = [normalize_path(frame, image_root) for frame in frames]
            depth = None
            if args.depth_root:
                depth = [resolve_depth_path(frame, image_root, args.depth_root, args.depth_format)
                         for frame in frames[-args.depth_frames:]]
            for path in images + (depth or []):
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            task = 'control' if target.velocity is not None else 'navigation'
            samples.append({'question': policy_question(record.get('instruction', record.get('q')), task),
                            'target': target, 'task': task, 'images': images, 'depth': depth})
        except (ValueError, TypeError, FileNotFoundError) as exc:
            raise ValueError(f'Invalid annotation {index} in {source}: {exc}') from exc
    return samples


def refresh_policy(inferencer, updates, current, model_id):
    manifest = Path(updates) / 'latest.json'
    if not manifest.is_file():
        return current
    status = json.loads(manifest.read_text())
    if status['model_id'] != model_id:
        raise ValueError('Trainer and generator must use the same initial full model directory')
    identity = (status['run_id'], status['version'])
    if identity == current:
        return current
    snapshot = torch.load(Path(updates) / 'latest_model.pt', map_location='cpu', weights_only=True)
    if (snapshot['run_id'], snapshot['version']) != identity:
        return current  # Atomic publication advanced between the two reads.
    inferencer.model.load_state_dict(snapshot['state_dict'], strict=True)
    inferencer.model.eval()
    print(f'Loaded sampling policy {identity}', flush=True)
    return identity


def make_group(inferencer, sample, args, identity, model_id):
    # Match SFT temporal sampling; point sampling and image preparation happen
    # once per group and those exact tensors are uploaded to the other policies.
    count = inferencer.model.config.num_video_frames or 8
    frames = sample['images']
    if len(frames) < count:
        frames = frames + [frames[-1]] * (count - len(frames))
    indices = [int(i * (len(frames)-1) / (count-1)) if count > 1 else len(frames)-1 for i in range(count)]
    prepared = inferencer.prepare_observation(
        [frames[i] for i in indices], sample['question'], sample['depth'],
        'from_depth' if args.point_cloud and sample['depth'] else None,
    )
    answers, scores = [], []
    for _ in range(args.group_size):
        response, ids = inferencer.sample_prepared(prepared, max_new_tokens=args.max_new_tokens,
                                                   temperature=1.0, top_p=1.0, do_sample=True)
        answers.append(ids)
        scores.append(reward(response, sample['target'], sample['task'])['total'])
    lengths = torch.tensor([len(ids) for ids in answers])
    completion_ids = pad_sequence(answers, batch_first=True, padding_value=inferencer.tokenizer.pad_token_id)
    mask = torch.arange(completion_ids.shape[1])[None, :] < lengths[:, None]
    batch = {'protocol': PROTOCOL_VERSION, 'run_id': identity[0], 'policy_version': identity[1],
             'model_id': model_id, 'prompt_ids': to_cpu(prepared['input_ids']),
             'images': to_cpu(prepared['images']), 'completion_ids': completion_ids,
             'completion_mask': mask, 'advantages': group_advantages(torch.tensor(scores))}
    with torch.no_grad():
        batch['old_logps'] = completion_logps(inferencer.model, batch).cpu()
    validate_batch(batch)
    return batch


def main():
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = make_samples(args)
    inferencer = NaVILAImageInference(args.model_path, device=args.device,
                                     depth_scale=args.depth_scale, pointcloud_points=args.point_count)
    require_auxiliary_checkpoint(inferencer.model)
    model_id = str(Path(args.model_path).resolve())
    if inferencer.model.depth_tower is not None and not args.depth_root:
        raise ValueError('This policy has a depth encoder; supply --depth-root')
    if inferencer.model.point_tower is not None and (not args.depth_root or not args.point_cloud):
        raise ValueError('This policy has a point encoder; enable depth-derived point clouds')
    current = None
    while True:
        current = refresh_policy(inferencer, args.updates, current, model_id)
        if current is None:
            time.sleep(1)
            continue
        batch = make_group(inferencer, random.choice(samples), args, current, model_id)
        if not batch['advantages'].any():
            continue
        payload = dumps(batch)
        while True:
            response = requests.post(args.server + '/upload', data=payload, timeout=120)
            if response.status_code != 429:
                response.raise_for_status()
                break
            time.sleep(1)


if __name__ == '__main__':
    main()
