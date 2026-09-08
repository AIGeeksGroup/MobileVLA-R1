"""Validate/normalize existing CoT annotations without generating data or loading models."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobilevla.data import load_records, sft_target
from mobilevla.actions import format_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', choices=['episode', 'navigation', 'control'], required=True)
    parser.add_argument('--limit', type=int, help='Deterministic subset size, e.g. 10000 for the Step cold start')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()
    normalized = []
    for index, record in enumerate(load_records(args.input)):
        try:
            if record.get('split', 'train') not in ('train', 'training'):
                raise ValueError('Only training-split records may be used for SFT')
            record = dict(record)
            record['task_type'] = args.task
            answer, task = sft_target(record)
            if task != args.task:
                raise ValueError(f'Expected {args.task} but annotation contains {task}')
            if task != 'episode' and not format_valid(answer, task):
                raise ValueError('Target requires nonempty reasoning and a valid final answer')
            images = record.get('images', record.get('frames'))
            instruction = record.get('instruction', record.get('q'))
            if not isinstance(images, list) or not images or not all(isinstance(x, str) for x in images):
                raise ValueError('Expected a nonempty list of RGB paths')
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('Missing instruction')
            record.update(images=images, instruction=instruction, a=answer)
            record.pop('frames', None)
            normalized.append(record)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid annotation {index}: {exc}') from exc
    if args.limit is not None:
        if not 0 < args.limit <= len(normalized):
            raise ValueError('Requested subset exceeds available validated records')
        normalized = random.Random(args.seed).sample(normalized, args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for record in normalized:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'Wrote {len(normalized)} validated {args.task} records to {output}')


if __name__ == '__main__':
    main()
