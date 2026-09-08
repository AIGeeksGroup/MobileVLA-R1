"""Annotation and path handling without importing the model runtime."""
import json
from pathlib import Path, PurePosixPath

from .actions import normalize_target, NAV_PROMPT, CONTROL_PROMPT


def normalize_path(raw_path, root=None):
    path = PurePosixPath(str(raw_path).replace('\\', '/'))
    if root is None:
        return str(path)
    root_path = PurePosixPath(str(root).replace('\\', '/'))
    # Already-resolved paths must be idempotent.
    if path.is_relative_to(root_path):
        return str(path)
    parts = path.parts
    if 'scans' in parts:
        parts = parts[parts.index('scans') + 1:]
    elif parts and parts[0].endswith(':'):
        parts = parts[1:]
    elif path.is_absolute():
        raise ValueError(f'Absolute path {path} is outside {root_path}; use paths relative to the modality root')
    if '..' in parts:
        raise ValueError('Observation paths cannot escape their root')
    return str(root_path.joinpath(*parts))


def load_records(source):
    if source is None:
        raise ValueError('Annotation path is not configured; set the dataset environment variable')
    source = Path(source)
    if source.is_dir():
        source = source / 'annotations.json'
    with source.open(encoding='utf-8') as stream:
        if source.suffix == '.jsonl':
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f'{source} must contain a nonempty list of annotation records')
    return records


def record_target(record):
    if 'a' in record:
        text = record['a']
    else:
        reasoning = record.get('think') or ''
        answer = record.get('answer', record.get('action', record.get('action_label')))
        if answer is None:
            raise ValueError('Annotation requires an answer or action')
        if not isinstance(answer, str):
            answer = json.dumps(answer)
        text = f'<think>{reasoning}</think>\n<answer>{answer}</answer>'
    if not isinstance(text, str):
        text = json.dumps(text)
    return normalize_target(text, action_label=record.get('action_label'))


def policy_question(instruction, task):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Instruction is missing')
    if task == 'episode':
        return (f'You are a quadruped robot. Instruction: {instruction.strip()}\n'
                'Explain the episode-level plan in <think>...</think> and summarize the outcome in <answer>...</answer>.')
    if task not in ('navigation', 'control'):
        raise ValueError(f'Unsupported policy task: {task}')
    return f'You are a quadruped robot. Instruction: {instruction.strip()}\n' + (
        CONTROL_PROMPT if task == 'control' else NAV_PROMPT)


def sft_target(record):
    if record.get('task_type') == 'episode':
        import re
        text = record.get('a')
        if text is None:
            answer = record.get('answer')
            thought = record.get('think')
            if not isinstance(answer, str) or not isinstance(thought, str):
                raise ValueError('Episode samples need think and answer text')
            text = f'<think>{thought}</think>\n<answer>{answer}</answer>'
        pattern = re.compile(r'\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*', re.DOTALL)
        match = pattern.fullmatch(text)
        if not match or not all(x.strip() for x in match.groups()):
            raise ValueError('Episode target must contain nonempty think and answer blocks')
        return text, 'episode'
    text, parsed = record_target(record)
    return text, 'control' if parsed.velocity is not None else 'navigation'


def observation_question(question, token_types):
    names = {'rgb': 'RGB observations', 'depth': 'Depth maps', 'point': 'Point cloud observations'}
    chunks = []
    for token in ('rgb', 'depth', 'point'):
        count = token_types.count(token)
        if count:
            chunks.append(names[token] + ':\n' + '\n'.join(['<image>'] * count))
    if token_types != sorted(token_types, key=lambda x: ('rgb', 'depth', 'point').index(x)):
        raise ValueError('Observation tokens must be ordered RGB, depth, point')
    return '\n'.join(chunks + [question])


def resolve_depth_path(frame, image_root, depth_root, depth_format='png'):
    raw = PurePosixPath(str(frame).replace('\\', '/'))
    if image_root is not None:
        rgb_root = PurePosixPath(str(image_root).replace('\\', '/'))
        if raw.is_relative_to(rgb_root):
            raw = raw.relative_to(rgb_root)
    return str(Path(normalize_path(str(raw), depth_root)).with_suffix('.' + depth_format.lstrip('.')))
