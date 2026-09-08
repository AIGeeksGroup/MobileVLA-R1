"""Deterministic final-answer parsing; no model or numerical-library imports."""
import ast
import json
import math
import re
from dataclasses import dataclass

ACTIONS = ("go forward", "turn right", "turn left", "stop", "jump", "dance", "hello", "stretch")
STRUCTURE = re.compile(r"\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*", re.DOTALL)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NAV_PROMPT = ('Reason in <think>...</think>, then output <answer>move forward N cm</answer>, '
              '<answer>turn left N degree</answer>, <answer>turn right N degree</answer>, '
              'or <answer>stop</answer>. Put only the next action in the answer.')
CONTROL_PROMPT = ('Reason in <think>...</think>, then output <answer>{"velocity": [vx, vy, yaw], '
                  '"action": "label"}</answer>. Velocities must be finite numbers; action must be one of: '
                  + ', '.join(ACTIONS) + '.')

@dataclass(frozen=True)
class Action:
    action: str
    velocity: tuple = None
    amount: float = None
    unit: str = None


def canonical_action(value):
    if not isinstance(value, str):
        raise ValueError("Action label must be a string")
    label = value.strip().lower()
    label = {"move forward": "go forward", "forward": "go forward", "left": "turn left",
             "right": "turn right", "halt": "stop"}.get(label, label)
    if label not in ACTIONS:
        raise ValueError(f"Unknown action label: {value!r}")
    return label


def _velocity(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("velocity must contain exactly three numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("velocity must contain finite numbers")
    return tuple(float(v) for v in values)


def parse_answer(text, *, legacy_control=False, action_label=None):
    """Parse only the answer, never keywords or numbers from reasoning.

    Legacy 12-number QUARD labels require a separately supplied discrete label.
    They are not silently interpreted as an action class.
    """
    if not isinstance(text, str):
        raise ValueError("Answer must be text")
    matches = ANSWER.findall(text)
    if len(matches) > 1:
        raise ValueError("Expected one answer block")
    content = matches[0].strip() if matches else text.strip()
    if '<' in content or '>' in content:
        raise ValueError("Malformed or unsupported answer tags")
    if content.startswith(('{', '[')):
        try:
            value = json.loads(content)
        except (ValueError, TypeError):
            try:
                value = ast.literal_eval(content)
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Invalid control answer") from exc
        if isinstance(value, dict):
            return Action(canonical_action(value.get('action')), _velocity(value.get('velocity')))
        if isinstance(value, list) and len(value) == 4:
            return Action(canonical_action(value[3]), _velocity(value[:3]))
        if legacy_control and isinstance(value, list) and len(value) == 12 and action_label is not None:
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
                raise ValueError("Invalid legacy control values")
            return Action(canonical_action(action_label), _velocity(value[:3]))
        raise ValueError("Control answers require velocity and an explicit discrete action")
    match = re.fullmatch(r"(?:the next action is |is )?(move forward|go forward|turn left|turn right|stop|jump|dance|hello|stretch)(?:\s+(\d+(?:\.\d+)?)\s*(cm|degree|degrees))?\.?", content.lower())
    if not match:
        # A standalone label is accepted for navigation annotations.
        return Action(canonical_action(content))
    action, amount, unit = match.groups()
    action = canonical_action(action)
    amount = float(amount) if amount else None
    unit = 'degree' if unit in ('degree', 'degrees') else unit
    if amount is not None:
        if amount <= 0 or not math.isfinite(amount):
            raise ValueError("Action magnitude must be positive and finite")
        if (action == 'go forward' and unit != 'cm') or (action.startswith('turn ') and unit != 'degree') or action not in ACTIONS[:3]:
            raise ValueError("Invalid navigation magnitude/unit")
    return Action(action, amount=amount, unit=unit)


def format_valid(text, task):
    match = STRUCTURE.fullmatch(text)
    if match is None or not match['think'].strip() or re.search(r'</?(think|answer)>', match['think']):
        return False
    try:
        parsed = parse_answer(match['answer'])
        return (parsed.velocity is not None) if task == 'control' else parsed.velocity is None
    except ValueError:
        return False


def serialize_answer(parsed):
    if parsed.velocity is not None:
        return json.dumps({'velocity': list(parsed.velocity), 'action': parsed.action})
    label = 'move forward' if parsed.action == 'go forward' else parsed.action
    if parsed.amount is not None:
        label += f' {parsed.amount:g} {parsed.unit}'
    return label


def normalize_target(text, action_label=None):
    """Normalize legacy annotation wrappers without guessing missing labels."""
    reasoning = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer = ANSWER.search(text) or re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    content = answer.group(1) if answer else text
    parsed = parse_answer(content, legacy_control=True, action_label=action_label)
    thought = reasoning.group(1).strip() if reasoning else ''
    return f'<think>{thought}</think>\n<answer>{serialize_answer(parsed)}</answer>', parsed


def reward(response, target, task):
    """Paper rewards for control; navigation uses action and format only."""
    format_r = float(format_valid(response, task))
    try:
        predicted = parse_answer(response)
    except ValueError:
        return {'format': 0.0, 'movement': 0.0, 'action': 0.0, 'total': 0.0}
    action_r = float(predicted.action == target.action)
    movement = 0.0
    if task == 'control':
        if target.velocity is None:
            raise ValueError('Control target has no velocity')
        if predicted.velocity is not None:
            a, b = predicted.velocity, target.velocity
            # Scaling first avoids overflow for large but finite values.
            sa, sb = max(map(abs, a)), max(map(abs, b))
            if sa == 0 or sb == 0:
                movement = float(sa == sb)
            else:
                a, b = [v / sa for v in a], [v / sb for v in b]
                movement = sum(x*y for x,y in zip(a,b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
                movement = max(-1.0, min(1.0, movement))
    return {'format': format_r, 'movement': movement, 'action': action_r,
            'total': format_r + movement + action_r}


def habitat_action(response):
    parsed = parse_answer(response)
    if parsed.velocity is not None:
        raise ValueError('Habitat navigation requires a navigation answer, not velocity control')
    mapping = {'stop': 0, 'go forward': 1, 'turn left': 2, 'turn right': 3}
    if parsed.action not in mapping:
        raise ValueError('Action is not supported by Habitat navigation')
    quantum = 25 if parsed.action == 'go forward' else 15
    count = max(1, int(math.floor((parsed.amount or quantum) / quantum + 0.5)))
    return mapping[parsed.action], count
