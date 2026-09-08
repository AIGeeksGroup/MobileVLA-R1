"""Pure logic regressions: no model loading, training, or inference."""
import ast
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

from mobilevla.actions import (Action, format_valid, habitat_action, normalize_target,
                              parse_answer, reward, serialize_answer)
from mobilevla.data import load_records, normalize_path, observation_question, policy_question, sft_target, resolve_depth_path
from mobilevla.checkpoints import strip_adapter_prefix

ROOT = Path(__file__).resolve().parents[1]


class ActionContractTests(unittest.TestCase):
    def test_reasoning_cannot_override_final_action(self):
        text = '<think>Do not stop. Turning right would be wrong.</think><answer>turn left 30 degree</answer>'
        self.assertEqual(habitat_action(text), (2, 2))

    def test_missing_reasoning_is_not_valid_format(self):
        self.assertFalse(format_valid('<answer>stop</answer>', 'navigation'))

    def test_wrong_legal_action_gets_zero_action_reward(self):
        target = Action('turn left', (0., 0., 1.))
        score = reward('<think>Turn.</think><answer>{"velocity":[0,0,-1],"action":"turn right"}</answer>', target, 'control')
        self.assertEqual(score['action'], 0)
        self.assertEqual(score['movement'], -1)
        self.assertEqual(score['format'], 1)

    def test_equal_velocity_and_action_get_full_reward(self):
        target = Action('go forward', (1., 0., 0.))
        text = '<think>Clear path.</think><answer>{"velocity":[2,0,0],"action":"go forward"}</answer>'
        self.assertEqual(reward(text, target, 'control')['total'], 3.)

    def test_navigation_has_no_invented_velocity_reward(self):
        target = parse_answer('turn left')
        self.assertIsNone(target.velocity)
        self.assertEqual(reward('<think>Turn.</think><answer>turn left</answer>', target, 'navigation')['movement'], 0)

    def test_unknown_or_unstructured_text_is_not_stop(self):
        for value in ['nonsense', '<action>stop</action>', 'The robot should stop because the goal is near']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_answer(value)

    def test_legacy_control_requires_explicit_action_label(self):
        legacy = str([0.] * 12)
        with self.assertRaises(ValueError):
            normalize_target(legacy)
        _, parsed = normalize_target(legacy, action_label='stop')
        self.assertEqual(parsed, Action('stop', (0., 0., 0.)))

    def test_legacy_nav_wrapper_is_normalized(self):
        text, parsed = normalize_target('<think>Door on left.</think><action>turn left</action>')
        self.assertEqual(parsed.action, 'turn left')
        self.assertTrue(format_valid(text, 'navigation'))

    def test_gt_uses_answer_not_reasoning_numbers(self):
        text = '<think>1 2 3 4 5 6 7 8 9 10 11 12</think><answer>[0,1,0,"hello"]</answer>'
        _, parsed = normalize_target(text)
        self.assertEqual(parsed.velocity, (0., 1., 0.))
        self.assertEqual(parsed.action, 'hello')

    def test_malformed_and_nonfinite_values_rejected(self):
        for value in ['{"velocity":[true,0,0],"action":"stop"}',
                      '{"velocity":[NaN,0,0],"action":"stop"}',
                      '{"velocity":[1e999,0,0],"action":"stop"}',
                      'move forward 10 degree', 'turn left 0 degree',
                      '<answer>stop</answer><answer>turn left</answer>']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_answer(value)

    def test_format_requires_task_specific_answer(self):
        self.assertFalse(format_valid('<think>Plan.</think><answer>stop</answer>', 'control'))
        self.assertFalse(format_valid('<think></think><answer>stop</answer>', 'navigation'))
        self.assertFalse(format_valid('<think>A<think>B</think></think><answer>stop</answer>', 'navigation'))

    def test_round_trip(self):
        for value in ['move forward 50 cm', 'turn left 30 degree', 'stop', '[0.1,0,0,"go forward"]']:
            parsed = parse_answer(value)
            self.assertEqual(parse_answer(serialize_answer(parsed)), parsed)

    def test_zero_velocity_convention_and_large_values(self):
        target = Action('stop', (0., 0., 0.))
        self.assertEqual(reward('<answer>[0,0,0,"stop"]</answer>', target, 'control')['movement'], 1.)
        target = Action('go forward', (1e300, 0., 0.))
        value = reward('<answer>[1e300,0,0,"go forward"]</answer>', target, 'control')['movement']
        self.assertTrue(math.isfinite(value))
        self.assertEqual(value, 1.)


class DataContractTests(unittest.TestCase):
    def test_windows_root_rewrite_and_idempotence(self):
        source = r'C:\dataset\scans\scene\skybox\frame.jpg'
        rgb = normalize_path(source, '/data/rgb')
        self.assertEqual(rgb, '/data/rgb/scene/skybox/frame.jpg')
        self.assertEqual(normalize_path(rgb, '/data/rgb'), rgb)
        self.assertEqual(normalize_path(source, '/data/depth'), '/data/depth/scene/skybox/frame.jpg')

    def test_absolute_rgb_path_resolves_under_matching_depth_root(self):
        self.assertEqual(resolve_depth_path('/data/rgb/scene/frame.jpg', '/data/rgb', '/data/depth'),
                         '/data/depth/scene/frame.png')

    def test_relative_path_keeps_scene_component(self):
        self.assertEqual(normalize_path('scene/frame.jpg', '/data/rgb'), '/data/rgb/scene/frame.jpg')

    def test_wrong_absolute_root_is_not_silently_concatenated(self):
        with self.assertRaises(ValueError):
            normalize_path('/data/rgb/scene/frame.jpg', '/data/depth')

    def test_dataset_blank_lines_and_split_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.jsonl'
            path.write_text('\n{"instruction":"go"}\n\n')
            self.assertEqual(load_records(path), [{'instruction': 'go'}])

    def test_sft_and_rl_target_share_action(self):
        text, task = sft_target({'instruction': 'go', 'think': 'Path is clear', 'action': 'move forward 25 cm'})
        self.assertEqual(task, 'navigation')
        self.assertTrue(format_valid(text, task))
        self.assertIn('<answer>', text)

    def test_episode_is_explicit_and_preserves_text(self):
        text, task = sft_target({'task_type': 'episode', 'think': 'Plan', 'answer': 'Reached the kitchen'})
        self.assertEqual(task, 'episode')
        self.assertIn('Reached the kitchen', text)

    def test_observation_question_has_one_placeholder_per_modality_frame(self):
        prompt = observation_question(policy_question('go', 'navigation'), ['rgb', 'rgb', 'depth', 'point'])
        self.assertEqual(prompt.count('<image>'), 4)
        self.assertNotIn('geodesic', prompt)

    def test_peft_parameter_prefix_is_removed_once(self):
        self.assertEqual(strip_adapter_prefix('base_model.model.depth_tower.head.weight'), 'depth_tower.head.weight')
        self.assertEqual(strip_adapter_prefix('llm.model.layers.0.weight'), 'llm.model.layers.0.weight')

    def test_vln_dataset_dispatches_both_formats(self):
        # Execute the actual dataset dispatch method without importing torch.
        tree = ast.parse((ROOT / 'llava/data/dataset.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LazyVLNCEDataset')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_build_sample')
        namespace = {}
        source = 'from __future__ import annotations\n' + ast.unparse(method)
        exec(compile(source, '<dataset-dispatch>', 'exec'), namespace)
        dummy = types.SimpleNamespace(data_args=types.SimpleNamespace(navcot_use_depth=False),
                    _build_video_sample=lambda sample: 'legacy',
                    _build_navcot_sample=lambda sample: ('payload', 'question', 'answer', ['rgb']))
        self.assertEqual(namespace['_build_sample'](dummy, {'video_id': 'x', 'frames': []}), 'legacy')
        self.assertEqual(namespace['_build_sample'](dummy, {'images': [], 'instruction': 'go'}),
                         ('payload', 'question', 'answer', ['rgb'], True))


class MultimodalEntryTests(unittest.TestCase):
    def test_different_feature_lengths_are_preserved(self):
        # Run the real assembly method with shape-only stand-ins. No neural
        # modules are constructed or invoked.
        tree = ast.parse((ROOT / 'llava/model/llava_arch.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LlavaMetaModel')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_encode_multi_modal')
        namespace = {}
        exec(compile(ast.unparse(method), '<modality-assembly>', 'exec'), namespace)
        class Features:
            def __init__(self, lengths):
                self.lengths = lengths
            def to(self, **kwargs):
                return self
            def __getitem__(self, index):
                return self.lengths[index]
        dummy = types.SimpleNamespace(
            device='cpu', config=types.SimpleNamespace(model_dtype='float'),
            depth_tower=lambda value: value, point_tower=lambda value: value,
            depth_bridge=None, point_bridge=None,
            get_vision_tower=lambda: (lambda value: value),
            get_mm_projector=lambda: (lambda value: value))
        result = namespace['_encode_multi_modal'](dummy, {
            'token_types': ['rgb', 'depth', 'point'],
            'rgb': Features([196]), 'depth': Features([1]), 'point': Features([1])})
        self.assertEqual(result, [196, 1, 1])

    def test_sft_shell_forwards_overrides_without_starting_training(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / 'bin'
            bin_dir.mkdir()
            executable = bin_dir / 'torchrun'
            executable.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                                  'open(os.environ["CAPTURE_ARGS"], "w").write(json.dumps(sys.argv[1:]))\n')
            executable.chmod(0o755)
            capture = Path(tmp) / 'arguments.json'
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       MODEL_PATH='/example/model', NAVCOT_IMAGE_ROOT='/example/rgb',
                       NAVCOT_DEPTH_ROOT='/example/depth', CAPTURE_ARGS=str(capture))
            subprocess.run(['bash', 'scripts/train/sft_8frames.sh', '--learning_rate', '0.00001'],
                           cwd=ROOT, env=env, check=True, capture_output=True)
            args = json.loads(capture.read_text())
            self.assertEqual(args[-2:], ['--learning_rate', '0.00001'])
            self.assertIn('/example/model', args)

    def test_annotation_normalizer_runs_without_model_dependencies(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'input.json', Path(tmp) / 'output.jsonl'
            source.write_text(json.dumps([{'images': ['scene/frame.jpg'], 'instruction': 'go',
                                          'think': 'Path is clear', 'action': 'move forward 25 cm'}]))
            subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_cot_annotations.py'),
                            '--input', str(source), '--output', str(output), '--task', 'navigation'],
                           check=True, capture_output=True)
            record = load_records(output)[0]
            self.assertIn('<answer>move forward 25 cm</answer>', record['a'])
            self.assertEqual(record['images'], ['scene/frame.jpg'])


if __name__ == '__main__':
    unittest.main()
