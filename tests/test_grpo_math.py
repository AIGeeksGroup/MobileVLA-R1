"""Optional tensor-only checks. Never load a model or execute a forward pass."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from mobilevla.grpo import dumps, loads, group_advantages, grpo_loss, completion_logps_from_logits


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed; tensor-only checks deferred')
class GRPOMathTests(unittest.TestCase):
    def test_group_advantage_survives_microbatch_slicing(self):
        advantage = group_advantages(torch.tensor([0., 1., 3.]))
        self.assertNotEqual(advantage[-1].item(), 0.)
        batch = {'old_logps': torch.zeros(1, 1), 'ref_logps': torch.zeros(1, 1),
                 'advantages': advantage[-1:], 'completion_mask': torch.ones(1, 1, dtype=torch.bool)}
        self.assertAlmostEqual(grpo_loss(torch.zeros(1, 1), batch, beta=0).item(), -advantage[-1].item())

    def test_zero_kl_for_identical_policies_and_padding_ignored(self):
        current = torch.tensor([[-2., -3., 0.], [-1., 0., 0.]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        batch = {'old_logps': current.clone(), 'ref_logps': current.clone(),
                 'advantages': torch.tensor([1., -1.]), 'completion_mask': mask}
        self.assertEqual(grpo_loss(current, batch).item(), 0.)
        self.assertTrue(torch.equal(batch['advantages'], torch.tensor([1., -1.])))

    def test_clip_uses_correct_sign_for_negative_advantage(self):
        current = torch.tensor([[1.], [-1.]])
        batch = {'old_logps': torch.zeros(2, 1), 'ref_logps': current.clone(),
                 'advantages': torch.tensor([1., -1.]), 'completion_mask': torch.ones(2, 1, dtype=torch.bool)}
        self.assertAlmostEqual(grpo_loss(current, batch, beta=0, clip=0.2).item(), -0.2, places=6)

    def test_first_completion_uses_previous_logit_and_keeps_eos(self):
        logits = torch.tensor([[[-1., -1., 5.], [5., -1., -1.], [-1., 5., -1.]]])
        labels = torch.tensor([[-100, 2, 0]])  # zero is a real EOS, not padding
        mask = torch.tensor([[True, True]])
        actual = completion_logps_from_logits(logits, labels, mask)
        expected = torch.log_softmax(torch.tensor([-1., -1., 5.]), -1)[2]
        self.assertEqual(actual.shape, (1, 2))
        self.assertTrue(torch.allclose(actual, expected.expand_as(actual)))

    def test_visual_ignored_labels_do_not_create_extra_completion_tokens(self):
        logits = torch.zeros(2, 6, 3)
        labels = torch.tensor([[-100, -100, -100, 1, 2, 0], [-100, -100, -100, 0, -100, -100]])
        mask = torch.tensor([[True, True, True], [True, False, False]])
        actual = completion_logps_from_logits(logits, labels, mask)
        self.assertEqual(actual.shape, (2, 3))
        self.assertTrue(torch.equal(actual[1, 1:], torch.zeros(2)))

    def test_transport_keeps_multimodal_tensors(self):
        data = {'protocol': 2, 'images': {'rgb': torch.ones(2, 3, 2, 2),
                                        'depth': torch.ones(1, 1, 2, 2), 'token_types': ['rgb', 'rgb', 'depth']}}
        restored = loads(dumps(data))
        self.assertTrue(torch.equal(restored['images']['depth'], data['images']['depth']))
        self.assertEqual(restored['images']['token_types'], data['images']['token_types'])


if __name__ == '__main__':
    unittest.main()
