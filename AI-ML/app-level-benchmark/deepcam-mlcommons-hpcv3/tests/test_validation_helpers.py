# The MIT License (MIT)

import unittest

import torch

from utils import validation_helpers


class RecordingCriterion:
    def __init__(self):
        self.output_shape = None
        self.label_shape = None

    def __call__(self, outputs, labels):
        self.output_shape = tuple(outputs.shape)
        self.label_shape = tuple(labels.shape)
        return outputs.mean()


class ValidationHelpersTest(unittest.TestCase):
    def test_compute_unpadded_loss_excludes_padding(self):
        outputs = torch.tensor([
            [[[2.0]]],
            [[[4.0]]],
            [[[1000.0]]],
            [[[1000.0]]],
        ])
        labels = torch.zeros((4, 1, 1), dtype=torch.int64)
        criterion = RecordingCriterion()

        loss = validation_helpers.compute_unpadded_loss(
            criterion, outputs, labels, num_samples=2)

        self.assertEqual(criterion.output_shape, (2, 1, 1, 1))
        self.assertEqual(criterion.label_shape, (2, 1, 1))
        self.assertEqual(loss.item(), 3.0)

    def test_finite_validation_result_passes(self):
        validation_helpers.ensure_finite_validation_result(0.25, 0.75)

    def test_nan_loss_is_rejected(self):
        with self.assertRaisesRegex(
                FloatingPointError, "non-finite validation result"):
            validation_helpers.ensure_finite_validation_result(
                float("nan"), 0.75)

    def test_infinite_accuracy_is_rejected(self):
        with self.assertRaisesRegex(
                FloatingPointError, "non-finite validation result"):
            validation_helpers.ensure_finite_validation_result(
                0.25, float("inf"))


if __name__ == "__main__":
    unittest.main()
