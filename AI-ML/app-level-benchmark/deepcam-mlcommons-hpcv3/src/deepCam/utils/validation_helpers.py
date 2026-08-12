# The MIT License (MIT)

import math


def compute_unpadded_loss(criterion, outputs, labels, num_samples):
    """Compute validation loss without synthetic batch-padding samples."""
    return criterion(
        outputs[:num_samples, ...],
        labels[:num_samples, ...])


def ensure_finite_validation_result(loss, accuracy):
    """Reject invalid validation metrics before they are logged."""
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise FloatingPointError(
            f"non-finite validation result: "
            f"loss={loss}, accuracy={accuracy}")
