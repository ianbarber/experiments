"""Fixed class weighting, with one denominator per effective batch.

This module imports no model library. The same denominator must be passed to
every microbatch; padding and prefix tokens have zero loss from the caller.
"""
import math


def class_weight(decision, mode):
    if decision not in ('REPORT', 'CLEAR'):
        raise ValueError('Unknown target decision.')
    if mode == 'uniform':
        return 1.0
    if mode == 'reweighted':
        return 0.75 if decision == 'REPORT' else 1.5
    raise ValueError('Unknown class weighting mode.')


def effective_denominator(examples):
    if not examples:
        raise ValueError('Empty effective batch.')
    for example in examples:
        if (type(example['target_tokens']) is not int or example['target_tokens'] <= 0
                or not math.isfinite(example['loss_weight']) or example['loss_weight'] <= 0):
            raise ValueError('Invalid token count or weight.')
    return sum(example['loss_weight'] * example['target_tokens'] for example in examples)


def weighted_microbatch_loss(losses, weights, denominator):
    if len(weights) != losses.shape[0] or denominator <= 0:
        raise ValueError('Invalid microbatch weighting dimensions.')
    # Retain the inherited uniform target-token mean computation exactly.
    if all(weight == 1.0 for weight in weights):
        return losses.sum() / denominator
    return (losses.sum(dim=1) * losses.new_tensor(weights)).sum() / denominator
