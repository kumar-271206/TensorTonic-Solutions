def linear_lr(
    step: int,
    total_steps: int,
    initial_lr: float,
    final_lr: float = 0.0,
    warmup_steps: int = 0,
) -> float:
    """
    Returns the learning rate as a float.
    """
    if total_steps <= 0:
        return float(final_lr)

    if warmup_steps > 0 and step < warmup_steps:
        lr = initial_lr * (step / warmup_steps)
    elif step < total_steps:
        decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = initial_lr + decay_progress * (final_lr - initial_lr)
    else:
        lr = final_lr

    return float(lr)