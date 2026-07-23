import jax


def straight_through(fun, x):
    """Straight through gradient estimator.

    Args:
        fun:
            a (positively monotonic) function of x that produces a discrete value.

        x:
            the value where to evaluate `fun`

    Returns:
        Returns fun(x), such that ∂fun/∂x = 1
    """
    zero = x - jax.lax.stop_gradient(x)
    return fun(x) + zero


def straight_through_threshold(value, threshold):
    return straight_through(lambda v: v >= threshold, value)


def sigmoid_through_threshold(value, threshold):
    x = jax.nn.sigmoid(value - threshold)
    zero = x - jax.lax.stop_gradient(x)
    return (value >= threshold) + zero
