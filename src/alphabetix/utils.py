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


if __name__ == "__main__":
    x = 2.0
    theta = 1.0

    y = x >= theta
    # y ∈ {0, 1}
    # and ∂y/∂x = 0

    y = straight_through_threshold(x, theta)
    # y ∈ {0, 1} (as before)
    # but ∂y/∂x = 1
