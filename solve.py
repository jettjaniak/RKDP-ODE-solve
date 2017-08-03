import numpy as np
import typing


def rkdp_solve(f: typing.Callable[[float, np.ndarray], np.ndarray],
               a: float, b: float, n: int, x0: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Solve ordinary differential equation given by function f using Dormand–Prince method.

    Arguments:
        f: Vector-valued function of time and 'X'.
        a: Start of interval.
        b: End of interval.
        n: Number of iterations.
        x0: Starting point.

    Returns:
        T: Vector of time points.
        X: Vector of corresponding solution points.
    """
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 55, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = 9017 / 3186, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656

    c2, c3, c4, c5, c6 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1

    B = np.array((35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84))

    T = [a]
    X = [x0]
    h = (b - a) / n
    for i in range(1, n + 1):
        k1 = f(T[-1], X[-1])
        k2 = f(T[-1] + c2 * h, X[-1] + h * a21 * k1)
        k3 = f(T[-1] + c3 * h, X[-1] + h * (a31 * k1 + a32 * k2))
        k4 = f(T[-1] + c4 * h, X[-1] + h * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(T[-1] + c5 * h, X[-1] + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(T[-1] + c6 * h, X[-1] + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k = np.array((k1, k2, k3, k4, k5, k6))
        X.append(
            X[-1] + h * B @ k
        )
        T.append(T[-1] + h)
    T = np.array(T)
    X = np.array(X)
    return T, X
