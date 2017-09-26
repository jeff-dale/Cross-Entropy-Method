import numpy as np


def Rastrigin(x: np.ndarray, A: float = 10, n: int = 2) -> float:
    """
    Benchmark function with many local optima.
    Global minimum is at x = 0.
    Bounds: (-5.12, 5.12)

    :param x: input vector
    :param A: arbitrary constant, typically A=10
    :param n: number of dimensions
    :return: value of Rastrigin function
    """
    total = np.ones((x.shape[0]))*A*n
    for i in range(n):
        total += (np.power(x[:, i], 2) - A*np.cos(2*np.pi*x[:, i]))
    return total.T


def Ackley(x: np.ndarray) -> float:
    """
    Ackley function for benchmarking.
    Global minimum at (0, 0).
    Bounds: (-5, 5)

    :param x: input vector, must be 2D
    :return: value of Ackley function
    """
    return -20*np.exp(-0.2*np.sqrt(0.5*(np.power(x[:, 0], 2) + np.power(x[:, 1], 2)))) - \
        np.exp(0.5*(np.cos(2*np.pi*x[:, 0]) + np.cos(2*np.pi*x[:, 1]))) + np.e + 20


def Easom(x: np.ndarray) -> float:
    """
    Easom function for benchmarking.
    Global minimum at f(pi, pi) = -1
    Bounds: (-100, 100)

    :param x: input vector, must be 2D
    :return: value of Easom function
    """
    return -np.cos(x[:, 0])*np.cos(x[:, 1])*np.exp(-(np.power(x[:, 0] - np.pi, 2) + np.power(x[:, 1] - np.pi, 2)))


def Cross_in_Tray(x: np.ndarray) -> float:
    """
    Cross-in-tray function for benchmarking.
    Global minimums at f(+- 1.34941, +-1.34941) = -2.06261
    Bounds: (-10, 10)

    :param x:input vector, must be 2D
    :return: value of Cross_in_Tray function
    """
    return -0.0001*np.power(np.abs(np.sin(x[:, 0])*np.sin(x[:, 1])*np.exp(np.abs(100-np.sqrt(np.power(x[:, 0], 2) + np.power(x[:, 1], 2))/np.pi))) + 1, 0.1)
