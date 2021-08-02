if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2+y**2)-0.48*x*y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1)**2
    return y


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=1e-150):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2*i+1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


def main():
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))

    lr = 0.001  # 学習率
    iters = 10000  # 繰り返し回数

    for _ in range(iters):
        print(x0, x1)

        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()

        y.backward()
        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad


if __name__ == '__main__':
    main()
