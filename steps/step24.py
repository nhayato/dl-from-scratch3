if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


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


def main():
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = sphere(x, y)
    z.backward()
    print(x.grad, y.grad)

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = matyas(x, y)
    z.backward()
    print(x.grad, y.grad)

    a = Variable(np.array(1.0))
    b = Variable(np.array(1.0))
    c = goldstein(a, b)
    c.backward()
    print(a.grad, b.grad)


if __name__ == '__main__':
    main()
