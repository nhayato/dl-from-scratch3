#!/usr/bin/env python

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from weakref import KeyedRef
import numpy as np
from dezero import Variable
import dezero.functions as F


def main():
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()
    print(y)       # variable(21)
    print(x.grad)  # variable([1 1 1 1 1 1])

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()
    print(y)       # variable(21)
    print(x.grad)  # variable([1 1 1][1 1 1])

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.random.randn(2, 3, 4, 5))
    y = x.sum(keepdims=True)
    print(y.shape)


if __name__ == '__main__':
    main()
