#!/usr/bin/env python

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


def main():
    x0 = np.array([1, 2, 3])
    x1 = np.array([10])
    y = x0 + x1
    print(y)

    # Add
    print("Add")
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    print(y)

    y.backward()
    print(x0.grad)
    print(x1.grad)

    # Sub
    print("Sub")
    x0 = Variable(np.array([10, 20, 30]))
    x1 = Variable(np.array([1]))
    y = x0 - x1
    print(y)

    y.backward()
    print(x0.grad)
    print(x1.grad)

    # Mul
    print("Mul")
    x0 = Variable(np.array([10, 20, 30]))
    x1 = Variable(np.array([2]))
    y = x0 * x1
    print(y)

    y.backward(retain_grad=True)
    print(y.grad)
    print(x0.grad)
    print(x1.grad)

    # Div
    print("Div")
    x0 = Variable(np.array([10, 20, 30]))
    x1 = Variable(np.array([2]))
    y = x0 / x1
    print(y)

    y.backward(retain_grad=True)
    print(y.grad)
    print(x0.grad)
    print(x1.grad)


if __name__ == '__main__':
    main()
