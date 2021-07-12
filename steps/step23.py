if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


def main():
    x = Variable(np.array(1.0))

    print(x)
    y = (x + 3)**2
    y.backward()

    print(y)
    print(x.grad)


if __name__ == '__main__':
    main()
