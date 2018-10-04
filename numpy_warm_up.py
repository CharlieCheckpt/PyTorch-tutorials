"""Implements a one hidden layer neural network in numpy.
Manual implementation of forward and backward (backpropagation) pass.
""" 

import numpy as np

def main():
    # N is y size
    # D_in is input dimension
    # D_out is output dimension
    # H is hidden layer dimension
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # Randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    learning_rate = 1e-6

    for t in range(500):
        # forward pass : compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Computes and prints loss
        loss = np.square(y - y_pred).sum()
        print(t, loss)

        # Backward pass
        # computes gradients using back-propagation
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        # with gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


if __name__ == '__main__':
    main()
