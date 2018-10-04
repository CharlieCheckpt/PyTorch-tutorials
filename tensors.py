"""Implements a one hidden layer neural network in Pytorch.
Manual implementation of forward and backward (backpropagation) pass.
"""

import torch

def main():
    dtype = torch.float
    device = torch.device("cpu")

    # Dimensions
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Creates random input abd output
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Initialize weight
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(500):
        # Forward pass
        h = x.mm(w1)
        h_relu = h.clamp(min=0)  # RELU activation
        y_pred = h_relu.mm(w2)

        # Computes and prints loss
        loss = (y_pred - y).pow(2).sum().item()
        print(t, loss)

        # Backprogation
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Updates weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


if __name__ == '__main__':
    main()
