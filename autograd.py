"""Implements a one hidden layer neural network in Pytorch.
Use of autograd to get gradients for backward pass.
"""
import torch


def main():
    dtype = torch.float
    device = torch.device("cpu")

    # Dimensions
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Creates random input and output
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Initialize weight (requires_grad = True)
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        # Forward pass
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # Computes loss
        # loss is as tensor of shape (1, )
        # loss.item() gets the scalar value held in the loss
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

        # Computes backward pass using autograd
        # computes the gradient of loss w.r.t all tensors with requires_grad=True
        # after this call, w1.grad and w2.grad will be tensors holding the gradient of the loss
        loss.backward()

        # Manually update weights using gradient descent
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()

if __name__ == '__main__':
    main()

