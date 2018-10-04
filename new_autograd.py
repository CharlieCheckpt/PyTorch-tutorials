"""Implements a one hidden layer neural network in Pytorch.
Let's define our own autograd operator for ReLU non linearity
by defining a subclass of torch.autograd.Function and implementing 
the Forward and Backward functions.
"""
import torch

class MyReLU(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        """We receive a tensor containing the input
        and return a tensor containing the output. ctx is a context object
        that can be used to stash information for backward computation.
        You can cache arbitrary objects for use in the backward pass using
        the ctx.save_for_backward method.
        
        Args:
            ctx (): 
            input (Tensor): input tensor.
        
        Returns:
            Tensor: output tensor.
        """

        ctx.save_for_backward(input)
        return input.clamp(min=0)


    @staticmethod
    def backward(ctx, grad_output):
        """We receive a tensor containing the gradient of the loss wrt to
        the output and we need to compute the gradient of the loss wrt to 
        the input.
        
        Args:
            ctx ():
            grad_output (Tensor): Gradient of the output.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


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
        relu = MyReLU.apply
        # Forward pass
        y_pred = relu(x.mm(w1)).mm(w2)

        # Computes and print loss
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

        # Use autograd to compute the backward pass
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
