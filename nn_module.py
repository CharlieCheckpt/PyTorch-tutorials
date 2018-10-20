"""Implements a one hidden layer neural network in Pytorch.
Use of nn package to create the network.
"""
import torch


def main():
    # Dimensions
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Creates random float input and output
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)


    # Use of the package nn to define our model, Keras-like
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )

    # defines loss using the nn module
    loss_fn = torch.nn.MSELoss(reduction="sum")

    learning_rate = 1e-4
    for t in range(500):
        # Forward pass : get predictions
        y_pred = model(x)

        # Compute and print loss
        loss = loss_fn(y_pred, y)
        print(t, loss.item())

        # Zero the gradients before running the backward pass
        model.zero_grad()

        # Backward pass : get gradients
        loss.backward()

        # Update the weights using gradient descent
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad


if __name__ == '__main__':
    main()
