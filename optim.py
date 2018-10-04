"""Implements a one hidden layer neural network in Pytorch.
Use of optim package to update weights.
"""

import torch

def main():
    # Dimensions
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Creates random input and output
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Use of the package nn to define our model
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )

    # defines loss using the nn module
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Use the optim package to define an optimizer that will update the 
    # weights. The first arguent tells the optimizer which parameter it 
    # should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(500):
        # Forward pass
        y_pred = model(x)
        # Computes and prints loss
        loss = loss_fn(y_pred, y)
        print(t, loss.item())
        # Before backward pass, need to set optimizer object to zero
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Calling the step function to update the weights
        optimizer.step()


if __name__ == '__main__':
    main()
