"""Implements a one hidden layer neural network in Pytorch.
Use of custom nn module to build the neural network.
"""

import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    # seems like only a forward function is needed in the nn module
    # activations are in the forward function.
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
            

def main():
    # Dimensions
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Creates random input and output
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    learning_rate = 1e-4
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(500):
        # Forward pass
        y_pred = model(x)

        # Computes and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradient & Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
