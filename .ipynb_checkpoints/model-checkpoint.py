import torch

class TheModel(torch.nn.Module):

    def __init__(self):
        super(TheModel, self).__init__()

        self.linear1 = torch.nn.Linear(784, 1)

    def forward(self, x):
        x1 = x.flatten(start_dim = 1)
        return self.linear1(x1).flatten()