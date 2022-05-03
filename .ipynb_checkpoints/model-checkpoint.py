import torch

class TheModel(torch.nn.Module):

    def __init__(self):
        super(TheModel, self).__init__()

        self.linear1 = torch.nn.Linear(784, 400)
        self.linear2 = torch.nn.Linear(400, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = x.flatten(start_dim = 1)
        return self.linear2(self.relu(self.linear1(x1)))