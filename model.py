import torch
import torch.nn.utils.prune as prune

class TheModel(torch.nn.Module):

    def __init__(self):
        super(TheModel, self).__init__()
        self.linear1 = torch.nn.Linear(784, 1)

    def forward(self, x):
        x1 = x.flatten(start_dim = 1)
        return self.linear1(x1).flatten()

    def prune(self, amt) :
        prune.l1_unstructured(self.linear1, name="weight", amount=amt)
        prune.remove(self.linear1, "weight")

class TheConvModel(torch.nn.Module):

    def __init__(self):
        super(TheConvModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (3, 3))
        self.linear1 = torch.nn.Linear(676, 1)

    def forward(self, x):
        x1 = x.reshape((-1, 1, 28, 28))
        x2 = self.conv1(x1)
        x3 = x2.flatten(start_dim = 1)
        return self.linear1(x3).flatten()

    def prune(self, amt) :
        prune.l1_unstructured(self.conv1, name="weight", amount=amt)
        prune.remove(self.conv1, "weight")
        prune.l1_unstructured(self.linear1, name="weight", amount=amt)
        prune.remove(self.linear1, "weight")

class TheLogisticModel(torch.nn.Module):

    def __init__(self):
        super(TheLogisticModel, self).__init__()
        self.linear1 = torch.nn.Linear(784, 10)

    def forward(self, x):
        x1 = x.flatten(start_dim = 1)
        return self.linear1(x1)

    def prune(self, amt) :
        prune.l1_unstructured(self.linear1, name="weight", amount=amt)
        prune.remove(self.linear1, "weight")