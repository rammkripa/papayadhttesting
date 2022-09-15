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