import torch
import torch.nn as nn

class TestMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim,dim)
        self.linear_2 = nn.Linear(dim,dim)

    def forward(self,x):
        return self.linear_2(self.linear_1(x))

class LoadMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        testmlp = TestMLP(dim)
        self.prefix = nn.Parameter(torch.randn(dim,1))

    def forward(self,x,y):
        x = torch.stack(self.prefix, x)
        return self.testmlp(x)

test = TestMLP(10)
load = LoadMLP(10)

torch.save(test.state_dict(),'dir.pth')
state_dict = torch.load('dir.pth')
load.load_state_dict(state_dict)

print(test)
print(load)