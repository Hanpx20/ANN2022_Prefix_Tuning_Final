import jittor as jt
import torch.nn as nn
import torch
from zipfile import ZipFile

class TestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(10,10)
        self.linear_2 = nn.Linear(10,10)

    def forward(self,x):
        return self.linear_2(self.linear_1(x))

class jittorMLP(jt.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = jt.nn.Linear(10,10)
        self.linear_2 = jt.nn.Linear(10,10)
    
    def execute(self, x):
        return self.linear_2(self.linear_1(x))

def extract_zip(input_zip):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

test_model = TestMLP()
torch.save(test_model.state_dict(),'dir.pth')

mode = jt.load('dir.pth')
print('=== mode ===')
print(mode)
print('=== jt_model ===')
jt_model = jittorMLP()
jt_model.load('dir.pth')
print(jt_model)