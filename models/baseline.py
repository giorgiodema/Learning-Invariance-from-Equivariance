from turtle import forward
import torch
from torch import nn
from functools import reduce
from groupconv.conv import GlobalMaxPooling



class SimpleCNN(torch.nn.Module):
    def __init__(self,nclasses) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.c2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.r2 = nn.ReLU()
        self.p2 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.c3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3)
        self.r3 = nn.ReLU()

        self.gp = GlobalMaxPooling()
        self.clf = nn.Linear(256,nclasses)
        

    def forward(self,x):
        o = self.c1(x)
        o = self.r1(o)
        o = self.p1(o)
        o = self.c2(o)
        o = self.r2(o)
        o = self.p2(o)
        o = self.c3(o)
        o = self.r3(o)
        o = self.gp(o)
        o = self.clf(o)
        return o



if __name__=="__main__":
    m = SimpleCNN(2)
    m(torch.rand((1,1,32,32)))