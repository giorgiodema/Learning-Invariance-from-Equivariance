from turtle import forward
from typing import List
import torch
from torch import nn
from groupconv.conv import *
from groupconv.groups import *
from functools import reduce
from torch.nn import Conv2d
import math

class EqSimpleCNN(torch.nn.Module):
    def __init__(self,group,order,nclasses) -> None:
        super().__init__()
        div = int(math.sqrt(order))
        self.c1 = LiftingConvolution(group,in_channels=1,out_channels=64//div,kernel_size=3)
        self.r1 = nn.ReLU()
        self.p1 = SpatialMaxPool2d(kernel_size=2,stride=2)

        self.c2 = GroupConvolution(group,in_channels=64//div,out_channels=128//div,kernel_size=3)
        self.r2 = nn.ReLU()
        self.p2 = SpatialMaxPool2d(kernel_size=2,stride=2)

        self.c3 = GroupConvolution(group,in_channels=128//div,out_channels=256//div,kernel_size=3)
        self.r3 = nn.ReLU()

        self.gp = GlobalMaxPooling()
        self.group_pooling = GroupMaxPool()
        self.clf = nn.Linear(256//div,nclasses)
        

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
        o = self.group_pooling(o)
        o = self.clf(o)
        return o

if __name__=="__main__":
    m = EqSimpleCNN(CyclicGroup(4),4,2)
    m(torch.rand((1,1,32,32)))