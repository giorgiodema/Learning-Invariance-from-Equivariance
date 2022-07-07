import torch
import torchvision
from torchvision.transforms import ToTensor, Lambda,Compose,Resize,Normalize
import matplotlib.pyplot as plt
from functools import reduce
from models.equivariant import *
from groupconv.groups import *
from utils.train import count_params, train,val
from data.loader import get_loader
import os

BS=16
EPOCHS=15
NCLASSES = 10
NAME = "C4SimpleCNNSigmoid_FashionMNIST"

print(f"MODEL NAME:{NAME}")
trainloader = get_loader("train",BS,NCLASSES)
valloader = get_loader("val",BS,NCLASSES)
testloader = get_loader("test",BS,NCLASSES)

m = EqSimpleCNNSigmoid(CyclicGroup(4),4,NCLASSES)
print(m)
print(f"N PARAMS: {count_params(m)}")
m = m.cuda()

testloader = get_loader("test",BS,NCLASSES)
m.load_state_dict(torch.load(f"ckpt/{NAME}.pth"))
acc,loss = val(
    testloader,
    m,
    torch.nn.CrossEntropyLoss(),
    "cuda"
)