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
NAME = "D4SimpleCNNDecay_FashionMNIST"

print(f"MODEL NAME:{NAME}")
trainloader = get_loader("train",BS,NCLASSES)
valloader = get_loader("val",BS,NCLASSES)
testloader = get_loader("test",BS,NCLASSES)

m = EqSimpleCNN(DihedralGroup(4),8,NCLASSES)
print(m)
print(f"N PARAMS: {count_params(m)}")
m = m.cuda()

if not os.path.exists(f"ckpt/{NAME}.pth"):
    train(
        trainloader,
        m,
        torch.nn.CrossEntropyLoss(),
        torch.optim.Adam(m.parameters(),weight_decay=.1),
        device="cuda",
        epochs=EPOCHS,
        val_loader=valloader,
        model_name=NAME
    )
else:
    for aug in [None,"c4","d4"]:
        testloader = get_loader("test",BS,NCLASSES,augment=aug)
        m.load_state_dict(torch.load(f"ckpt/{NAME}.pth"))
        acc,loss = val(
            testloader,
            m,
            torch.nn.CrossEntropyLoss(),
            "cuda"
        )
        print(f"Augmentation: {aug}, Test Acc:{acc*100:.2f}%     Test Loss:{loss:>7f}")