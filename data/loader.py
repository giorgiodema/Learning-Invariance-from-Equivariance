import torchvision
import torch
from torchvision.transforms import Compose,Resize,ToTensor,Lambda,Normalize,RandomRotation,RandomHorizontalFlip,RandomVerticalFlip
torch.manual_seed(0)

def get_loader(split:str,bs:int,nclass:int,first_class_idx=0,augment=None):
    transforms = []
    if augment=="c4":
        transforms+=[
            RandomRotation(90),
            RandomRotation(180),
            RandomRotation(270)
        ]
    elif augment=="d4":
        transforms+=[
            RandomRotation(90),
            RandomRotation(180),
            RandomRotation(270),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(90),
            RandomRotation(180),
            RandomRotation(270)
        ]
    transforms += [
        #Resize((256,256)),
        ToTensor()
    ]
    dataset = torchvision.datasets.FashionMNIST(
        "dataset/",
        download=True,
        train=True if (split=="train" or split=="val") else False,
        target_transform=Lambda(lambda y: torch.zeros(nclass, dtype=torch.float).scatter_(0, torch.tensor(y)-first_class_idx, value=1)))
    if split=="val" or split=="train":
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size,val_size])
        if split=="val":
            dataset = val_dataset
        else:
            dataset = train_dataset
        dataset.dataset.transform = Compose(transforms)
    else:
        dataset.transform = Compose(transforms)
    loader = torch.utils.data.DataLoader(dataset,batch_size=bs,shuffle=True)
    return loader

if __name__=="__main__":
    import matplotlib.pyplot as plt

    tl = get_loader("test",4,10,augment=None)
    for x,y in tl:
        print(y)
        for i in range(x.shape[0]):
            plt.imshow(torch.einsum("bcxy->bxyc",x)[i])
        plt.show()