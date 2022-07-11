import torchvision
import torch
from groupconv.groups import *
import random
from torchvision.transforms import Compose,Resize,ToTensor,Lambda,Normalize,RandomRotation,RandomHorizontalFlip,RandomVerticalFlip
torch.manual_seed(0)

def apply_random_action(x:torch.tensor,transformed_grid:torch.tensor):
    x = x.unsqueeze(0)
    n_actions = transformed_grid.shape[0]
    selected_action = random.randint(0,n_actions-1)
    grid = transformed_grid[[selected_action],...]
    xt = torch.nn.functional.grid_sample(x,grid,align_corners=True, mode="bilinear")
    return xt[0,...]

def get_loader(split:str,bs:int,nclass:int,first_class_idx=0,augment=None,img_size=28):
    transforms = []
    transforms += [
        ToTensor()
    ]
    if augment!=None:
        group = None
        if augment=="c4":
            group = CyclicGroup(4)
        elif augment=="d4":
            group = DihedralGroup(4)
        else:
            raise NotImplementedError
        img_grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, img_size),
            torch.linspace(-1, 1, img_size),
        ))
        transformed_grid = group.left_action_on_R2(group.elements(),img_grid_R2)
        transforms.append(
            torchvision.transforms.Lambda(lambda x:apply_random_action(x,transformed_grid))
        )
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

    tl = get_loader("test",4,10,augment="d4")
    for x,y in tl:
        print(y)
        for i in range(x.shape[0]):
            plt.imshow(torch.einsum("bcxy->bxyc",x)[i])
        plt.show()
        """
        print("transformed")
        group = CyclicGroup(4)
        img_grid_R2 = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, 28),
            torch.linspace(-1, 1, 28),
        ))
        transformed_grid = group.left_action_on_R2(group.elements(),img_grid_R2)
        n_actions = transformed_grid.shape[0]
        selected_action = random.randint(0,n_actions-1)
        print(f"selected action:{selected_action}")
        grid = transformed_grid[[selected_action],...].repeat(n_actions,1,1,1)
        xt = torch.nn.functional.grid_sample(x,grid,align_corners=True, mode="bilinear")

        for i in range(xt.shape[0]):
            plt.imshow(torch.einsum("bcxy->bxyc",xt)[i])
        plt.show()
        """