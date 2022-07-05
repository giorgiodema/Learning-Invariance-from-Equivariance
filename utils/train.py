import numpy as np
import torch
from functools import reduce

def val(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct,test_loss

def train(dataloader, model, loss_fn, optimizer, device, epochs, val_loader=None, save_best = True, save_path="ckpt", model_name=None):
    
    best_loss = np.inf

    size = len(dataloader.dataset)
    model.train()
    for ep in range(epochs):
        correct = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"ep:{ep}/{epochs}    loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    
        if val_loader != None:
            test_acc,test_loss = val(val_loader,model,loss_fn,device)
            print("-----------------------------")
            print(f"Val Acc:{test_acc:.2f}  Val loss:{test_loss:>7f}    Train Acc:{correct/size:.2f}")
            if test_loss < best_loss:
                print(f"Loss decreased {best_loss:>7f}->{test_loss:>7f}, Saving")
                best_loss = test_loss
                torch.save(model.state_dict(),f"{save_path}/{model_name if model_name!=None else model.__class__.__name__}.pth")

def count_params(model):
    s = 0
    for p in model.parameters():
        s += reduce(lambda x,y:x*y,p.size())
    return s