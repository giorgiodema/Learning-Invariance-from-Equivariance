import torch

class GlobalMaxPooling(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x):
        o = torch.max(x,dim=-1).values
        o = torch.max(o,dim=-1).values
        return o