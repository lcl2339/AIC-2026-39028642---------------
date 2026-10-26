"""TRAIN-fitted adaptation on the frozen visual representation."""
import torch
from torch import nn
from torch.nn import functional as F

class OwnPrior(nn.Module):
    def __init__(self, buffers):
        super().__init__()
        for name, value in buffers.items():
            self.register_buffer(name, value.clone())
        self.null_delta = nn.Parameter(torch.zeros(750,256))

    def forward(self, visual, feature, posterior):
        null = F.normalize(F.normalize(visual.float(),dim=-1)@self.complement,dim=-1)
        adapted = F.normalize(self.null_centers+.035*self.null_delta.tanh(),dim=-1)
        difference = 13*null@(adapted-F.normalize(self.null_centers,dim=-1)).T
        return 1.25*(1-posterior.detach().max(-1,keepdim=True).values)*torch.tanh(difference/1.25)
