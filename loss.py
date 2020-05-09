import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        x = x.view(-1, x.shape[2])
        e = F.softmax(x, dim=0)*F.log_softmax(x, dim=0)
        loss = -1 * e.sum()/x.shape[2]

        return loss
