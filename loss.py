import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        # N, 1, H, W
        x = x.view(-1, x.shape[2]*x.shape[3])
        e = F.softmax(x, dim=1)*F.log_softmax(x, dim=1)
        loss = -1 * e.sum()/x.shape[0]

        return loss
