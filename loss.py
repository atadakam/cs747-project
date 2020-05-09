import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, eps=1e-20):
        # N, 1, H, W
        x = x.view(-1, x.shape[2]*x.shape[3])
        # Add epsilon to avoid log(0)
        x = x + eps
        # e = F.softmax(x, dim=1)*F.log_softmax(x, dim=1)
        e = x*torch.log(x)
        loss = -1 * e.sum()/x.shape[0]

        return loss


if __name__ == '__main__':
    import torch
    a = torch.randn((2, 1, 3, 3), requires_grad=True)
    print(a)
    s = F.softmax(a, dim=0)
    print(s)
    print(a.grad)
    e = EntropyLoss()
    l = e(s)
    print(l)
    l.backward()
    print(a.grad)