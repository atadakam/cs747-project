import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetAttention(nn.Module):
    def __init__(self):
        super(ResNetAttention, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet_embedding_dim = self.resnet.fc.in_features

        # RNN Hyper-Params
        self.rnn_hidden_dim = 24
        self.bidirectional_att = True
        self.effective_rnn_hidden_dim = self.rnn_hidden_dim
        # bidirectional means 2*hidden dim
        if self.bidirectional_att:
            self.effective_rnn_hidden_dim *= 2

        # Final CONV Hyper-Params
        self.att_conv_filters = 20

        self.lstm = nn.LSTM(self.resnet_embedding_dim, self.rnn_hidden_dim, bidirectional=self.bidirectional_att)
        self.att_extractor = nn.Linear(self.effective_rnn_hidden_dim, 1)
        self.att_conv = nn.Conv2d(self.resnet_embedding_dim, self.att_conv_filters, kernel_size=1)
        self.resnet.fc = nn.Linear(self.att_conv_filters, 10)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # 8x8 for 227x227 with d=512 [batch, channels, h, w]

        # Skipped resnet functions
        # x = self.resnet.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.resnet.fc(x)

        # Prepare dims for Recurrent
        x_channel = x.shape[1]
        x_batch_size = x.shape[0]
        x_att_in = x.permute(0, 2, 3, 1)                        # batch, h, w, channels
        x_att_in = x_att_in.view(x_batch_size, -1, x_channel)   # batch, h*w, channels
        x_att_in = x_att_in.permute(1, 0, 2)                    # h*w, batch, channels

        # Recurrent
        att_vec, _ = self.lstm(x_att_in)                            # h*w, batch, embedding

        # Single Normalized Attention Extraction from Recurrent Embedded Vec
        att_vec_flat = att_vec.view(-1, self.effective_rnn_hidden_dim)
        att = F.relu(self.att_extractor(att_vec_flat))              # h*w*batch, 1
        att = att.view(-1, x_batch_size)                            # h*w, batch
        att = F.normalize(att, p=1, dim=0)

        att = att.view(x.shape[2], x.shape[3], x_batch_size, 1)
        att = att.permute(2, 3, 0, 1)

        x = x*att
        x = F.relu(self.att_conv(x))
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x, att

    def __str__(self):
        return self.resnet.__str__()


if __name__ == '__main__':
    r = ResNetAttention()
    # x = torch.randn(2, 3, 227, 227)
    # y, a = r(x)
    # print(y.shape)
    # print(a.shape)
