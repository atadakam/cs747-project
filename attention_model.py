import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetAttention1(nn.Module):
    def __init__(self):
        super(ResNetAttention1, self).__init__()
        self.model_name = 'ResNet18_att_relu_normal'
        self.resnet = models.resnet18(pretrained=True)
        self.resnet_embedding_dim = self.resnet.fc.in_features

        # RNN Hyper-Params
        self.rnn_hidden_dim = 12
        self.bidirectional_att = True
        self.effective_rnn_hidden_dim = self.rnn_hidden_dim
        # bidirectional means 2*hidden dim
        if self.bidirectional_att:
            self.effective_rnn_hidden_dim *= 2

        # Final CONV Hyper-Params
        self.att_conv_filters = 20

        self.recurrent = nn.LSTM(self.resnet_embedding_dim, self.rnn_hidden_dim, bidirectional=self.bidirectional_att)
        self.att_extractor = nn.Linear(self.effective_rnn_hidden_dim, 1)
        self.att_conv = nn.Conv2d(self.resnet_embedding_dim, self.att_conv_filters, kernel_size=1)
        self.resnet.fc = nn.Linear(self.att_conv_filters, 10)

        # Weight Init
        nn.init.kaiming_normal_(self.att_conv.weight, mode='fan_out', nonlinearity='relu')
        for name, param in self.recurrent.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

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
        x_att_in = x.permute(0, 2, 3, 1)                        # N, H, W, C
        x_att_in = x_att_in.view(x_batch_size, -1, x_channel)   # N, H*W, C
        x_att_in = x_att_in.permute(1, 0, 2)                    # H*W, N, C

        # Recurrent
        att_vec, _ = self.recurrent(x_att_in)                            # H*W, N, embedding

        # Single Normalized Attention Extraction from Recurrent Embedded Vec
        att_vec_flat = att_vec.view(-1, self.effective_rnn_hidden_dim)
        att = F.relu(self.att_extractor(att_vec_flat))              # H*W*N, 1
        att = att.view(-1, x_batch_size)                            # H*W, N
        att = F.normalize(att, p=1, dim=0)

        att = att.view(x.shape[2], x.shape[3], x_batch_size, 1)     # H, W, N, 1
        att = att.permute(2, 3, 0, 1)                               # N, 1, H, W

        x = x*att
        x = F.relu(self.att_conv(x))
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return {'logits': x, 'att': att}


class ResNetAttention2(nn.Module):
    def __init__(self):
        super(ResNetAttention2, self).__init__()
        self.model_name = 'ResNet18_att_softmax'
        self.resnet = models.resnet18(pretrained=True)
        self.resnet_embedding_dim = self.resnet.fc.in_features

        # RNN Hyper-Params
        self.rnn_hidden_dim = 12
        self.bidirectional_att = True
        self.effective_rnn_hidden_dim = self.rnn_hidden_dim
        # bidirectional means 2*hidden dim
        if self.bidirectional_att:
            self.effective_rnn_hidden_dim *= 2

        # Final CONV Hyper-Params
        self.att_conv_filters = 20

        self.recurrent = nn.LSTM(self.resnet_embedding_dim, self.rnn_hidden_dim, bidirectional=self.bidirectional_att)
        self.att_extractor = nn.Linear(self.effective_rnn_hidden_dim, 1)
        self.att_conv = nn.Conv2d(self.resnet_embedding_dim, self.att_conv_filters, kernel_size=1)
        self.resnet.fc = nn.Linear(self.att_conv_filters, 10)

        # Weight Init
        nn.init.kaiming_normal_(self.att_conv.weight, mode='fan_out', nonlinearity='relu')
        for name, param in self.recurrent.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

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
        x_att_in = x.permute(0, 2, 3, 1)                        # N, H, W, channels
        x_att_in = x_att_in.view(x_batch_size, -1, x_channel)   # N, H*W, channels
        x_att_in = x_att_in.permute(1, 0, 2)                    # H*W, N, channels

        # Recurrent
        att_vec, _ = self.recurrent(x_att_in)                            # H * W, N, embedding

        # Single Normalized Attention Extraction from Recurrent Embedded Vec
        att_vec_flat = att_vec.view(-1, self.effective_rnn_hidden_dim)  # H * W * N, embedding
        att = self.att_extractor(att_vec_flat)                          # H * W * N, 1

        att = att.view(-1, x_batch_size)                                # H * W, N
        att = F.softmax(att, dim=0)
        att = att.view(x.shape[2], x.shape[3], x_batch_size, 1)         # H, W, N, 1
        att = att.permute(2, 3, 0, 1)                                   # N, 1, H, W

        x = x*att
        x = F.relu(self.att_conv(x))
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return {'logits': x, 'att': att}


class ResNetAttention3(nn.Module):
    def __init__(self):
        super(ResNetAttention3, self).__init__()
        self.model_name = 'ResNet18_att_relu_fc'
        self.resnet = models.resnet18(pretrained=True)
        self.resnet_embedding_dim = self.resnet.fc.in_features

        # RNN Hyper-Params
        self.rnn_hidden_dim = 12
        self.bidirectional_att = True
        self.effective_rnn_hidden_dim = self.rnn_hidden_dim
        # bidirectional means 2*hidden dim
        if self.bidirectional_att:
            self.effective_rnn_hidden_dim *= 2

        self.recurrent = nn.LSTM(self.resnet_embedding_dim, self.rnn_hidden_dim, bidirectional=self.bidirectional_att)
        self.att_extractor = nn.Linear(self.effective_rnn_hidden_dim, 1)
        self.resnet.fc = nn.Linear(self.resnet_embedding_dim, 10)

        # Weight Init
        for name, param in self.recurrent.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

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
        x_att_in = x.permute(0, 2, 3, 1)                        # N, H, W, channels
        x_att_in = x_att_in.view(x_batch_size, -1, x_channel)   # N, H*W, channels
        x_att_in = x_att_in.permute(1, 0, 2)                    # H*W, N, channels

        # Recurrent
        att_vec, _ = self.recurrent(x_att_in)                            # H * W, N, embedding

        # Single Normalized Attention Extraction from Recurrent Embedded Vec
        att_vec_flat = att_vec.view(-1, self.effective_rnn_hidden_dim)  # H * W * N, embedding
        att = F.relu(self.att_extractor(att_vec_flat))                          # H * W * N, 1

        att = att.view(-1, x_batch_size)                                # H * W, N
        # Normalize so that attention**2 sum is 1
        att = F.normalize(att, p=2, dim=0)
        att = att.view(x.shape[2], x.shape[3], x_batch_size, 1)         # H, W, N, 1
        att = att.permute(2, 3, 0, 1)                                   # N, 1, H, W

        x = x*att                                                       # N, C, H, W
        x = x.view(x_batch_size, x_channel, -1)                         # N, C, H*W
        x = torch.sum(x, dim=2)
        x = self.resnet.fc(x)

        return {'logits': x, 'att': att}


class ResNetAttention4(nn.Module):
    def __init__(self):
        super(ResNetAttention4, self).__init__()
        self.model_name = 'ResNet18_att_softmax_fc'
        self.resnet = models.resnet18(pretrained=True)
        self.resnet_embedding_dim = self.resnet.fc.in_features

        # RNN Hyper-Params
        self.rnn_hidden_dim = 12
        self.bidirectional_att = True
        self.effective_rnn_hidden_dim = self.rnn_hidden_dim
        # bidirectional means 2*hidden dim
        if self.bidirectional_att:
            self.effective_rnn_hidden_dim *= 2

        self.recurrent = nn.LSTM(self.resnet_embedding_dim, self.rnn_hidden_dim, bidirectional=self.bidirectional_att)
        self.att_extractor = nn.Linear(self.effective_rnn_hidden_dim, 1)
        self.resnet.fc = nn.Linear(self.resnet_embedding_dim, 10)

        # Weight Init
        for name, param in self.recurrent.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

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
        x_att_in = x.permute(0, 2, 3, 1)                        # N, H, W, channels
        x_att_in = x_att_in.view(x_batch_size, -1, x_channel)   # N, H*W, channels
        x_att_in = x_att_in.permute(1, 0, 2)                    # H*W, N, channels

        # Recurrent
        att_vec, _ = self.recurrent(x_att_in)                            # H * W, N, embedding

        # Single Normalized Attention Extraction from Recurrent Embedded Vec
        att_vec_flat = att_vec.view(-1, self.effective_rnn_hidden_dim)  # H * W * N, embedding
        att = self.att_extractor(att_vec_flat)                          # H * W * N, 1

        att = att.view(-1, x_batch_size)                                # H * W, N
        att = F.softmax(att, dim=0)
        att = att.view(x.shape[2], x.shape[3], x_batch_size, 1)         # H, W, N, 1
        att = att.permute(2, 3, 0, 1)                                   # N, 1, H, W

        x = x*att                                                       # N, C, H, W
        x = x.view(x_batch_size, x_channel, -1)                         # N, C, H*W
        x = torch.sum(x, dim=2)
        x = self.resnet.fc(x)

        return {'logits': x, 'att': att}


if __name__ == '__main__':
    r = ResNetAttention2()
    print(r.model_name)
    # r.train()
    # r.zero_grad()
    #
    # for n, p in r.named_parameters():
    #     print(n, p.shape)

    # print('-'*70)
    # x = torch.randn(2, 3, 227, 227)
    # y, a = r(x)
    #
    # for n, p in r.named_parameters():
    #     print(n, p.shape, p)
    #     break
    # print(y.shape)
    # print(a.shape)
