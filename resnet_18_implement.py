import torch
import tqdm
import numpy as np
from datetime import datetime

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from parameters import *
from dataset import load_data


def train_resnet():
    batch_size = 32
    num_epochs = 2
    learning_rate = 0.0001
    T = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data(batch_size)
    # Train model
    resnet18 = models.resnet18(pretrained=True)
    num_dim = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_dim, 10)
    resnet18 = resnet18.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)
    run_time = datetime.now().strftime("%m-%d %H-%M")
    writer = SummaryWriter(os.path.join('runs', "teacher_"+run_time))

    for epoch in range(1, num_epochs + 1):
        print(f'Starting epoch {epoch}')
        resnet18.train()
        
        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = resnet18(images)
            predict = F.log_softmax(logits/T, dim=1)
            pred_max = torch.argmax(predict, dim = 1)
            loss = criterion(predict, labels)
            loss.backward()
            optimizer.step()
            accuracy = (pred_max == labels).sum().item() / pred_max.size()[0]
            # accuracies.append(accuracy)
            # print(f'Loss: {loss}')
            # print(f'Accuracy {accuracy}')
            writer.add_scalar('eval_train_acc', accuracy, (epoch-1)*len(train_loader)+i)
            writer.add_scalar('eval_train_loss', loss.item(), (epoch-1)*len(train_loader)+i)
            
        # Validate model
        resnet18.eval()
        t_losses = []
        t_acc = []
        print('Evaluating on VAL data')
        with torch.no_grad():
            for i, (t_images, t_labels) in enumerate(val_loader):
                t_images, t_labels = t_images.to(device), t_labels.to(device)
                t_logits = resnet18(t_images)
                t_pred = F.log_softmax(t_logits/T, dim=1)
                t_pred_max = torch.argmax(t_pred, dim=1)
                t_loss = criterion(t_pred, t_labels)
                t_accuracy = (t_pred_max == t_labels).sum().item() / t_pred_max.size()[0]
                t_losses.append(t_loss.item())
                t_acc.append(t_accuracy)

        t_avg_acc = np.mean(t_acc)
        t_avg_loss = np.mean(t_losses)
        
        print(f'Validation loss at the end of epoch {epoch}', t_avg_loss)
        print(f'Validation accuracy at the end of epoch {epoch}', t_avg_acc)
        writer.add_scalar('eval_val_loss', t_avg_loss, epoch-1)
        writer.add_scalar('eval_val_acc', t_avg_acc, epoch-1)

        print('Saving the model')
        torch.save(resnet18.state_dict(), os.path.join('models', f'resnet18_{T}_{run_time}.pt'))


if __name__ == '__main__':
    train_resnet()
