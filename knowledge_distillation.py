import tqdm
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from dataset import load_data
from parameters import *


class Student_network(nn.Module):
    def __init__(self):
        super(Student_network, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.b1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.b2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 20, 3)
        self.b3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.b1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.b2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.b3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, x.size(1))
        x = self.fc1(x)
        return x


class softCrossEntropy(nn.Module):
    def __init__(self):
        super(softCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num
        return loss


def train_student_plain():

    batch_size = 32
    num_epochs = 200
    learning_rate = 0.0001
    prev_t_avg_loss = float('inf')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = Student_network()
    model = model.to(device)
    train_loader, val_loader = load_data(batch_size)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    run_time = datetime.now().strftime(("%m-%d_%H-%M"))
    writer = SummaryWriter(os.path.join('runs', 'student_'+run_time))

    for epoch in range(1, num_epochs + 1):
        if epoch == 50:
            learning_rate = 0.5 * learning_rate

        if epoch == 125:
            learning_rate = 0.1 * learning_rate

        print(f'Starting epoch {epoch}')
        model.train()

        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            predict = F.log_softmax(logits, dim=1)
            pred_max = torch.argmax(predict, dim=1)
            loss = criterion(predict, labels)
            loss.backward()
            optimizer.step()
            accuracy = (pred_max == labels).sum().item() / pred_max.size()[0]
            writer.add_scalar('eval_train_acc', accuracy, (epoch - 1) * len(train_loader) + i)
            writer.add_scalar('eval_train_loss', loss.item(), (epoch - 1) * len(train_loader) + i)

        print('Saving the model')
        torch.save(model.state_dict(), os.path.join('models', f'student_{run_time}.pt'))

        # Validate model
        model.eval()
        t_losses = []
        t_acc = []
        print('Evaluating on VAL data')
        with torch.no_grad():
            for i, (t_images, t_labels) in enumerate(val_loader):
                t_images, t_labels = t_images.to(device), t_labels.to(device)
                t_logits = model(t_images)
                t_pred = F.log_softmax(t_logits, dim=1)
                t_pred_max = torch.argmax(t_pred, dim=1)
                t_loss = criterion(t_pred, t_labels)
                t_accuracy = (t_pred_max == t_labels).sum().item() / t_pred_max.size()[0]
                t_losses.append(t_loss.item())
                t_acc.append(t_accuracy)

        t_avg_acc = np.mean(t_acc)
        t_avg_loss = np.mean(t_losses)

        print(f'Validation loss at the end of epoch {epoch}', t_avg_loss)
        print(f'Validation accuracy at the end of epoch {epoch}', t_avg_acc)
        writer.add_scalar('eval_val_loss', t_avg_loss, epoch - 1)
        writer.add_scalar('eval_val_acc', t_avg_acc, epoch - 1)
        print('Saving the model')
        torch.save(model.state_dict(), os.path.join('models', f'student_{run_time}.pt'))

        # Check for best validation loss
        if t_avg_loss < prev_t_avg_loss:
            print('Saving best val loss model')
            torch.save(model.state_dict(), os.path.join('models', f'student_{run_time}_val.pt'))
            prev_t_avg_loss = t_avg_loss


def train_student_distilled():

    batch_size = 32
    num_epochs = 200
    learning_rate = 0.0001
    T = 3
    teacher_path = os.path.join(curr_dir, 'models', "resnet18_3_04-26 22-38.pt")
    prev_t_avg_hard_loss = float('inf')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    teacher_network = models.resnet18(pretrained=False)
    num_dim = teacher_network.fc.in_features
    teacher_network.fc = nn.Linear(num_dim, 10)
    teacher_network.load_state_dict(torch.load(teacher_path))
    teacher_network.eval()
    teacher_network = teacher_network.to(device)

    model = Student_network()
    model = model.to(device)
    train_loader, val_loader = load_data(batch_size)
    criterion_hard = nn.NLLLoss()
    criterion_soft = softCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    run_time = datetime.now().strftime('%m-%d %H-%M')

    writer = SummaryWriter(os.path.join('runs', 'distilled_'+run_time))

    for epoch in range(1, num_epochs + 1):
        print(f'Starting epoch {epoch}')
        model.train()

        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            predict = F.log_softmax(logits, dim=1)
            pred_max = torch.argmax(predict, dim=1)


            # Hard target
            loss_hard = criterion_hard(predict, labels)
            # print(f"Hard loss {loss_hard}")

            # Soft target
            with torch.no_grad():
                soft_labels = F.softmax(teacher_network(images), dim=1)
            loss_soft = criterion_soft(logits, soft_labels) * (T**2)

            # print(f"Soft loss {loss_soft}")

            total_loss = loss_hard+loss_soft
            total_loss.backward()
            optimizer.step()
            accuracy = (pred_max == labels).sum().item() / pred_max.size()[0]

            writer.add_scalar('eval_train_acc', accuracy, (epoch - 1) * len(train_loader) + i)
            writer.add_scalar('eval_train_loss', loss_hard.item(), (epoch - 1) * len(train_loader) + i)
            writer.add_scalar('eval_train_soft_loss', loss_soft.item(), (epoch - 1) * len(train_loader) + i)

        print('Saving the model')
        torch.save(model.state_dict(), os.path.join('models', f'distilled_{T}_{run_time}.pt'))

        # Validate model
        model.eval()
        t_losses_soft = []
        t_losses_hard = []
        t_acc = []
        print('Evaluating on VAL data')
        with torch.no_grad():
            for i, (t_images, t_labels) in enumerate(val_loader):
                t_images, t_labels = t_images.to(device), t_labels.to(device)
                t_logits = model(t_images)
                t_pred = F.log_softmax(t_logits, dim=1)
                t_pred_max = torch.argmax(t_pred, dim=1)

                # Hard target
                t_loss_hard = criterion_hard(t_pred, t_labels)

                # Soft target
                t_soft_targets = F.softmax(teacher_network(t_images), dim=1)
                t_loss_soft = criterion_soft(t_logits, t_soft_targets) * (T**2)
                t_total_loss = t_loss_hard + t_loss_soft
                t_accuracy = (t_pred_max == t_labels).sum().item() / t_pred_max.size()[0]
                t_acc.append(t_accuracy)
                t_losses_hard.append(t_loss_hard.item())
                t_losses_soft.append(t_loss_soft.item())

        t_avg_acc = np.mean(t_acc)
        t_avg_hard_loss = np.mean(t_losses_hard)
        t_avg_soft_loss = np.mean(t_losses_soft)

        print(f'Validation loss at the end of epoch {epoch}     : {t_avg_hard_loss:.4f}')
        print(f'Validation accuracy at the end of epoch {epoch} : {t_avg_acc:.4f}')

        writer.add_scalar('eval_val_loss', t_avg_hard_loss, epoch - 1)
        writer.add_scalar('eval_val_acc', t_avg_acc, epoch - 1)
        writer.add_scalar('eval_val_soft_loss', t_avg_soft_loss, epoch - 1)
        writer.add_scalar('eval_val_total_loss', t_total_loss, epoch-1)

        # Check for best validation loss
        if t_avg_hard_loss < prev_t_avg_hard_loss:
            print('Saving best val loss model')
            torch.save(model.state_dict(), os.path.join('models', f'distilled_{T}_{run_time}_val.pt'))
            prev_t_avg_hard_loss = t_avg_hard_loss


if __name__ == '__main__':
    train_student_distilled()
