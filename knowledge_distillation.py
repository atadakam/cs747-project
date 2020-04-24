from resnet_18_implement import load_data
from parameters import *
from dataset import *
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

teacher_network = torch.load( os.path.join(curr_dir,'models', "resnet18_5_04-23 21-14.pt"))

batch_size = 32
num_epochs = 2
learning_rate = 0.0001


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
        x1 = F.relu(self.b1(self.conv1(x)))
        x2 = self.pool(x1)
        x3 = F.relu(self.b2(self.conv2(x2)))
        x4 = self.pool(x3)
        x5 = F.relu(self.b3(self.conv3(x4)))
        x6 = self.adaptive_pool(x5)
        x6 = x6.view(-1, x6.size(1))
        x7 = self.fc1(x6)
        return x7


def train_student():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = Student_network()
    model = model.to(device)
    train_loader, val_loader = load_data()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    run_time = datetime.now().strftime(("%m-%d %H-%M"))
    writer = SummaryWriter(os.path.join('runs', run_time))

    for epoch in range(1, num_epochs + 1):
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
            writer.add_scalar('student_train_acc', accuracy, (epoch - 1) * len(train_loader) + i)
            writer.add_scalar('student_train_loss', loss.item(), (epoch - 1) * len(train_loader) + i)

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
        writer.add_scalar('student_val_loss', t_avg_loss, epoch - 1)
        writer.add_scalar('student_val_acc', t_avg_acc, epoch - 1)


if __name__ == '__main__':
    train_student()