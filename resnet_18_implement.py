import sys
from parameters import *
from dataset import *
import torch
from torch.utils.data import random_split
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import tqdm

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Transforms applied to the training data


# test_transform = transforms.Compose([ transforms.Resize(227),
#             transforms.ToTensor(),
#             normalize])

batch_size = 32
num_epochs = 2
learning_rate = 0.0001

def load_data():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(227, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = DistractedDriverDataset(annotation_path, train_dir, transform=train_transform)
    num_train = int(len(train_dataset)*0.8)
    lengths = [num_train, len(train_dataset)-num_train]
    train, val = random_split(train_dataset, lengths)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    return train_loader, val_loader


def train_resnet():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = load_data()
    # Train model
    resnet18 = models.resnet18(pretrained=True)
    num_dim = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_dim, 10)
    resnet18 = resnet18.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)
    run_time = datetime.now().strftime(("%m-%d %H-%M"))
    writer = SummaryWriter(os.path.join('runs', run_time))

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