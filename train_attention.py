import os
import sys
import tqdm
import logging
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import load_data
from loss import EntropyLoss


def train_attention(att_model, output_dir, batch_size, num_epochs, learning_rate,
                    entropy_att_loss=False, entropy_loss_coeff=0.3):

    prev_t_avg_loss = float('inf')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = att_model
    model = model.to(device)
    train_loader, val_loader = load_data(batch_size)
    criterion = nn.NLLLoss()

    if entropy_att_loss:
        criterion2 = EntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=10, factor=0.2,
                                                           threshold=0.02, verbose=True,
                                                           cooldown=0)

    # SETUP IDs
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        run_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        run_id = datetime.now().strftime("%m-%d_%H-%M")

    model_prefix = 'att_lstm'
    model_id = f'{model_prefix}_{run_id}'

    # SETUP LOGGERS
    log_path = os.path.join(output_dir, 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logging.basicConfig(filename=os.path.join(log_path, f'{model_id}.log'),
                        level=logging.DEBUG, format='%(levelname)s|%(asctime)s | %(message)s',
                        datefmt='%m/%d_%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'MODEL ID: {model_prefix}_{run_id}')
    writer = SummaryWriter(os.path.join(output_dir, 'runs',  f'{model_id}'))

    # START TRAINING
    for epoch in range(1, num_epochs + 1):
        logging.info(f'Starting epoch {epoch}')
        model.train()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        writer.add_scalar('param_lr', lr, epoch-1)

        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            logits = output['logits']
            att = output['att']
            predict = F.log_softmax(logits, dim=1)
            pred_max = torch.argmax(predict, dim=1)
            loss = criterion(predict, labels)

            if entropy_att_loss:
                loss = loss + criterion2(att) * entropy_loss_coeff

            loss.backward()
            optimizer.step()
            accuracy = (pred_max == labels).sum().item() / pred_max.size()[0]
            writer.add_scalar('eval_train_acc', accuracy, (epoch - 1) * len(train_loader) + i)
            writer.add_scalar('eval_train_loss', loss.item(), (epoch - 1) * len(train_loader) + i)

        logging.info('Saving the model')
        torch.save(model.state_dict(), os.path.join(output_dir, 'models', f'{model_id}.pt'))

        # Validate model
        model.eval()
        t_losses = []
        t_acc = []
        logging.info('Evaluating on VAL data')
        with torch.no_grad():
            for i, (t_images, t_labels) in enumerate(val_loader):
                t_images, t_labels = t_images.to(device), t_labels.to(device)
                t_logits, t_att = model(t_images)
                t_pred = F.log_softmax(t_logits, dim=1)
                t_pred_max = torch.argmax(t_pred, dim=1)
                t_loss = criterion(t_pred, t_labels)
                t_accuracy = (t_pred_max == t_labels).sum().item() / t_pred_max.size()[0]
                t_losses.append(t_loss.item())
                t_acc.append(t_accuracy)

        t_avg_acc = np.mean(t_acc)
        t_avg_loss = np.mean(t_losses)

        logging.info(f'Validation loss at the end of epoch {epoch}     : {t_avg_loss:.4f}')
        logging.info(f'Validation accuracy at the end of epoch {epoch} : {t_avg_acc:.4f}')
        writer.add_scalar('eval_val_loss', t_avg_loss, epoch - 1)
        writer.add_scalar('eval_val_acc', t_avg_acc, epoch - 1)

        # Check for best validation loss
        if t_avg_loss < prev_t_avg_loss:
            logging.info('Saving best val loss model')
            torch.save(model.state_dict(), os.path.join(output_dir, 'models', f'{model_id}_val.pt'))
            prev_t_avg_loss = t_avg_loss

        scheduler.step(t_avg_loss)


if __name__ == '__main__':
    from attention_model import ResNetAttention1, ResNetAttention2
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', help='Attention model to train', choices=['resnet18_relu', 'resnet18_softmax'])
    parser.add_argument('--output_dir', '-o', help='Path to root folder for saving output files', default='.')
    parser.add_argument('--batch_size', '-bs', help='Batch size', default='32', type=int)
    parser.add_argument('--epochs', '-e', help='Epochs', default='150', type=int)
    parser.add_argument('--learning_rate', '-lr', help='Learning rate', default='0.01', type=float)
    parser.add_argument('--entropy_loss', help='Flat to calculate entropy loss function of attention',
                        default=False, type=bool)
    parser.add_argument('--entropy_loss_coeff', help='Flat to calculate entropy loss function of attention',
                        default=0.3, type=float)
    args = parser.parse_args()

    if args.model == 'resnet18_relu':
        model = ResNetAttention1()
    elif args.model == 'resnet18_softmax':
        model = ResNetAttention2()
    else:
        raise ValueError(f'Invalid model {args.model}')

    train_attention(att_model=model, output_dir=args.output_dir, batch_size=args.batch_size, num_epochs=args.epochs,
                    learning_rate=args.learning_rate, entropy_att_loss=args.entropy_loss,
                    entropy_loss_coeff=args.entropy_loss_coeff)
