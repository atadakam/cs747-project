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
                    entropy_att_loss, entropy_loss_coeff):

    prev_val_avg_loss = float('inf')

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
    if "SLURM_JOB_ID" in os.environ:
        run_id = int(os.environ["SLURM_JOB_ID"])
    else:
        run_id = datetime.now().strftime("%m-%d_%H-%M")

    model_prefix = model.model_name
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
    for epoch in range(num_epochs):
        logging.info(f'Starting epoch {epoch}')
        model.train()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        writer.add_scalar('param_lr', lr, epoch)

        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            logits = output['logits']
            att = output['att']
            predict = F.log_softmax(logits, dim=1)
            pred_max = torch.argmax(predict, dim=1)
            loss = criterion(predict, labels)
            writer.add_scalar('eval_train_loss', loss.item(), epoch * len(train_loader) + i)

            if entropy_att_loss:
                loss_entropy = criterion2(att)
                loss = loss + loss_entropy * entropy_loss_coeff
                writer.add_scalar('eval_train_att_entropy_loss', loss_entropy.item(), epoch * len(train_loader) + i)
                writer.add_scalar('eval_train_total_loss', loss.item(), epoch * len(train_loader) + i)
            loss.backward()
            optimizer.step()

            accuracy = (pred_max == labels).sum().item() / pred_max.size()[0]

            writer.add_scalar('eval_train_acc', accuracy, epoch * len(train_loader) + i)

        logging.info('Saving the model')
        torch.save(model.state_dict(), os.path.join(output_dir, 'models', f'{model_id}.pt'))

        # Validate model
        model.eval()
        val_losses = []
        val_losses_entropy = []
        val_acc = []
        logging.info('Evaluating on VAL data')
        with torch.no_grad():
            for i, (val_images, val_labels) in enumerate(val_loader):
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                output = model(val_images)
                val_logits = output['logits']
                val_att = output['att']
                val_pred = F.log_softmax(val_logits, dim=1)
                val_pred_max = torch.argmax(val_pred, dim=1)
                val_loss = criterion(val_pred, val_labels)
                if entropy_att_loss:
                    val_loss_entropy = criterion2(val_att)
                    val_losses_entropy.append(val_loss_entropy.item())
                val_accuracy = (val_pred_max == val_labels).sum().item() / val_pred_max.size()[0]
                val_losses.append(val_loss.item())
                val_acc.append(val_accuracy)

        val_avg_acc = np.mean(val_acc)
        val_avg_loss = np.mean(val_losses)
        if entropy_att_loss:
            val_avg_entropy_loss = np.mean(np.array(val_losses_entropy))
            val_avg_total_loss = np.mean(np.array(val_losses) + entropy_loss_coeff * np.array(val_losses_entropy))
            logging.info(f'Validation entropy loss at the end of epoch {epoch}   : {val_avg_entropy_loss:.4f}')
            logging.info(f'Validation total loss at the end of epoch {epoch}     : {val_avg_total_loss:.4f}')
            writer.add_scalar('eval_val_att_entropy_loss', val_avg_entropy_loss, epoch)
            writer.add_scalar('eval_val_total_loss', val_avg_total_loss, epoch)
            scheduler.step(val_avg_total_loss)
        else:
            scheduler.step(val_avg_loss)

        logging.info(f'Validation loss at the end of epoch {epoch}     : {val_avg_loss:.4f}')
        logging.info(f'Validation accuracy at the end of epoch {epoch} : {val_avg_acc:.4f}')
        writer.add_scalar('eval_val_loss', val_avg_loss, epoch)
        writer.add_scalar('eval_val_acc', val_avg_acc, epoch)

        # Check for best validation loss
        if val_avg_loss < prev_val_avg_loss:
            logging.info('Saving best val loss model')
            torch.save(model.state_dict(), os.path.join(output_dir, 'models', f'{model_id}_val.pt'))
            prev_val_avg_loss = val_avg_loss


if __name__ == '__main__':
    from attention_model import ResNetAttention1, ResNetAttention2
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', help='Attention model to train', choices=['resnet18_relu', 'resnet18_softmax'],
                        default='resnet18_softmax')
    parser.add_argument('--output_dir', '-o', help='Path to root folder for saving output files', default='.')
    parser.add_argument('--batch_size', '-bs', help='Batch size', default='32', type=int)
    parser.add_argument('--epochs', '-e', help='Epochs', default='150', type=int)
    parser.add_argument('--learning_rate', '-lr', help='Learning rate', default='0.01', type=float)
    parser.add_argument('--entropy_loss_flag', '-ef',
                        help='Flag to calculate entropy loss function of attention (default=False)',
                        default=False, action='store_true')
    parser.add_argument('--entropy_loss_coeff', '-efc', help='Coefficient of entropy loss in total loss calculation',
                        default=0.3, type=float, metavar='COEFFICIENT')

    args = parser.parse_args()

    if args.model == 'resnet18_relu':
        model = ResNetAttention1()
    elif args.model == 'resnet18_softmax':
        model = ResNetAttention2()
    else:
        raise ValueError(f'Invalid model {args.model}')

    train_attention(att_model=model, output_dir=args.output_dir, batch_size=args.batch_size, num_epochs=args.epochs,
                    learning_rate=args.learning_rate, entropy_att_loss=args.entropy_loss_flag,
                    entropy_loss_coeff=args.entropy_loss_coeff)
