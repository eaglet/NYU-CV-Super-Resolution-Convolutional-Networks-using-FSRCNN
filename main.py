import configparser
import argparse
import os
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch import nn

from models import SRCNN, FSRCNN_S1, FSRCNN_S2, FSRCNN, SRCNN_WO_1, SRCNN_WO_2
from dataset import TrainDataset, EvalDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, required=True)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config_file)

train_file 		= config['MAIN']['TrainFile']
eval_file 		= config['MAIN']['EvalFile']
output_dir 	    = config['MAIN']['OutputDir']
model_type 	    = config['MODEL']['Type']
scale 	        = int(config['MODEL']['Scale'])
lr 			    = float(config['MODEL']['LR'])
batch_size		= int(config['MODEL']['BatchSize'])
epochs		    = int(config['MODEL']['Epochs'])
workers		    = int(config['MODEL']['Workers'])
seed		    = int(config['MODEL']['Seed'])

#Use tensorboard for visulization
tb = SummaryWriter("runs/" + model_type + "_x" + str(scale))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(seed)

#Model selection
if model_type == 'SRCNN':
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr * 0.1}
    ], lr = lr)

elif model_type == 'SRCNN_WO_1':
    model = SRCNN_WO_1().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr * 0.1}
    ], lr = lr)

elif model_type == 'SRCNN_WO_2':
    model = SRCNN_WO_2().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr * 0.1}
    ], lr = lr)

elif model_type == 'FSRCNN_S1':
    model = FSRCNN_S1(scale_factor=scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.last_part.parameters(), 'lr': lr * 0.1}
    ], lr=lr)

elif model_type == 'FSRCNN_S2':
    model = FSRCNN_S2(scale_factor=scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': lr * 0.1}
    ], lr=lr)

elif model_type == 'FSRCNN':
    model = FSRCNN(scale_factor=scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': lr * 0.1}
    ], lr=lr)


#Data preparation
train_dataset = TrainDataset(train_file)
eval_dataset = EvalDataset(eval_file)

train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = workers)
eval_dataloader = DataLoader(dataset = eval_dataset, batch_size=1)

best_count = 0
best_sum = 0
best_epoch = -1
best_weight = 0

# sample_idx = 0
#Train
for epoch in range(epochs):
    model.train()
    count = 0
    sum = 0

    with tqdm(total = (len(train_dataset) - len(train_dataset) % batch_size)) as progress:
        progress.set_description('epoch: {}/{}'.format(epoch, epochs - 1))

        for train in train_dataloader:
            data, labels = train
            data = data.to(device)
            labels = labels.to(device)
            preds = model(data)
            # print("data.shape:", data.shape)
            # print("preds.shape:", preds.shape, "labels.shape:", labels.shape)
            # input()
            loss = criterion(preds, labels)
            # print("loss:",loss.shape)
            # input()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum += loss.item() * len(data)
            count += len(data)

            avg = sum / count
            # tb.add_scalar('Loss/train', avg, sample_idx)
            progress.set_postfix(loss = '{:.6f}'.format(avg))
            progress.update(len(data))

        # sample_idx += 1

    tb.add_scalar('Loss/train', avg, epoch)
    torch.save(model.state_dict(), os.path.join(output_dir, 'epoch_{}.pth'.format(epoch)))

    model.eval()
    eval_count = 0
    eval_sum = 0

    for eval in eval_dataloader:
        data, labels = eval
        data = data.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(data).clamp(0.0, 1.0)

        eval_sum += (10. * torch.log10(1. / torch.mean((preds - labels) ** 2))) * len(data)
        eval_count += len(data)

    tb.add_scalar('psnr/eval', (eval_sum / eval_count), epoch)
    print('eval error: {:.2f}'.format(eval_sum / eval_count))

    if best_count == 0 or eval_sum / eval_count > best_sum / best_count:
        best_epoch = epoch
        best_count = eval_count
        best_sum = eval_sum
        best_weight = copy.deepcopy(model.state_dict())

print("Best epoch: {:.2f} with avg: {:.2f}".format(best_epoch, best_sum / best_count))
torch.save(best_weight, os.path.join(output_dir, model_type + "_model" + "_x" + str(scale) + ".pth"))
