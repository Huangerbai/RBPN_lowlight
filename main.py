from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import pdb
import socket
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--debug', type=int, default=0, help="debug mode")
parser.add_argument('--downsample', type=int, default=1, help="downsample rate")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=4001, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=500, help='Snapshots')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--gt_dir', type=str, default='../mydata/pair/Sony/long')
parser.add_argument('--seq_dir', type=str, default='../mydata/seq/Sony')
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7, help="num of frames to use")
parser.add_argument('--patch_size', type=int, default=128, help='0 to use original frame size')
parser.add_argument('--pretrained_sr', default='3x_dl10VDBPNF7_epoch_84.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--unet', action='store_true', default=False, help='use U-Net instead of DBPN or not')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--file_list', type=str, default='list.txt')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--residual', type=bool, default=False)
#parser.add_argument('--max_clip', type=int, default=100, help='Max_norm for clip function')




opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, neigbor, bicubic = batch[0], batch[1], batch[2], batch[3]
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input, neigbor)

        if opt.residual:
            prediction = prediction + bicubic
            
        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.item()
        loss.backward()
        #nn.utils.clip_grad_norm(model.parameters(), opt.max_clip)
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+str(opt.downsample)+'x_'+hostname+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.orthogonal(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight)
        nn.init.constant_(m.bias, 0.0)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --gpu_mode")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
#test_set = get_test_set(opt.gt_dir, opt.seq_dir, opt.nFrames, opt.test_list)
train_set = get_training_set(opt.gt_dir, opt.seq_dir, opt.nFrames, opt.data_augmentation, opt.file_list, opt.patch_size, opt.downsample, opt.debug)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)


print('===> Building model')
if(opt.unet):
    from rbpn import Net_DBPN as RBPN
else:
    from rbpn import Net_UNET as RBPN
model = RBPN(in_channels=4, out_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames)


print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')
model.apply(weights_init)
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss(reduction='mean')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)
    #test()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if epoch % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if epoch % (opt.snapshots) == 0:
        checkpoint(epoch)
