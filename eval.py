from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_test_set
from functools import reduce
import numpy as np

#from scipy.misc import imsave
import scipy.io as sio
import time
import cv2
import math
import pdb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=False)
parser.add_argument('--chop_forward', action='store_true',default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--gt_dir', type=str, default='../mydata/pair/Sony/long')
parser.add_argument('--seq_dir', type=str, default='../mydata/seq/Sony')
parser.add_argument('--test_list', type=str, default='test.txt')
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--model', default='weights/weights_7p_v2/2x_Titan-X4RBPNF7_epoch_4000.pth', help='sr pretrained base model')
parser.add_argument('--minsize', type=int, default=300000)
parser.add_argument('--unet', action='store_true', default=False)


opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --gpu_mode")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_test_set(opt.gt_dir, opt.seq_dir, opt.nFrames, opt.test_list)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.unet:
    from rbpn import Net_UNET as RBPN
else:
    from rbpn import Net_DBPN as RBPN
model = RBPN(in_channels=4, out_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor)

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
else:
    state_dict = torch.load(opt.model)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

#device = torch.device('cpu')
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
else:
    model = model.cpu()

def eval():
    model.eval()
    count=1
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        input, target, neigbor, bicubic = batch[0], batch[1], batch[2], batch[3]
        if cuda: 
            with torch.no_grad():
                input = Variable(input).cuda(gpus_list[0])
                target = Variable(target).cuda(gpus_list[0])
                bicubic = Variable(bicubic).cuda(gpus_list[0])
                neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
        else:
            with torch.no_grad():
                input = Variable(input).cpu()
                target = Variable(target).cpu()
                bicubic = Variable(bicubic).cpu()
                neigbor = [Variable(j).cpu() for j in neigbor]

        print(input.shape)

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, neigbor, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                prediction = model(input, neigbor) 
        
        if opt.residual:
            prediction = prediction + bicubic
       
        #print(prediction.size())
        #print(type(prediction))
        #print(target.size())
        #print(type(prediction))


        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        save_img(prediction.cpu().data, str(count), True)
        save_img(target.cpu().data, str(count), False)
        
        prediction=prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32).transpose(1,2,0)
        prediction = prediction*255.
        
        target = target.cpu()
        target = target.data[0].numpy().astype(np.float32).transpose(1,2,0)
        target = target*255.
        
        
        psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        avg_psnr_predicted += psnr_predicted
        count+=1
    
    print("PSNR_predicted=", avg_psnr_predicted/count)

def save_img(img, img_name, pred_flag):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img
    save_dir=os.path.join(opt.output, str(opt.nFrames)+'p')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if pred_flag:
        save_fn = save_dir +'/'+ img_name+'_'+opt.model_type+'_F'+str(opt.nFrames)+'.png'
    else:
        save_fn = save_dir +'/'+ img_name+'.png'
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def chop_forward(x, neigbor, model, scale, shave=8, min_size=opt.minsize, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        [x[:, :, 0:h_size, 0:w_size], [j[:, :, 0:h_size, 0:w_size] for j in neigbor]],
        [x[:, :, 0:h_size, (w - w_size):w], [j[:, :, 0:h_size, (w - w_size):w] for j in neigbor]],
        [x[:, :, (h - h_size):h, 0:w_size], [j[:, :, (h - h_size):h, 0:w_size] for j in neigbor]],
        [x[:, :, (h - h_size):h, (w - w_size):w], [j[:, :, (h - h_size):h, (w - w_size):w] for j in neigbor]]
        ]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batchs = inputlist[i:i+nGPUs]#torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                input_batch = input_batchs[0]
                for j in range(1, nGPUs):
                    input_batch[0] = torch.cat((input_batch[0],input_batchs[j][0]), dim=0)
                    input_batch[1] = [torch.cat((input_batch[1][k],input_batchs[j][1][k]), dim=0) for k in range(len(input_batch[1]))]
                print(input_batch[0].shape)
                output_batch = model(input_batch[0], input_batch[1])
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, 3, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
eval()
