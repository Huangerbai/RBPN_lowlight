import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from skimage import img_as_float
from random import randrange
import os.path
import rawpy
import glob

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def is_raw_file(filename):
    return any(filename.endswith(extension) for extension in [".ARW", ".CR2", ".tiff"])

def pack_raw(raw, ds):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    H = H//2
    W = W//2
    out = out[0:H:ds, 0:W:ds, :]
    return out

def load_raw_gt(filepath, ds):
    #random.shuffle(seq) #if random sequence
     
    gt_raw = rawpy.imread(filepath)
    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    target = np.float32(im / 65535.0)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    target = target[0:H:ds, 0:W:ds, :]
    return target

def load_raw_seq(dirpath, nFrames, ratio, ds):
    print(dirpath)
    in_files = sorted(glob.glob(join(dirpath, '*.ARW')))
    in_files = in_files[:nFrames]
    raw = rawpy.imread(in_files[0])
    input = np.minimum(pack_raw(raw, ds) * ratio, 1.0)
    neigbor = []
    for in_file in in_files[1:nFrames]:
        raw = rawpy.imread(in_file)
        neigbor.append(np.minimum(pack_raw(raw, ds) * ratio, 1.0))
    
    return input, neigbor

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]

        for i in seq:
            index = int(filepath[char_len-7:char_len-4])-i
            file_name=filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png'
            
            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len-7]+'{0:03d}'.format(index)+'.png').convert('RGB'),scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
    
    return target, input, neigbor

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len-7:char_len-4])+i
            file_name1=filepath[0:char_len-7]+'{0:03d}'.format(index1)+'.png'
            
            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp=input
                neigbor.append(temp)
            
    else:
        target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor

def modcrop(img, modulo):
    ishape = img.shape
    (ih, iw) = (ishape[0], ishape[1])
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img[0:ih, 0:iw, :]
    return img

def get_patch(img_in, img_tar, img_nn, patch_size, nFrames, ix=-1, iy=-1):
    ishape = img_in.shape
    (ih, iw) = (ishape[0], ishape[1])
    (th, tw) = (ih*2, iw*2)

    tp = 2 * patch_size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (2 * ix, 2 * iy)
    
    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    img_nn = [j[iy:iy + ip, ix:ix + ip, :] for j in img_nn] 
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch

def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = np.flipud(img_in)
        img_tar = np.flipud(img_tar)
        img_nn = [np.flipud(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = np.fliplr(img_in)
            img_tar = np.fliplr(img_tar)
            img_nn = [np.fliplr(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = np.rot90(img_in,2)
            img_tar = np.rot90(img_tar,2)
            img_nn = [np.rot90(j,2) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug
    
def rescale_img(img_in, scale=2):
    img_in = Image.fromarray(np.uint8(img_in*255)) 
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return np.float32(np.array(img_in) / 255.0)

class DatasetFromFolder(data.Dataset):
    def __init__(self, gt_dir, seq_dir, nFrames,  data_augmentation, file_list,  patch_size, ds, debug, transform=None):
        super(DatasetFromFolder, self).__init__()
        gtlist = sorted([line.rstrip() for line in open(join(gt_dir,file_list))])
        if not debug == 0:
            gtlist = gtlist[:2]
        seqlist = sorted([line.rstrip() for line in open(join(seq_dir,file_list))])
        if not debug == 0:
            seqlist = seqlist[:2]
        self.gt_filenames = [join(gt_dir,x) for x in gtlist]
        self.seq_dirnames = [join(seq_dir,x) for x in seqlist]
        self.ratios = [(float(gt_fn[9:-5])/0.016) for gt_fn in gtlist]
        self.nFrames = nFrames
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.targets = []
        self.inputs = []
        self.neigbors = []
        for index in range(len(self.seq_dirnames)):
            target = load_raw_gt(self.gt_filenames[index], ds)
            input, neigbor = load_raw_seq(self.seq_dirnames[index], self.nFrames, self.ratios[index], ds)
            self.targets.append(target)
            self.inputs.append(input)
            self.neigbors.append(neigbor)

    def __getitem__(self, index):
        target = self.targets[index]
        input = self.inputs[index]
        neigbor = self.neigbors[index]

        if self.patch_size != 0:
            input, target, neigbor, _ = get_patch(input,target,neigbor,self.patch_size, self.nFrames)
        
        if self.data_augmentation:
            input, target, neigbor, _ = augment(input, target, neigbor)
            
        
        bicubic = input.copy()
        bicubic[:,:,1] = (input[:,:,1]+input[:,:,2])/2
        bicubic[:,:,2] = input[:,:,3]
        bicubic = rescale_img(bicubic[:,:,:3])
        
        if self.transform:
            target = self.transform(np.ascontiguousarray(target,dtype=np.float32))
            input = self.transform(np.ascontiguousarray(input,dtype=np.float32))
            bicubic = self.transform(np.ascontiguousarray(bicubic,dtype=np.float32))
            neigbor = [self.transform(np.ascontiguousarray(j,dtype=np.float32)) for j in neigbor]

        return input, target, neigbor, bicubic

    def __len__(self):
        return len(self.seq_dirnames)

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, gt_dir, seq_dir, nFrames,  file_list, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        gtlist = [line.rstrip() for line in open(join(gt_dir,file_list))]
        seqlist = [line.rstrip() for line in open(join(seq_dir,file_list))]
        self.gt_filenames = [join(gt_dir,x) for x in gtlist]
        self.seq_dirnames = [join(seq_dir,x) for x in seqlist]
        self.ratios = [(float(gt_fn[9:-5])/0.016) for gt_fn in gtlist]
        self.nFrames = nFrames
        self.transform = transform
        self.targets = []
        self.inputs = []
        self.neigbors = []
        for index in range(len(self.seq_dirnames)):
            target = load_raw_gt(self.gt_filenames[index])
            input, neigbor = load_raw_seq(self.seq_dirnames[index], self.nFrames, self.ratios[index])
            self.targets.append(target)
            self.inputs.append(input)
            self.neigbors.append(neigbor)

    def __getitem__(self, index):
        target = self.targets[index]
        input = self.inputs[index]
        neigbor = self.neigbors[index]
    
        bicubic = input.copy()
        bicubic[:,:,1] = (input[:,:,1]+input[:,:,2])/2
        bicubic[:,:,2] = input[:,:,3]
        bicubic = rescale_img(bicubic[:,:,:3])

        if self.transform:
            target = self.transform(np.ascontiguousarray(target,dtype=np.float32))
            input = self.transform(np.ascontiguousarray(input,dtype=np.float32))
            bicubic = self.transform(np.ascontiguousarray(bicubic,dtype=np.float32))
            neigbor = [self.transform(np.ascontiguousarray(j,dtype=np.float32)) for j in neigbor]
        return input, target, neigbor, bicubic
      
    def __len__(self):
        return len(self.seq_dirnames)
