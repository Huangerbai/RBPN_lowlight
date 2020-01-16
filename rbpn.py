import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *
import torch.nn.functional as F
from dbpns import Net as DBPNS
from unet import Net as UNET


class Net_DBPN(nn.Module):
    def __init__(self, in_channels, out_channels, base_filter, feat, num_stages, n_resblock, nFrames):
        super(Net_DBPN, self).__init__()
        # base_filter=256
        # feat=64
        self.nFrames = nFrames

        kernel = 6
        stride = 2
        padding = 2


        # Initial Feature Extraction
        self.feat0 = ConvBlock(in_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(in_channels * 2, base_filter, 3, 1, 1, activation='prelu', norm=None)

        ###DBPNS
        self.DBPN = DBPNS(base_filter, feat, num_stages, 2)

        # Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        # Res-Block3
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)


        # Reconstruction
        self.output = ConvBlock((nFrames - 1) * feat, out_channels, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, neigbor):
        ### initial feature extraction
        feat_input = self.feat0(x)
        feat_frame = []
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j]), 1)))

        ####Projection
        Ht = []
        for j in range(len(neigbor)):
            h0 = self.DBPN(feat_input)
            h1 = self.res_feat1(feat_frame[j])

            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            Ht.append(h)
            feat_input = self.res_feat3(h)

        ####Reconstruction
        out = torch.cat(Ht, 1)
        output = self.output(out)

        return output


class Net_UNET(nn.Module):
    def __init__(self, in_channels, out_channels, base_filter, feat, num_stages, n_resblock, nFrames):
        super(Net_UNET, self).__init__()
        #base_filter=256
        #feat=64
        self.nFrames = nFrames

        
        #Initial Feature Extraction
        self.feat1 = ConvBlock(in_channels*2, base_filter, 3, 1, 1, activation='prelu', norm=None)

        ###UNET
        self.UNET = UNET(base_filter, feat)

        #Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)


        #Res-Block4
        modules_body4 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body4.append(ConvBlock(feat, base_filter, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat4 = nn.Sequential(*modules_body4)
        
        #Reconstruction
        self.output = ConvBlock((nFrames-1)*feat, out_channels*4, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x, neigbor):
        ### initial feature extraction

        feat_input = self.feat1(torch.cat((x, x),1))
        feat_frame=[]
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j]),1)))
        
        ####Projection
        Ht = []
        for j in range(len(neigbor)):
            h0 = self.UNET(feat_input)
            h1 = self.UNET(feat_frame[j])
            
            e = h0-h1
            e = self.res_feat2(e)
            h = h0+e
            Ht.append(h)
            feat_input = self.res_feat4(h)
        
        ####Reconstruction
        out = torch.cat(Ht,1)
        output = self.output(out)
        ps = nn.PixelShuffle(2)
        return ps(output)

class Net_UNET2(nn.Module):
    def __init__(self, in_channels, out_channels, base_filter, feat, num_stages, n_resblock, nFrames):
        super(Net_UNET2, self).__init__()
        #base_filter=256
        #feat=64
        self.nFrames = nFrames

        
        #Initial Feature Extraction
        self.feat1 = ConvBlock(in_channels*2, base_filter, 3, 1, 1, activation='prelu', norm=None)

        ###UNET
        self.conv0_1 = DoubleConv(base_filter, 32)
        self.conv1_2 = DownConv(32)
        self.conv2_3 = DownConv(64)
        self.conv3_4 = DownConv(128)
        self.conv4_5 = DownConv(256)

        self.Conv0_1 = DoubleConv(feat, 32)
        self.Conv1_2 = DownConv(32)
        self.Conv2_3 = DownConv(64)
        self.Conv3_4 = DownConv(128)
        self.Conv4_5 = DownConv(256)


        self.conv5_4 = UpCatConv(512)
        self.conv4_3 = UpCatConv(256)
        self.conv3_2 = UpCatConv(128)
        self.conv2_1 = UpCatConv(64)
        self.conv1_0 = ConvBlock(32, feat, 1, 1, 0, activation=None, norm=None)

        #Res-Block2
        modules_body2 = [
            ResnetBlock(512, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(512, 512, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)


        #Res-Block4
        modules_body4 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body4.append(ConvBlock(feat, 512, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat4 = nn.Sequential(*modules_body4)
        
        #Reconstruction
        self.output = ConvBlock((nFrames-1)*feat, out_channels*4, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def encode(self, x0):
        x1 = self.conv0_1(x0)
        x2 = self.conv1_2(x1)
        x3 = self.conv2_3(x2)
        x4 = self.conv3_4(x3)
        x5 = self.conv4_5(x4)
      
        return [x1,x2,x3,x4,x5]

    def encode1(self, x0):
        x1 = self.Conv0_1(x0)
        x2 = self.Conv1_2(x1)
        x3 = self.Conv2_3(x2)
        x4 = self.Conv3_4(x3)
        x5 = self.Conv4_5(x4)
      
        return x5


        
    def decode(self, x1, x2, x3, x4, x5):
        x4 = self.conv5_4(x5, x4)
        x3 = self.conv4_3(x4, x3)
        x2 = self.conv3_2(x3, x2)
        x1 = self.conv2_1(x2, x1)
        return self.conv1_0(x1)
 

    def forward(self, x, neigbor):
        ### initial feature extraction

        feat_input = self.feat1(torch.cat((x, x),1))
        feat_frame=[]
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j]),1)))

        [h1, h2, h3, h4, h5] = self.encode(feat_input)
        ####Projection
        Ht = []
        for j in range(len(neigbor)):
            [H1, H2, H3, H4, H5] = self.encode(feat_frame[j])
            e = H5 - h5
            e = self.res_feat2(e)
            h5 = h5+e
            h = self.decode(h1,h2,h3,h4,h5)
            Ht.append(h)
            h5 = self.encode1(h)
        
        ####Reconstruction
        out = torch.cat(Ht,1)
        output = self.output(out)
        ps = nn.PixelShuffle(2)
        return ps(output)

class Net_SID(nn.Module):
    def __init__(self, in_channels, out_channels, base_filter, feat, num_stages, n_resblock, nFrames):
        super(Net_SID, self).__init__()
        #base_filter=256
        #feat=64
        self.nFrames = nFrames

        
        #Initial Feature Extraction
        self.feat1 = ConvBlock(in_channels*2, base_filter, 3, 1, 1, activation='prelu', norm=None)

        ###UNET
        self.conv0_1 = DoubleConv(base_filter, 32)
        self.conv1_2 = DownConv(32)
        self.conv2_3 = DownConv(64)
        self.conv3_4 = DownConv(128)
        self.conv4_5 = DownConv(256)

        self.conv5_4 = UpCatConv(512)
        self.conv4_3 = UpCatConv(256)
        self.conv3_2 = UpCatConv(128)
        self.conv2_1 = UpCatConv(64)
        self.conv1_0 = ConvBlock(32, feat, 1, 1, 0, activation=None, norm=None)
        
        #Reconstruction
        self.conv0_out = ConvBlock(feat, out_channels*4, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def encode(self, x0):
        x1 = self.conv0_1(x0)
        x2 = self.conv1_2(x1)
        x3 = self.conv2_3(x2)
        x4 = self.conv3_4(x3)
        x5 = self.conv4_5(x4)
      
        return [x1,x2,x3,x4,x5]

        
    def decode(self, x1, x2, x3, x4, x5):
        x4 = self.conv5_4(x5, x4)
        x3 = self.conv4_3(x4, x3)
        x2 = self.conv3_2(x3, x2)
        x1 = self.conv2_1(x2, x1)
        return self.conv1_0(x1)
 

    def forward(self, x, neigbor):
        ### initial feature extraction

        import random
        j = random.randint(0,len(neigbor)-1)
        feat_input = self.feat1(torch.cat((x, neigbor[j]),1))

        [h1, h2, h3, h4, h5] = self.encode(feat_input)
        h = self.decode(h1,h2,h3,h4,h5)
        
        ####Reconstruction
        output = self.conv0_out(h)
        ps = nn.PixelShuffle(2)
        return ps(output)
