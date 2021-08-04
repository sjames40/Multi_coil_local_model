from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from pdb import set_trace

class CNN_wo_skip(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(CNN_wo_skip, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc2_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc3_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc4_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        """
        Decoder part
        """
        self.dec5_1 = CNR2d(1 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(1 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec4_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(1 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec3_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(1 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec2_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')

        self.dec1_2 = CNR2d(1 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.fc = Conv2d(1 * self.nch_ker,        1 * self.nch_out, kernel_size=1, padding=0)

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        # pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(enc1))
        # pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(enc2))
        # pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(enc3))
        # pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(enc4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        # unpool4 = self.unpool4(dec5)
        # cat4 = torch.cat([enc4, dec5], dim=1)
        dec4 = self.dec4_1(self.dec4_2(dec5))

        # unpool3 = self.unpool3(dec4)
        # cat3 = torch.cat([enc3, dec4], dim=1)
        dec3 = self.dec3_1(self.dec3_2(dec4))

        # unpool2 = self.unpool2(dec3)
        # cat2 = torch.cat([enc2, dec3], dim=1)
        dec2 = self.dec2_1(self.dec2_2(dec3))

        # unpool1 = self.unpool1(dec2)
        # cat1 = torch.cat([enc1, dec2], dim=1)
        dec1 = self.dec1_1(self.dec1_2(dec2))

        x = self.fc(dec1)
        # x = torch.sigmoid(x)

        return x

class CNN_w_skip(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(CNN_w_skip, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc2_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc3_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.enc4_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(1 * self.nch_ker, 2 * 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        """
        Decoder part
        """
        self.dec5_1 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec4_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec3_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec2_1 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')

        self.dec1_2 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0)

        self.fc = Conv2d(1 * self.nch_ker,        1 * self.nch_out, kernel_size=1, padding=0)

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        # pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(enc1))
        # pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(enc2))
        # pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(enc3))
        # pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(enc4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        # unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, dec5], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        # unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, dec4], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        # unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, dec3], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        # unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, dec2], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = self.fc(dec1)
        # x = torch.sigmoid(x)

        return x

class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', snorm=False):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.snorm = snorm

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        """
        Decoder part
        """
        self.dec5_1 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.dec4_1 = CNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.dec3_1 = CNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.dec2_1 = CNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')

        self.dec1_2 = CNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.fc = Conv2d(1 * self.nch_ker,        1 * self.nch_out, kernel_size=1, padding=0)

    def forward(self, x):

        """
        Encoder part
        """
        #enc1 = self.enc1_1(x)
        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)
        
        #enc2 = self.enc2_1(pool1)
        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)
        
        #enc3 = self.enc3_1(pool2) 
        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))
        #dec3 = self.dec3_1(enc3)
         
        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))
        #dec2 = self.dec2_1(enc2)

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))
        #dec1 = self.dec1_1(enc1)
        x = self.fc(dec1)
        # x = torch.sigmoid(x)

        return x

class Hourglass(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', snorm=False):
        super(Hourglass, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.snorm = snorm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        """
        Decoder part
        """

        self.dec5_1 = CNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='nearest')
        #self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='bilinear')
        # self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(1 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.dec4_1 = CNR2d(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='nearest')
        #self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='bilinear')
        # self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(1 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.dec3_1 = CNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='nearest')
        #self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='bilinear')
        # self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(1 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.dec2_1 = CNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='nearest')
        #self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='bilinear')
        # self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')

        self.dec1_2 = CNR2d(1 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, snorm=self.snorm, relu=0.0)

        self.fc = Conv2d(1 * self.nch_ker,        1 * self.nch_out, kernel_size=1, padding=0)

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        # cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(unpool4))

        unpool3 = self.unpool3(dec4)
        # cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(unpool3))

        unpool2 = self.unpool2(dec3)
        # cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(unpool2))

        unpool1 = self.unpool1(dec2)
        # cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(unpool1))

        x = self.fc(dec1)
        # x = torch.sigmoid(x)

        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)

    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)  # multi-GPUs
    return net

