import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from module import *


class Generator32(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=256, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf, upsample=True)
        self.block3 = GBlock(ngf, ngf, upsample=True)
        self.block4 = GBlock(ngf, ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(ngf)
        self.c5 = nn.Conv2d(ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = self.c5(h)
        y = torch.tanh(h)
        return y


class Discriminator32(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=128):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf)
        self.block2 = DBlock(ndf, ndf, downsample=True)
        self.block3 = DBlock(ndf, ndf, downsample=False)
        self.block4 = DBlock(ndf, ndf, downsample=False)
        self.l5 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l5(h)
        return y


class Generator64(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
    """

    def __init__(self, nz=128, ngf=1024, bottom_width=4):
        super().__init__()

        self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
        self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
        self.block2 = GBlock(ngf, ngf >> 1, upsample=True)
        self.block3 = GBlock(ngf >> 1, ngf >> 2, upsample=True)
        self.block4 = GBlock(ngf >> 2, ngf >> 3, upsample=True)
        self.block5 = GBlock(ngf >> 3, ngf >> 4, upsample=True)
        self.b6 = nn.BatchNorm2d(ngf >> 4)
        self.c6 = nn.Conv2d(ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c6.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = self.unfatten(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = self.c6(h)
        y = torch.tanh(h)
        return y


class Discriminator64(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """

    def __init__(self, ndf=1024):
        super().__init__()

        self.block1 = DBlockOptimized(3, ndf >> 4)
        self.block2 = DBlock(ndf >> 4, ndf >> 3, downsample=True)
        self.block3 = DBlock(ndf >> 3, ndf >> 2, downsample=True)
        self.block4 = DBlock(ndf >> 2, ndf >> 1, downsample=True)
        self.block5 = DBlock(ndf >> 1, ndf, downsample=True)
        self.l6 = SNLinear(ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        y = self.l6(h)
        return y

# FastGAN models

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class FastGAN_Generator(nn.Module):
    r"""
    FastGAN Generator block.
    Attributes:
        ngf (int): The number of filters in the generator.
        nz (int): The channel size of input noise vector.
        nc (int): The channel size of output.
        im_size (int): Size of generated image.
    """

    def __init__(self, ngf=64, nz=100, nc=3, im_size=256):
        super().__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init = InitBlock(nz, channel=ngf*16)
                                
        self.feat_8 = UpSample(nfc[4], nfc[8])
        self.feat_16 = UpSample(nfc[8], nfc[16])
        self.feat_32 = UpSample(nfc[16], nfc[32])
        self.feat_64 = UpSample(nfc[32], nfc[64])
        self.feat_128 = UpSample(nfc[64], nfc[128])  
        self.feat_256 = UpSample(nfc[128], nfc[256])

        self.sle_64 = SLEBlock(nfc[4], nfc[64])
        self.sle_128 = SLEBlock(nfc[8], nfc[128])
        self.sle_256 = SLEBlock(nfc[16], nfc[256])

        self.output_128 = nn.Sequential( 
            spectral_norm(nn.Conv2d(nfc[128], nc, 1, 1, 0, bias=False)),
            nn.Tanh())

        self.output = nn.Sequential(
            spectral_norm(nn.Conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)),
            nn.Tanh())

    def forward(self, x):
        feat_4 = self.init(x)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.sle_64(feat_4, self.feat_64(feat_32))
        feat_128 = self.sle_128(feat_8, self.feat_128(feat_64) )
        feat_256 = self.sle_256(feat_16, self.feat_256(feat_128) )
        
        return [self.output(feat_256), self.output_128(feat_128)]

class FastGAN_Discriminator(nn.Module):
    r"""
    FastGAN Discriminator block.
    Attributes:
        ndf (int): The number of filters in the discriminator.
        nc (int): The channel size of input image.
        im_size (int): Size of generated image.
    """

    def __init__(self, ndf=64, nc=3, im_size=256):
        super().__init__()

        nfc_multi = {8:16, 16:4, 32:2, 64:1, 128:0.5, 256:0.25}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.im_size = im_size

        self.feat_256 = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, nfc[256], 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.feat_128 = DownSample(nfc[256], nfc[128])
        self.feat_64 = DownSample(nfc[128], nfc[64])
        self.feat_32 = DownSample(nfc[64], nfc[32])
        self.feat_16 = DownSample(nfc[32], nfc[16])
        self.feat_8 = DownSample(nfc[16], nfc[8])
        
        self.logits = nn.Sequential(
            spectral_norm(nn.Conv2d(nfc[8] , nfc[8], 1, 1, 0, bias=False)),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(nfc[8], 1, 4, 1, 0, bias=False)))

        # Not in paper: In the official author implimentation, they add
        # SLE blocks in the discriminator, and report marked improvements
        # in performance
        self.sle_32_256 = SLEBlock(nfc[256], nfc[32])
        self.sle_16_128 = SLEBlock(nfc[128], nfc[16])
        self.sle_8_64 = SLEBlock(nfc[64], nfc[8])

        self.feat_16_small = nn.Sequential( 
            spectral_norm(nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(nfc[128],  nfc[64]),
            DownSample(nfc[64],  nfc[32]),
            DownSample(nfc[32],  nfc[16]))

        self.logits_small = spectral_norm(nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False))

        self.decoder = FastGANDecoder(nfc[8], nc)
        self.decoder_cropped = FastGANDecoder(nfc[16], nc)
        self.decoder_small = FastGANDecoder(nfc[16], nc)

    def forward(self, imgs, label):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]
        
        feat_256 = self.feat_256(imgs[0])
        feat_128 = self.feat_128(feat_256)
        feat_64 = self.feat_64(feat_128)
        feat_32 = self.sle_32_256(feat_256, self.feat_32(feat_64))
        feat_16 = self.sle_16_128(feat_128, self.feat_16(feat_32))
        feat_8 = self.sle_8_64(feat_64, self.feat_8(feat_16))
        logits = self.logits(feat_8)

        feat_16_small = self.feat_16_small(imgs[1])
        logits_small = self.logits_small(feat_16_small)

        if label == 'real':
            rec_img = self.decoder(feat_8)
            rec_img_small = self.decoder_small(feat_16_small)

            quadrant = random.randint(0, 3)
            feat_16_cropped = None
            if quadrant==0:
                feat_16_cropped = feat_16[:,:,:8,:8]
            elif quadrant==1:
                feat_16_cropped = feat_16[:,:,:8,8:]
            elif quadrant==2:
                feat_16_cropped = feat_16[:,:,8:,:8]
            else:
                feat_16_cropped = feat_16[:,:,8:,8:]

            rec_img_cropped = self.decoder_cropped(feat_16_cropped)

            return torch.cat([logits, logits_small], dim=1) , [rec_img, rec_img_small, rec_img_cropped], quadrant
        else:
            return torch.cat([logits, logits_small], dim=1)


class CondFastGAN_Generator(nn.Module):
    r"""
    FastGAN Generator block.
    Attributes:
        ngf (int): The number of filters in the generator.
        nz (int): The channel size of input noise vector.
        nc (int): The channel size of output.
        im_size (int): Size of generated image.
    """

    def __init__(self, num_classes=4, ngf=64, nz=100, nc=3, im_size=256):
        super().__init__()

        self.num_classes = num_classes
        self.linear0 = nn.Linear(in_features=nz, out_features=nz, bias=True)
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=nz)

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init = InitBlock(nz*2, channel=ngf*16)
                                
        self.feat_8 = UpSample(nfc[4], nfc[8])
        self.feat_16 = UpSample(nfc[8], nfc[16])
        self.feat_32 = UpSample(nfc[16], nfc[32])
        self.feat_64 = UpSample(nfc[32], nfc[64])
        self.feat_128 = UpSample(nfc[64], nfc[128])  
        self.feat_256 = UpSample(nfc[128], nfc[256])

        self.sle_64 = SLEBlock(nfc[4], nfc[64])
        self.sle_128 = SLEBlock(nfc[8], nfc[128])
        self.sle_256 = SLEBlock(nfc[16], nfc[256])

        self.output_128 = nn.Sequential( 
            spectral_norm(nn.Conv2d(nfc[128], nc, 1, 1, 0, bias=False)),
            nn.Tanh())

        self.output = nn.Sequential(
            spectral_norm(nn.Conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)),
            nn.Tanh())

    def forward(self, z, labels):
        # Embed labels, normalize, and concatenate with latents.
        embed = F.normalize(self.linear0(z), dim=1)
        proxy = F.normalize(self.embedding(labels), dim=1)
        x = torch.cat([embed, proxy], dim=1)
        feat_4 = self.init(x)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.sle_64(feat_4, self.feat_64(feat_32))
        feat_128 = self.sle_128(feat_8, self.feat_128(feat_64) )
        feat_256 = self.sle_256(feat_16, self.feat_256(feat_128) )
        
        return [self.output(feat_256), self.output_128(feat_128)]

class CondFastGAN_Discriminator(nn.Module):
    r"""
    FastGAN Discriminator block.
    Attributes:
        ndf (int): The number of filters in the discriminator.
        nc (int): The channel size of input image.
        im_size (int): Size of generated image.
    """

    def __init__(self, ndf=64, nc=3, im_size=256, num_classes=4):
        super().__init__()
        nfc_multi = {8:16, 16:4, 32:2, 64:1, 128:0.5, 256:0.25}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.im_size = im_size

        self.linear0 = nn.Linear(in_features=256*nfc[16], out_features=256, bias=True)
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=256)
        self.feat_256 = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, nfc[256], 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.feat_128 = DownSample(nfc[256], nfc[128])
        self.feat_64 = DownSample(nfc[128], nfc[64])
        self.feat_32 = DownSample(nfc[64], nfc[32])
        self.feat_16 = DownSample(nfc[32], nfc[16])
        self.feat_8 = DownSample(nfc[16], nfc[8])
        
        self.logits = nn.Sequential(
            spectral_norm(nn.Conv2d(nfc[8] , nfc[8], 1, 1, 0, bias=False)),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(nfc[8], 1, 4, 1, 0, bias=False)))

        # Not in paper: In the official author implimentation, they add
        # SLE blocks in the discriminator, and report marked improvements
        # in performance
        self.sle_32_256 = SLEBlock(nfc[256], nfc[32])
        self.sle_16_128 = SLEBlock(nfc[128], nfc[16])
        self.sle_8_64 = SLEBlock(nfc[64], nfc[8])

        self.feat_16_small = nn.Sequential( 
            spectral_norm(nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(nfc[128],  nfc[64]),
            DownSample(nfc[64],  nfc[32]),
            DownSample(nfc[32],  nfc[16]))

        self.logits_small = spectral_norm(nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False))

        self.decoder = FastGANDecoder(nfc[8], nc)
        self.decoder_cropped = FastGANDecoder(nfc[16], nc)
        self.decoder_small = FastGANDecoder(nfc[16], nc)

    def forward(self, imgs, label, labels):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]
        
        feat_256 = self.feat_256(imgs[0])
        feat_128 = self.feat_128(feat_256)
        feat_64 = self.feat_64(feat_128)
        feat_32 = self.sle_32_256(feat_256, self.feat_32(feat_64))
        feat_16 = self.sle_16_128(feat_128, self.feat_16(feat_32))
        feat_8 = self.sle_8_64(feat_64, self.feat_8(feat_16))
        logits = self.logits(feat_8)

        feat_16_small = self.feat_16_small(imgs[1])
        logits_small = self.logits_small(feat_16_small)

        embed = F.normalize(self.linear0(feat_16.view(feat_16.size(0),-1)), dim=1)
        proxy = F.normalize(self.embedding(labels), dim=1)

        if label == 'real':
            rec_img = self.decoder(feat_8)
            rec_img_small = self.decoder_small(feat_16_small)

            quadrant = random.randint(0, 3)
            feat_16_cropped = None
            if quadrant==0:
                feat_16_cropped = feat_16[:,:,:8,:8]
            elif quadrant==1:
                feat_16_cropped = feat_16[:,:,:8,8:]
            elif quadrant==2:
                feat_16_cropped = feat_16[:,:,8:,:8]
            else:
                feat_16_cropped = feat_16[:,:,8:,8:]

            rec_img_cropped = self.decoder_cropped(feat_16_cropped)

            return torch.cat([logits, logits_small], dim=1) , [rec_img, rec_img_small, rec_img_cropped], quadrant, embed, proxy
        else:
            return torch.cat([logits, logits_small], dim=1), embed, proxy