# Copied from Stan's notebook - Super_resolution_and_inpainting_Stan.ipynb
# DO NOT CLEAN THIS CODE

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_plus_conv(in_channels: int, out_channels: int):
    """
     Makes UNet block
    :param in_channels: input channels
    :param out_channels: output channels
    :return: UNet block
    :source: https://www.kaggle.com/code/evgenia12/unet-ipynb
    """
    return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
          nn.InstanceNorm2d(out_channels, affine=True),
          nn.LeakyReLU(0.2, inplace=True),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
          nn.InstanceNorm2d(out_channels, affine=True),
          nn.LeakyReLU(0.2, inplace=True),
      )

def bilinear_upsample_concat_conv(x, skip, conv_block):

 # Bilinear upsample x to match skip's (H,W) to concatenate
        #x:         (N, Cx, Hx, Wx)   decoder feature map
        #skip:      (N, Cs, Hs, Ws)   encoder skip feature map
        #conv_block: input channels Cx+Cs

    x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    x_cat = torch.cat([x_up, skip], dim=1)
    return conv_block(x_cat)

class U_net_generator(nn.Module):
  def __init__(self, in_channels = 4):
    # Input channels is 4, 3 RGB channels + corrupted mask.
    super().__init__()
    # Encoder
    self.down1 = conv_plus_conv(in_channels, 32)
    self.down2 = conv_plus_conv(32, 64)
    self.down3 = conv_plus_conv(64, 128)
    self.down4 = conv_plus_conv(128, 256)
    # Bottleneck
    self.bottleneck = conv_plus_conv(256, 512)
    # Decoder
    self.up4 = conv_plus_conv(512 + 256, 256)
    self.up3 = conv_plus_conv(256 + 128, 128)
    self.up2 = conv_plus_conv(128 + 64, 64)
    self.up1 = conv_plus_conv(64 + 32, 32)
    # Output prediction is 3 RGB channels with removed mask
    self.out = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
    #Reduce feature map size between up/downsampling
    self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
      x1 = self.down1(x)
      p1 = self.downsample(x1)
      x2 = self.down2(p1)
      p2 = self.downsample(x2)
      x3 = self.down3(p2)
      p3 = self.downsample(x3)
      x4 = self.down4(p3)
      p4 = self.downsample(x4)
      b = self.bottleneck(p4)

      u4 = bilinear_upsample_concat_conv(b,  x4, self.up4)
      u3 = bilinear_upsample_concat_conv(u4, x3, self.up3)
      u2 = bilinear_upsample_concat_conv(u3, x2, self.up2)
      u1 = bilinear_upsample_concat_conv(u2, x1, self.up1)
      out = self.out(u1)
      return out


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator (pix2pix-style).
    Operates on local patches instead of a single global score.
    """
    def __init__(self, in_channels=3, base=64):
        super().__init__()

        # Basic conv block, same naming style as U-Net
        def conv_block(in_channels: int, out_channels: int, stride: int):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # First layer: no normalization (standard for GANs)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = conv_block(base, base * 2, stride=2)
        self.layer3 = conv_block(base * 2, base * 4, stride=2)
        self.layer4 = conv_block(base * 4, base * 8, stride=1)

        # Output patch logits
        self.out = nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        # x: (B, 3, H, W) RGB image
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.out(x)


# SR model - different conv_plus_conv without normalization
def conv_plus_conv_sr(in_channels: int, out_channels: int):
    """
    Same structure as before, without normalization.
    Lim et al. (and many more) have shown that batchNorm actually worsens results for SR: https://arxiv.org/pdf/1707.02921
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
    )

def bilinear_upsample_concat_conv_sr(x, skip, conv_block):
 # Bilinear upsample x to match skip's (H,W) to concatenate
        #x:         (N, Cx, Hx, Wx)   decoder feature map
        #skip:      (N, Cs, Hs, Ws)   encoder skip feature map
        #conv_block: nn.Module that expects input channels Cx+Cs
    x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    x_cat = torch.cat([x_up, skip], dim=1)
    return conv_block(x_cat)

class UNetSRGenerator(nn.Module):
    """
    U-Net generator for single-image super-resolution.
    Input  : bicubic-upsampled LR image
    Output : residual added on top of bicubic baseline
    """
    def __init__(self, base=32):
        super().__init__()

        # encoder
        self.down1 = conv_plus_conv_sr(3, base)
        self.down2 = conv_plus_conv_sr(base, base * 2)
        self.down3 = conv_plus_conv_sr(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = conv_plus_conv_sr(base * 4, base * 8)

        # decoder
        self.up3 = conv_plus_conv_sr(base * 8 + base * 4, base * 4)
        self.up2 = conv_plus_conv_sr(base * 4 + base * 2, base * 2)
        self.up1 = conv_plus_conv_sr(base * 2 + base, base)

        # predict residual (same resolution as input)
        self.out = nn.Conv2d(base, 3, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))

        # bottleneck
        b = self.bottleneck(self.pool(x3))

        # decoder
        u3 = bilinear_upsample_concat_conv_sr(b,  x3, self.up3)
        u2 = bilinear_upsample_concat_conv_sr(u3, x2, self.up2)
        u1 = bilinear_upsample_concat_conv_sr(u2, x1, self.up1)

        # residual prediction
        delta = self.out(u1)
        return delta
