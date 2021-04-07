# network

import torch
from torch import nn
import torch.nn.functional as F

import settings as S


# UNET3D

class ConvUnit(nn.Module):
    """
        Convolution Unit: (Conv3D -> BatchNorm -> ReLu) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost

            nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

class EncoderUnit(nn.Module):
    """
    An Encoder Unit with the ConvUnit and MaxPool
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2),
            ConvUnit(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

class DecoderUnit(nn.Module):
    """
    ConvUnit and upsample with Upsample or convTranspose
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvUnit(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, s_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.s_channels = s_channels

        self.conv = ConvUnit(in_channels, s_channels)
        self.enc1 = EncoderUnit(s_channels, 2 * s_channels)
        self.enc2 = EncoderUnit(2 * s_channels, 4 * s_channels)
        self.enc3 = EncoderUnit(4 * s_channels, 8 * s_channels)
        self.enc4 = EncoderUnit(8 * s_channels, 16 * s_channels)
        self.enc5 = EncoderUnit(16 * s_channels, 16 * s_channels) # new
        
        self.dec0 = DecoderUnit(32 * s_channels, 8 * s_channels) # new
        self.dec1 = DecoderUnit(16 * s_channels, 4 * s_channels)
        self.dec2 = DecoderUnit(8 * s_channels, 2 * s_channels)
        self.dec3 = DecoderUnit(4 * s_channels, s_channels)
        self.dec4 = DecoderUnit(2 * s_channels, s_channels)
        self.out = OutConv(s_channels, n_classes)
        
        # crop 
        self.conv_crop = nn.ModuleList()
        self.enc_1_crop = nn.ModuleList()
        self.enc_2_crop = nn.ModuleList()
        self.enc_3_crop = nn.ModuleList()
        self.enc_4_crop = nn.ModuleList()         
        self.dec_1_crop = nn.ModuleList()
        self.dec_2_crop = nn.ModuleList()
        self.dec_3_crop = nn.ModuleList()
        self.dec_4_crop = nn.ModuleList()
        self.out_crop = nn.ModuleList()
        for i in range(len(S.landmarks)):
            self.conv_crop.append(ConvUnit(in_channels, s_channels))
            self.enc_1_crop.append(EncoderUnit(s_channels, 2 * s_channels))
            self.enc_2_crop.append(EncoderUnit(2*s_channels, 2 * s_channels))
            #self.enc_3_crop.append(EncoderUnit(4*s_channels, 8 * s_channels))
            #self.enc_4_crop.append(EncoderUnit(8*s_channels, 8 * s_channels))
            
            #self.dec_1_crop.append(DecoderUnit(16*s_channels, 4 * s_channels))
            #self.dec_2_crop.append(DecoderUnit(8*s_channels, 2 * s_channels))
            self.dec_3_crop.append(DecoderUnit(4*s_channels, s_channels))
            self.dec_4_crop.append(DecoderUnit(2 * s_channels, s_channels))
            self.out_crop.append(OutConv(s_channels, 1))
        

    def forward(self, x, crop):
        
        if crop == False:
            x1 = self.conv(x)
            x2 = self.enc1(x1)
            x3 = self.enc2(x2)
            x4 = self.enc3(x3)
            x5 = self.enc4(x4)
            x10 = self.enc5(x5)
    
            x11 = self.dec0(x5,x10)
            x6 = self.dec1(x11, x4)
            x7 = self.dec2(x6, x3)
            x8 = self.dec3(x7, x2)
            x9 = self.dec4(x8, x1)
            output = self.out(x9)
            return output
            
        else:
            # x = sample list
            # i.e. x[0] is cropped image around landmark 0    
            # want channel axis to become landmark axis
            for i in range(len(S.landmarks)):
                x1 = self.conv_crop[i](x[:,i,:,:,:,:])
                x2 = self.enc_1_crop[i](x1)
                x3 = self.enc_2_crop[i](x2)
                #x4 = self.enc_3_crop[i](x3)
                #x5 = self.enc_4_crop[i](x4)

                #x6 = self.dec_1_crop[i](x5, x4)
                #x7 = self.dec_2_crop[i](x6, x3)
                x8 = self.dec_3_crop[i](x3, x2)
                x9 = self.dec_4_crop[i](x8, x1)
                output = self.out_crop[i](x9)
                
                if i == 0:
                    # i am expecting output to have dimension B x C x H x W x D
                    outputs = output #.unsqueeze(1)
                else:
                    outputs = torch.cat([outputs,output], dim = 1)
            return outputs
            
    
class Transfer_model(nn.Module):
    def __init__(self, n_classes, s_channels, pre_trained_model):
        super().__init__()
        self.n_classes = n_classes
        self.s_channels = s_channels
        
        self.pre_trained = nn.Sequential(
        *list(pre_trained_model.children())[:-1]) # think asterix is unpacking
        self.out = OutConv(s_channels, n_classes)

    def forward(self, x): 
        x1 = self.pre_trained[0](x)
        x2 = self.pre_trained[1](x1)
        x3 = self.pre_trained[2](x2)
        x4 = self.pre_trained[3](x3)
        x5 = self.pre_trained[4](x4)
        
        # missing extra enc5 and dec0 added in 23/3/21 19:27
        
        x6 = self.pre_trained[5](x5,x4)
        x7 = self.pre_trained[6](x6, x3)
        x8 = self.pre_trained[7](x7, x2)
        x9 = self.pre_trained[8](x8, x1)
        
        output = self.out(x9)
        return output 
        
# SCNET 3D

dilation = [1,1,1]

# 4 features goes through test


class SCNET(nn.Module): # need to add bottleneck
    
    def __init__(self, in_channels, out_channels, features):
        super().__init__()
        
        #initialise sigma to model
        
        # Local appearance
        
        # initialise sigma
        #self.
        

        # block 1
        self.conv1 = self.conv_block1(in_channels, features, 3, 1) 
        self.conv2 = self.conv_block1(features, features, 3, 1)
        self.conv3 = self.conv_block1(features,features,3, 1)

        # block 2
        self.pool_block_2 = self.pool_layer_block_2()

        # block 3
        self.conv4 = self.conv_block3(features, 2*features, 3, 1) # note conv4 acts on pool, which acts on conv2
        self.conv5 = self.conv_block3(2*features, 2*features, 3, 1)
        self.conv6 = self.conv_block3(2*features,2*features,3, 1)

        # block 4
        self.pool_block_4 = self.pool_layer_block_4()

        # block 5
        self.conv7 = self.conv_block5(2*features, 4*features, 3, 1) # note conv7 acts on pool, which acts on conv5
        self.conv8 = self.conv_block5(4*features, 4*features, 3, 1)
        self.conv9 = self.conv_block5(4*features,4*features,3, 1)

        # block 6
        self.upsample_block_6 = self.upsample_layer_block_6(4*features, 2*features)

        # block 7
        # addition
        self.upsample_block_7 = self.upsample_layer_block_7(2*features, features)


        # block 8
        # addition
        self.conv10 = self.conv_block8(features, out_channels, 3, 1)

        # Spatial configuration

        # block 9
        self.pool_block_9 = self.pool_layer_block_9()

        # block 10
        self.conv11 = self.conv_block10(out_channels, features, 3, 1) 
        self.conv12 = self.conv_block10(features, features, 3, 1)
        self.conv13 = self.conv_block10(features,features,3, 1)

        # block 11
        self.upsample_block_11 = self.upsample_layer_block_11(features, out_channels)

        # block 12
        # multiplication


    def __call__(self, x):

        # block 1
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # block 2
        pool_block_2 = self.pool_block_2(conv2) # note middle one

        # block 3
        conv4 = self.conv4(pool_block_2)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        # block 4
        pool_block_4 = self.pool_block_4(conv5)

        # block 5
        conv7 = self.conv7(pool_block_4)
        conv8 = self.conv8(conv7)
        conv9 = self.conv9(conv8)

        # block 6
        upsample_block_6 = self.upsample_block_6(conv9)

        # block 7
        add_block_7 = conv6 + upsample_block_6
        upsample_block_7 = self.upsample_block_7(add_block_7)

        # block 8
        add_block_8 = upsample_block_7 + conv3
        conv10 = self.conv10(add_block_8)

        # block 9
        pool_block_9 = self.pool_block_9(conv10)

        # block 10
        conv11 = self.conv11(pool_block_9)
        conv12 = self.conv12(conv11)
        conv13 = self.conv13(conv12)

        # block 11
        upsample_block_11 = self.upsample_block_11(conv13)

        # block 12
        output = upsample_block_11 * conv10

        return output

    def conv_block1(self, in_channels, out_channels, kernel_size, padding):

        conv_block1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, dilation = dilation, padding = padding), # add dilation
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
        return conv_block1

    def pool_layer_block_2(self):

        pool_layer_block_2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        )

        return pool_layer_block_2
    
    def conv_block3(self, in_channels, out_channels, kernel_size, padding):

        conv_block2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, dilation = dilation, padding = padding), # add dilation
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
        return conv_block2

    def pool_layer_block_4(self):

        pool_layer_block_4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        )

        return pool_layer_block_4
    
    def conv_block5(self, in_channels, out_channels, kernel_size, padding):

        conv_block5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, dilation = dilation, padding = padding), # add dilation
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
        return conv_block5
    
    def upsample_layer_block_6(self, in_channels, out_channels):

        upsample_block_6 = nn.Sequential(
          nn.ConvTranspose3d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding=1) # add dilation
        )
        return upsample_block_6
    
    def upsample_layer_block_7(self, in_channels, out_channels):

        upsample_block_7 = nn.Sequential(
          nn.ConvTranspose3d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, dilation = dilation, output_padding=1) # add dilation
        )
        return upsample_block_7

    def conv_block8(self, in_channels, out_channels, kernel_size, padding):

        conv_block8 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, dilation = dilation, padding = padding), # add dilation
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
        return conv_block8

    def pool_layer_block_9(self):

        pool_layer_block_9 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        )

        return pool_layer_block_9

    def conv_block10(self, in_channels, out_channels, kernel_size, padding):

        conv_block10 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, dilation = dilation, padding = padding), # add dilation
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
        return conv_block10

    def upsample_layer_block_11(self, in_channels, out_channels):

        upsample_block_11 = nn.Sequential(
          nn.ConvTranspose3d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1,dilation = dilation, output_padding=1) # add dilation
        )
        return upsample_block_11

