import torch
import torch.nn as nn
import torch.nn.modules.conv as conv


class AddCoords(nn.Module):
    def __init__(self, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W, D)
        :return:
        """
        batch_size_tensor, channel_in_shape, y_dim, x_dim, z_dim = input_tensor.shape
        
        xx_ones = torch.ones([batch_size_tensor, x_dim], dtype=torch.float32) # e.g. (batch, x_dim)
        xx_ones = torch.unsqueeze(xx_ones, -1)      # e.g. (batch, x_dim, 1)
        a = torch.unsqueeze(torch.arange(y_dim, dtype = torch.float32), 0)
        xy_range = a.repeat([batch_size_tensor, 1]) # e.g. (batch, y_dim)
        xy_range = torch.unsqueeze(xy_range, 1)  # e.g. (batch, 1, y_dim)
        
        xx_channel = torch.matmul(xx_ones, xy_range) # e.g. (batch, x_dim, y_dim, 1)
        xx_channel = torch.unsqueeze(xx_channel, -1)
        
        xx_channel = xx_channel.repeat(1,1,1,z_dim)
        xx_channel = torch.unsqueeze(xx_channel, -1)             # e.g. (batch, x_dim, y_dim, z_dim, 1)
        
        yy_ones = torch.ones([batch_size_tensor, y_dim], dtype=torch.int32)  # e.g. (batch, y_dim)
        yy_ones = torch.unsqueeze(yy_ones, -1)                    # e.g. (batch, y_dim, 1)
        b = torch.unsqueeze(torch.arange(z_dim, dtype = torch.int32), 0)
        yz_range = b.repeat([batch_size_tensor, 1])             # (batch, z_dim)
        yz_range = torch.unsqueeze(yz_range, 1)                 # e.g. (batch, 1, z_dim)
        
        
        yy_channel = torch.matmul(yy_ones, yz_range)               # e.g. (batch, y_dim, z_dim)
        yy_channel = torch.unsqueeze(yy_channel, 1)             # e.g. (batch, 1, y_dim, z_dim)
        
        yy_channel = yy_channel.repeat(1,x_dim,1,1)
        yy_channel = torch.unsqueeze(yy_channel, -1)  # e.g. (batch, x_dim, y_dim, z_dim , 1)
        
        zz_ones = torch.ones([batch_size_tensor, z_dim], dtype=torch.int32)  # e.g. (batch, z_dim)
        zz_ones = torch.unsqueeze(zz_ones, 1)                    # e.g. (batch, 1, z_dim)
        c = torch.unsqueeze(torch.arange(x_dim, dtype = torch.int32), 0)
        xz_range = c.repeat([batch_size_tensor, 1])             # (batch, x_dim)
        xz_range = torch.unsqueeze(xz_range, -1)                 # e.g. (batch, x_dim, 1)
        
        zz_channel = torch.matmul(xz_range, zz_ones)               # e.g. (batch, x_dim, z_dim)
        zz_channel = torch.unsqueeze(zz_channel, 2)             # e.g. (batch, x_dim, 1, z_dim)
        
        zz_channel = zz_channel.repeat(1,1,y_dim,1)
        zz_channel = torch.unsqueeze(zz_channel, -1)  # e.g. (batch, x_dim, y_dim, z_dim , 1)
        
        
        xx_channel = (xx_channel) / (y_dim - 1)
        yy_channel = (yy_channel) / (z_dim - 1)
        zz_channel = (zz_channel) / (x_dim - 1)
        
        xx_channel = xx_channel*2 - 1                           # [-1,1]
        yy_channel = yy_channel*2 - 1
        zz_channel = zz_channel*2 - 1
        
        xx_channel = xx_channel.permute(0,4,2,1,3)
        yy_channel = yy_channel.permute(0,4,2,1,3)
        zz_channel = zz_channel.permute(0,4,2,1,3)
        
        
        xx_channel_mod = zz_channel.cuda()
        zz_channel_mod = yy_channel.cuda()
        yy_channel_mod = xx_channel.cuda()
        
        out = torch.cat([input_tensor, yy_channel_mod, xx_channel_mod, zz_channel_mod], axis=1)    # e.g. (batch, c+3, y_dim, x_dim, z_dim)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel_mod - 0.5, 2) +
                            torch.pow(yy_channel_mod - 0.5, 2) +
                            torch.pow(zz_channel_mod - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out



class CoordConv3d(conv.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.addcoords = AddCoords(with_r, use_cuda=use_cuda)
        self.conv = nn.Conv3d(in_channels + 3 + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W,D)
        output_tensor_shape: N,C_out,H_out,W_out, D_out）
        :return: CoordConv3d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out