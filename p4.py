import torch
import pytorch_lightning as pl
import math


class gconv_z2p4(pl.LightningModule):
    """ 
    group convolutionfrom z2 to p4:
    input tensor shape: [batch size, img channels, img_x, img_y] 
    weight tensor shape: [batch size, img channels, img_x, img_y]
    output shape: 

    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        self.angles = 4
        self.w = torch.nn.Parameter(w)
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        w = self.w  # [bs, channel, k,k]
        ws = w.size()
        # rotates the weight 0, 90, 180 and 270 degrees. 
        # Rotation axis is the perpendicular to the kernel
        weight_rot = []
        for a in range(self.angles):
            weight_rot.append(torch.rot90(w, a, (2, 3)))
        weight_rot = torch.cat(weight_rot, 1)
        # reshape weight
        weight_conv = weight_rot.reshape(-1, ws[1], ws[2], ws[3])
        out = torch.nn.functional.conv2d(x, weight_conv)
        out2 = out.reshape(out.size(0), -1, 4, out.size(2), out.size(3))
        return out2


class gconv_p4p4(pl.LightningModule):

    """ 
    group convolutionfrom p4 to p4:
    input tensor shape: [batch size, img channels, img_x, img_y] 
    weight tensor shape: [batch size, img channels, img_x, img_y]  
    output shape: 
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.angles = 4
        w = torch.empty(out_channels, in_channels,
                        self.angles, kernel_size, kernel_size)
        self.w = torch.nn.Parameter(w)
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, x):

        # x shape  [64, 10, 4, 26, 26]
        x = x.reshape(x.size(0), -1, x.size(3), x.size(4))  # bs, 40, 3,3
        w = self.w  # [10, 10, 4, 3, 3]
        ws = w.size()
        # TODO: torch.rot90(weight, ang, (3, 4))

        # rotates the weight 0, 90, 180 and 270 degrees
        # roll: loop through angle layers, such that all layers 
        # are represented in this layer's feature feature map
        weight_rot = []
        for ang in range(self.angles):
            weight_rot.append(torch.rot90(w.roll(ang, 2), ang, (3, 4)))
        weight_rot = torch.cat(weight_rot, 1)  # ([10, 40, 4, 3, 3])
        weight_rot = weight_rot.reshape(
            ws[0], -1, ws[3], ws[4])  # ([10, 160, 3, 3])
        weight_conv = weight_rot.reshape(
            self.angles * ws[0], self.angles * ws[1], ws[3], ws[4])  # ([40, 40, 3, 3])
        out = torch.nn.functional.conv2d(
            x, weight_conv)
        out2 = out.reshape(out.size(0), -1, 4, out.size(2), out.size(3))
        return out2

class BatchNorm4d(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features=num_features, **kwargs)
    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError("expected 6D input (got {}D input)".format(input.dim()))

if __name__ == "__main__":
    pass
