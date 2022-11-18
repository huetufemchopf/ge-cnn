import torch
import pytorch_lightning as pl
from p4 import gconv_p4p4, gconv_z2p4, BatchNorm4d


class p4cnn(pl.LightningModule):
    """Layers: 7 layers with g-convolutions and 10 channels each.
    """
    def __init__(self):
        super(p4cnn, self).__init__()

        self.conv1 = gconv_z2p4(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2to6 = gconv_p4p4(
            in_channels=10, out_channels=10, kernel_size=3)
        self.conv7 = gconv_p4p4(in_channels=10, out_channels=10, kernel_size=4)
        self.norm2d = torch.nn.BatchNorm2d(10)
        self.norm3d = torch.nn.BatchNorm3d(10)


    def forward(self, x):
        
        x = torch.nn.functional.relu(torch.nn.functional.layer_norm(
            self.conv1(x), self.conv1(x).shape[-4:])) #everything but batch
        x = torch.nn.functional.relu(torch.nn.functional.layer_norm(
            self.conv2to6(x), self.conv2to6(x).shape[-4:]))
        x = torch.nn.functional.relu(torch.nn.functional.layer_norm(
            self.conv2to6(x), self.conv2to6(x).shape[-4:]))
        x = torch.nn.functional.relu(torch.nn.functional.layer_norm(
            self.conv2to6(x), self.conv2to6(x).shape[-4:]))
        x = torch.nn.functional.relu(torch.nn.functional.layer_norm(
            self.conv2to6(x), self.conv2to6(x).shape[-4:]))
        x = torch.nn.functional.relu(torch.nn.functional.layer_norm(
            self.conv2to6(x), self.conv2to6(x).shape[-4:]))
        x = self.conv7(x)
        x = torch.amax(x,2) #maxvalue over rotations [bs, channels, rot, x, y] -->[bs, channels, x, y]
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).squeeze() #to [bs, channels]
        output = torch.nn.functional.softmax(x, dim=1)
        return output
