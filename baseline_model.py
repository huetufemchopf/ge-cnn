import torch
import pytorch_lightning as pl


class z2cnn(pl.LightningModule):
    """ Implementation of the z2cnn baseline model as described in the paper. TODO: check layer norm vs batchnorm performance
    """
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=3)
        self.l2to6 = torch.nn.Conv2d(
            in_channels=20, out_channels=20, kernel_size=3)
        self.l7 = torch.nn.Conv2d(
            in_channels=20, out_channels=10, kernel_size=4)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.norm = torch.nn.BatchNorm2d(20)
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):

        x = torch.nn.functional.layer_norm(self.l1(x), self.l1(x).shape[-3:])
        x = torch.nn.functional.relu(x)

        x = torch.nn.functional.layer_norm(self.l2to6(x), self.l2to6(x).shape[-3:])
        x = torch.nn.functional.relu(x)
        x = self.pool(x)

        for i in range(4):

            x = torch.nn.functional.relu(torch.nn.functional.layer_norm(self.l2to6(x), self.l2to6(x).shape[-3:]))
            x = torch.nn.functional.relu(x)


        x = self.l7(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.fc(x)

        output = torch.nn.functional.log_softmax(x, dim=1)
        return output
