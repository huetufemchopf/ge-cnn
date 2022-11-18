import torch
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from baseline_model import z2cnn
from gcnn import p4cnn
import parser


class ClassifierModule(pl.LightningModule):

    """
    Lighning module responsible for data loading, training and validation of the models. 
    """

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.lr = args.lr
        self.bs = args.batchsize

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):

        imgs, labels = batch
        labels_predicted = self.model(imgs)
        loss = torch.nn.functional.cross_entropy(labels_predicted, labels)
        self.log("train_loss:", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        acc = accuracy(labels_predicted, labels)
        self.log('train_acc_step', acc)

        return loss

    def train_dataloader(self):
        # loads MNIST training set
        train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomRotation([0, 360]),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(
                                                              (0.1307,), (0.3081,))
                                                          ])

        train_dataset = torchvision.datasets.MNIST(
            root="dataset", train=True, transform=train_transform, download=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.bs, shuffle=True)

        return train_loader

    def val_dataloader(self):
        # loads MNIST validation set and applies rotation to the images

        test_transform = torchvision.transforms.Compose([torchvision.transforms.RandomRotation([0, 360]),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                             (0.1307,), (0.3081,))
                                                         ])

        val_dataset = torchvision.datasets.MNIST(
            root="dataset", train=False, transform=test_transform, download=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.bs, shuffle=False)

        return val_loader

    def validation_step(self, batch, batch_idx):
        # validates model on validation set
        imgs, labels = batch
        labels_predicted = self.model(imgs)
        loss = torch.nn.functional.cross_entropy(labels_predicted, labels)
        acc = accuracy(labels_predicted, labels)
        self.log('validation_acc_step', acc)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, labels_predicted):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss']
                                for x in labels_predicted]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in labels_predicted]).mean()
        print("acc", avg_acc)
        self.log("val_loss:", avg_loss, on_epoch=True)
        return avg_loss


if __name__ == '__main__':

    args = parser.arg_parse()
    torch.manual_seed(args.seed)

    if args.model == "z2cnn":
        model = z2cnn()
    elif args.model == "p4cnn":
        model = p4cnn()

    # train model
    model = ClassifierModule(model, args)
    # , accelerator="gpu", devices=1)
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto")
    trainer.fit(model)

    # model = ClassifierModule.load_from_checkpoint("/lightning_logs/version_38/checkpoints/epoch=1-step=1875.ckpt")
    # model.eval()
