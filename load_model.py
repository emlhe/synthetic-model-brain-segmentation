import monai 
from monai.networks.nets import UNet
import torch 
from monai.networks.layers.factories import Norm
import pytorch_lightning as pl
import torchio as tio


def load(path, model, lr, n_class):
    if model == "unet":
        net = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_class,
            channels=(24, 48, 96, 192, 384),
            strides=(2, 2, 2, 2),
            norm = Norm.BATCH
        )
        optim=torch.optim.Adam

    model = Model(
        net=net,
        criterion=monai.losses.DiceCELoss(softmax=True),
        learning_rate=lr,
        optimizer_class=optim,
    )

    if path != None:
        model.load_state_dict(torch.load(path))
        model.eval()
    return model

class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_batch(self, batch):
        return batch["img"][tio.DATA], batch["seg"][tio.DATA]
    
    def prepare_batch_subject(self, batch):
        return batch["img"][tio.DATA], batch["seg"][tio.DATA], batch["subject"]


    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss