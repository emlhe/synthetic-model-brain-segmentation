import monai 
from monai.networks.nets import UNet
import torch 
from monai.networks.layers.factories import Norm
import pytorch_lightning as pl
import torchio as tio
import numpy as np


def load(weights_path, model, lr, dropout, loss_type, n_class, channels, epochs):
    if model == "unet":
        net = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_class,
            channels=channels, #(24, 48, 96, 192, 384),#(32, 64, 128, 256, 320, 320),#
            strides=np.ones(len(channels)-1, dtype=np.int8)*2,#(2, 2, 2, 2),
            norm = Norm.BATCH,
            dropout=dropout     
        )
        optim=torch.optim.AdamW

    if loss_type == "Dice":
        crit = monai.losses.DiceLoss(include_background=True,
        to_onehot_y=False,
        sigmoid=False,
        softmax=True,
        other_act=None,
        squared_pred=False,
        jaccard=False,
        reduction="mean",
        smooth_nr=1e-05,
        smooth_dr=1e-05,
        batch=True)# monai.losses.GeneralizedWassersteinDiceLoss
    elif loss_type == "DiceCE":
        crit = monai.losses.DiceCELoss(include_background=True,
        to_onehot_y=False,
        sigmoid=False,
        softmax=True,
        other_act=None,
        squared_pred=False,
        jaccard=False,
        reduction="mean",
        smooth_nr=1e-05,
        smooth_dr=1e-05,
        batch=True)# monai.losses.GeneralizedWassersteinDiceLoss
        
    model = Model(
        net=net,
        criterion= crit,
        learning_rate=lr,
        optimizer_class=optim,
        epochs = epochs,
    )

    if weights_path != None:
        model.load_state_dict(torch.load(weights_path))
        # model.eval() # deactivate dropout layers https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323
    return model



class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class, epochs):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.epochs = epochs
        # self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        lr_scheduler = {
        'scheduler': torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=self.epochs),
        'name': 'lr_scheduler'
        }
        return [optimizer], [lr_scheduler]

    def prepare_batch(self, batch):
        return batch["img"][tio.DATA], batch["seg"][tio.DATA]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        # print(x.shape)
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