import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
    EnsureType,
    NormalizeIntensityd,
    SpatialPadd,
    Resized,
    RandGaussianSmoothd,
    RandFlipd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandAffined,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandRicianNoised,
    SelectItemsd,
    )
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
import torch
import json
import wandb
import logging
from datetime import datetime


wandb.login()

from pytorch_lightning.loggers import WandbLogger
log_filename = f"training_two.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')

print_config()

root_dir ="./"

def datafold_read(datalist, basedir, fold, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)
    json_data = json_data[key]
    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    return tr, val

class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.15 #0.1021842475497251,
        )
        self.json_path="/container_workspace/Code/CL/MAMA_MIA_Again/step_one/step_one_p1.json"
        self.data_dir="./"
        self.fold=fold
        #self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True, smooth_nr=1e-3,smooth_dr=1e-3)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.train_dice_metric=DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.train_step_losses = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        
        train_files, val_files = datafold_read(self.json_path, self.data_dir, fold=self.fold)

        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.644, 0.644, 1.2), #median of training dataset
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
                SpatialPadd(keys= ["image", "label"],spatial_size=[193,192,112],mode="constant"), 
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=[192, 192, 64],
                    num_samples=4,
                    image_key="image",
                    pos=2,
                    neg=1,
                    #image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], prob=0.42830160546710866, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.42830160546710866, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.42830160546710866, spatial_axis=2),
                RandAffined(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=0.35167637309549354,
                    spatial_size=[192, 192, 64],
                    rotate_range=(0, 0, 0.26),
                    scale_range=(0.2, 0.2, 0.2)
                ),
                RandAdjustContrastd(keys=["image"], prob=0.12344079332514876, gamma=(0.5, 4.5), invert_image=False, retain_stats=False),
                RandGaussianSmoothd(keys=["image"], prob=0.23493566210493225, sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0]),
                RandScaleIntensityd(keys=["image"], prob=0.6785737089892544, factors=0.3),
                RandShiftIntensityd(keys=["image"], prob=0.6139000346754855, offsets=0.1),
                RandGaussianNoised(keys=["image"], prob=0.26796348398149694, mean=0.0, std=0.1),
                RandBiasFieldd(keys=["image"], degree=3, coeff_range=(0.0, 0.1), prob=0.1),
                RandRicianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05, channel_wise=False, 
                            relative=False, sample_std=True),
                SelectItemsd(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.644, 0.644, 1.2), #(the median of training set
                    mode=("bilinear", "nearest"),
                ),
                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
                SpatialPadd(keys= ["image", "label"],spatial_size=[192, 192, 64],mode="constant",), #spatial size should be the same as your model's
                SelectItemsd(keys=["image", "label"]),

            ]
        )
        
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=0.7, #0.1
            num_workers=12, #16
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=0.7, #1.0
            num_workers=12, #16
        )
        
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=15, #32,
            shuffle=True,
            num_workers=10, #16
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=10)
        return val_loader

    def configure_optimizers(self):
        factor = 0.4224757876118794
        patience = 3
        threshold = 0.005
        
        optimizer = torch.optim.AdamW(self._model.parameters(), 
                                      lr=0.0006030342439497735, weight_decay=0.0008945732976347424) #0.0006030342439497735
        
        
        wandb.log({
        "scheduler_factor": factor,
        "scheduler_patience": patience,
        "scheduler_threshold": threshold
    })
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, threshold=threshold),
                "monitor": "val_dice",
                "interval": "epoch",
                "frequency": 5,
            },
        }

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        
        self.train_step_losses.append(loss.item())
        
        outputs_post = [self.post_pred(i) for i in decollate_batch(output)]
        labels_post = [self.post_label(i) for i in decollate_batch(labels)]
        self.train_dice_metric(y_pred=outputs_post, y=labels_post)

        
        return {"loss": loss}

    def on_train_epoch_end(self):
        mean_train_loss = sum(self.train_step_losses) / len(self.train_step_losses)
        self.train_step_losses.clear()
        
        train_dice = self.train_dice_metric.aggregate().item()
        self.train_dice_metric.reset()

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        wandb.log({
            "train_loss": mean_train_loss,
            "train_dice": train_dice,
            "epoch": self.current_epoch,
            "lr": current_lr
        })
        self.train_dice_metric.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = [192, 192, 64]
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        
        per_image_dice_metric = DiceMetric(
        include_background=False, 
        reduction="none",  # This gives per-class scores
        get_not_nans=False
        )
    
        # Calculate per-image dice scores
        per_image_scores = per_image_dice_metric(y_pred=outputs, y=labels)
        per_image_dice_metric.reset()
        
        # Since batch_size=1, per_image_scores shape: [1, num_classes]
        dice_score = per_image_scores.cpu().numpy()[0, 0]
        
        logging.info(
        f"Epoch {self.current_epoch} | Val Image {batch_idx} | Dice: {dice_score:.4f}"
        )
        
        #wandb.log({
        #    f"val_dice_image_{batch_idx}": dice_score,
        #    "epoch": self.current_epoch,
        #})
        
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items, mean_val_dice = 0, 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        logging.info(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        wandb.log({
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss.item(),
            "epoch": self.current_epoch
        })
        self.log("val_dice", mean_val_dice, prog_bar=True) #
        self.validation_step_outputs.clear()  # free memory



fold=0
logging.info(f"--- Starting Fold {fold} ---")  
    # initialise the LightningModule  
net = Net()

# set up loggers and checkpoints
log_dir = os.path.join(root_dir, "logs")
wandb_logger = WandbLogger(project="step_one_scanner", name=f"unet_fold_{fold}")
checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
    filename=f"unet-fold{fold}-best",
    monitor="val_dice",
    mode="max",
    save_top_k=1,
    save_last=True,  # last epoch always saved
)
        
    # initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    devices=[0],
    accelerator="gpu",
    max_epochs=1000,
    logger=wandb_logger,
    enable_checkpointing=True,
    gradient_clip_val=1.0, #gradient clipping to avoid nan dice
    num_sanity_val_steps=1,
    log_every_n_steps=1,
    check_val_every_n_epoch=5,
    callbacks=checkpoint_callback,
    precision="bf16-mixed"
)
trainer.fit(net)
logging.info(f"train completed, best_metric: {net.best_val_dice:.4f} " f"at epoch {net.best_val_epoch}")