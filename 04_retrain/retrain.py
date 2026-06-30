"""
═══════════════════════════════════════════════════════════════
CONTINUAL LEARNING — RETRAINING (Whole Dataset, No Fold Split)
═══════════════════════════════════════════════════════════════
Trains on buffer (old exemplars) + new_ood (new cluster patients).

Added vs base version:
  - CosineAnnealingLR  : lr 3e-4 -> 1e-6 over MAX_EPOCHS
                         high LR early (plasticity), low LR late (settle)
  - WeightedRandomSampler : buffer upweighted to match new_ood count
                            so each batch sees balanced old/new gradient
                            (weights computed dynamically from 'source')

Checkpoints:
  - best model by train_dice
  - every 10 epochs (all kept)
═══════════════════════════════════════════════════════════════
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import logging
import pytorch_lightning
import torch
import wandb
import numpy as np

from datetime import datetime
from torch.utils.data import WeightedRandomSampler
from monai.data import (CacheDataset, DataLoader,
                         decollate_batch, list_data_collate)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete, Compose, EnsureChannelFirstd, EnsureType,
    LoadImaged, NormalizeIntensityd, Orientationd,
    RandAdjustContrastd, RandAffined, RandBiasFieldd,
    RandCropByPosNegLabeld, RandFlipd, RandGaussianNoised,
    RandGaussianSmoothd, RandRicianNoised, RandScaleIntensityd,
    RandShiftIntensityd, Spacingd, SpatialPadd, SelectItemsd
)
from pytorch_lightning.loggers import WandbLogger

wandb.login()


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
#PRETRAINED_CKPT = ("/container_workspace/Code/CL/MAMA_MIA_Again/step_one/step_one_scanner/mtybh8y3/checkpoints/unet-fold0-best.ckpt")
PRETRAINED_CKPT = ("/container_workspace/Code/CL/MAMA_MIA_Again/ablation_studies/buffer_memory/checkpoints/lgca9pxj/epoch-500.ckpt")
#CL_JSON  = ("/container_workspace/Code/CL/MAMA_MIA_Again/ablation_studies/buffer_memory/step_two_albation_buffer.json")#("/container_workspace/Code/CL/MAMA_MIA_Again/model_feature/ood_buffer/cl_step_two.json")
CL_JSON  = ("/container_workspace/Code/CL/MAMA_MIA_Again/ablation_studies/buffer_memory/step_three_ablation_buffer_seed2.json")

#RESUME_CKPT = "/container_workspace/Code/CL/MAMA_MIA_Again/ablation_studies/buffer_memory/checkpoints/ut3vnfas/epoch-150.ckpt"



DATA_DIR = "./"
CKPT_DIR = ("/container_workspace/Code/CL/MAMA_MIA_Again/ablation_studies/buffer_memory/checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

LR          = 3e-4    # cosine annealing start
ETA_MIN     = 1e-6    # cosine annealing floor
MAX_EPOCHS  = 500
BATCH_SIZE  = 8
N_CROPS     = 4

logging.basicConfig(
    filename="cl_retrain_whole.txt",
    level=logging.INFO,
    format='%(message)s')


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def datafold_read(datalist, basedir, key="training"):
   with open(datalist) as f:
       json_data = json.load(f)
   json_data = json_data[key]

   path_keys = ("image", "label")
   for d in json_data:
       for k in path_keys:
           if k in d and isinstance(d[k], str) and len(d[k]) > 0:
               d[k] = os.path.join(basedir, d[k])

   return json_data, []

# def datafold_read(datalist, basedir, key="training"):
#     """
#     Load entries from the JSON — no fold split.
#     ABLATION (no-buffer arm): drop all entries with source == "buffer",
#     keeping only new_ood (new cluster) patients.
#     Only 'image' and 'label' path fields are joined with basedir;
#     metadata fields (source, manufacturer, scanner_model, ...) are left
#     untouched so the sampler can read them correctly.
#     """
#     with open(datalist) as f:
#         json_data = json.load(f)
#     json_data = json_data[key]

#     # ── Ablation filter: exclude buffer (old exemplar) samples ──
#     n_total = len(json_data)
#     json_data = [d for d in json_data
#                  if d.get("source") != "buffer"]
#     n_dropped = n_total - len(json_data)
#     print(f"  [no-buffer ablation] dropped {n_dropped} buffer "
#           f"samples, kept {len(json_data)} new_ood samples")

#     path_keys = ("image", "label")
#     for d in json_data:
#         for k in path_keys:
#             if k in d and isinstance(d[k], str) and len(d[k]) > 0:
#                 d[k] = os.path.join(basedir, d[k])

#     return json_data, []


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════
class CLNet(pytorch_lightning.LightningModule):

    def __init__(self, pretrained_path=None):
        super().__init__()

        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.15,
        )

        self.json_path = CL_JSON
        self.data_dir  = DATA_DIR

        self.loss_function = DiceCELoss(
            to_onehot_y=True, softmax=True,
            smooth_nr=1e-3, smooth_dr=1e-3)

        self.post_pred  = Compose([
            EnsureType("tensor", device="cpu"),
            AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([
            EnsureType("tensor", device="cpu"),
            AsDiscrete(to_onehot=2)])

        self.train_dice_metric = DiceMetric(
            include_background=False,
            reduction="mean", get_not_nans=False)

        self.train_step_losses = []

        # Load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading weights: {pretrained_path}")
            ckpt  = torch.load(pretrained_path, map_location='cpu')
            state = {k[7:]: v
                     for k, v in ckpt['state_dict'].items()
                     if k.startswith('_model.')}
            missing, unexpected = self._model.load_state_dict(
                state, strict=False)
            print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
        else:
            print("No pretrained weights — training from scratch")

    # ── Optimiser + Scheduler ─────────────────────────────────
    def configure_optimizers(self):
        weight_decay = 0.0009

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=LR, weight_decay=weight_decay)

        # CosineAnnealingLR: smooth decay LR -> ETA_MIN over MAX_EPOCHS
        #   - high LR early for plasticity (learning the new cluster)
        #   - low LR late to settle without destroying old knowledge
        #   - no monitoring signal -> immune to noisy train curve
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=ETA_MIN)

        wandb.log({
            "cl_lr"          : LR,
            "cl_weight_decay": weight_decay,
            "scheduler"      : "CosineAnnealingLR",
            "T_max"          : MAX_EPOCHS,
            "eta_min"        : ETA_MIN,
            "batch_size"     : BATCH_SIZE,
            "n_crops"        : N_CROPS,
        })

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval" : "epoch",
                "frequency": 1,
            },
        }

    def forward(self, x):
        return self._model(x)

    # ── Data ──────────────────────────────────────────────────
    def prepare_data(self):
        train_files, _ = datafold_read(
            self.json_path, self.data_dir, key="training")

        print(f"\n  JSON     : {self.json_path}")
        print(f"  Train    : {len(train_files)} samples (whole dataset)")

        # Breakdown by source (buffer / new_ood)
        sources = {}
        for d in train_files:
            src = d.get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        print(f"  Breakdown: {sources}")

        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(
                keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.644, 0.644, 1.2),
                mode=("bilinear", "nearest")),
            NormalizeIntensityd(
                keys=["image"],
                nonzero=False, channel_wise=False),
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=[192, 192, 112],
                mode="constant"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[192, 192, 64],
                num_samples=N_CROPS,
                image_key="image",
                pos=2, neg=1),
            RandFlipd(
                keys=["image", "label"],
                prob=0.43, spatial_axis=0),
            RandFlipd(
                keys=["image", "label"],
                prob=0.43, spatial_axis=1),
            RandFlipd(
                keys=["image", "label"],
                prob=0.43, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.35,
                spatial_size=[192, 192, 64],
                rotate_range=(0, 0, 0.26),
                scale_range=(0.2, 0.2, 0.2)),
            RandAdjustContrastd(
                keys=["image"], prob=0.12,
                gamma=(0.5, 4.5),
                invert_image=False, retain_stats=False),
            RandGaussianSmoothd(
                keys=["image"], prob=0.23,
                sigma_x=[0.5, 1.0],
                sigma_y=[0.5, 1.0],
                sigma_z=[0.5, 1.0]),
            RandScaleIntensityd(
                keys=["image"], prob=0.68, factors=0.3),
            RandShiftIntensityd(
                keys=["image"], prob=0.61, offsets=0.1),
            RandGaussianNoised(
                keys=["image"], prob=0.27,
                mean=0.0, std=0.1),
            RandBiasFieldd(
                keys=["image"], degree=3,
                coeff_range=(0.0, 0.1), prob=0.1),
            RandRicianNoised(
                keys=["image"], prob=0.1,
                mean=0.0, std=0.05,
                channel_wise=False,
                relative=False, sample_std=True),
            SelectItemsd(keys=["image", "label"]),
        ])

        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=0.7, num_workers=6)

        # Store raw files for sampler construction
        self._train_files = train_files
        self.val_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = BATCH_SIZE,
            shuffle     = True,
            num_workers = 10,
            collate_fn  = list_data_collate)

    def val_dataloader(self):
        return None

    # ── Training ──────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss   = self.loss_function(output, labels)
        self.train_step_losses.append(loss.item())
        outputs_post = [self.post_pred(i)
                        for i in decollate_batch(output)]
        labels_post  = [self.post_label(i)
                        for i in decollate_batch(labels)]
        self.train_dice_metric(
            y_pred=outputs_post, y=labels_post)
        return {"loss": loss}

    def on_train_epoch_end(self):
        mean_loss  = (sum(self.train_step_losses) /
                      len(self.train_step_losses))
        self.train_step_losses.clear()

        train_dice = self.train_dice_metric.aggregate().item()
        self.train_dice_metric.reset()

        current_lr = (self.trainer.optimizers[0]
                      .param_groups[0]["lr"])

        wandb.log({
            "train_loss": mean_loss,
            "train_dice": train_dice,
            "epoch"     : self.current_epoch,
            "lr"        : current_lr,
        })

        self.log("train_dice", train_dice, prog_bar=True)

        logging.info(
            f"Epoch {self.current_epoch} | "
            f"train_loss: {mean_loss:.4f} | "
            f"train_dice: {train_dice:.4f} | "
            f"lr: {current_lr:.2e}")


# ═══════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════
net = CLNet(pretrained_path=PRETRAINED_CKPT)

wandb_logger = WandbLogger(
    project="step_two_deep_breast_criteria",
    name=f"cl_retrain"
         f"{datetime.now().strftime('%Y%m%d_%H%M')}")

CKPT_DIR = os.path.join("/container_workspace/Code/CL/MAMA_MIA_Again/ablation_studies/buffer_memory/checkpoints", wandb_logger.experiment.id)
os.makedirs(CKPT_DIR, exist_ok=True)

# Save a checkpoint every 10 epochs — keeps ALL of them
class PeriodicCheckpoint(pytorch_lightning.Callback):
    def __init__(self, dirpath, every_n_epochs=10):
        self.dirpath        = dirpath
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # 1-indexed
        if epoch % self.every_n_epochs == 0:
            path = os.path.join(
                self.dirpath, f"epoch-{epoch:03d}.ckpt")
            trainer.save_checkpoint(path)
            print(f"  [ckpt] saved → {path}")

class BestDiceCheckpoint(pytorch_lightning.Callback):
    def __init__(self, dirpath):
        self.dirpath   = dirpath
        self.best_dice = 0.0
        self.best_path = None

    def on_train_epoch_end(self, trainer, pl_module):
        train_dice = float(
            trainer.callback_metrics.get("train_dice", 0.0))
        if train_dice > self.best_dice:
            self.best_dice = train_dice
            if self.best_path and os.path.exists(self.best_path):
                os.remove(self.best_path)
            self.best_path = os.path.join(
                self.dirpath,
                f"best-epoch{trainer.current_epoch+1:03d}"
                f"-dice{train_dice:.4f}.ckpt")
            trainer.save_checkpoint(self.best_path)
            print(f"  [ckpt] new best → {self.best_path}")

trainer = pytorch_lightning.Trainer(
    devices=[0],
    accelerator="gpu",
    max_epochs=MAX_EPOCHS,
    logger=wandb_logger,
    enable_checkpointing=False,
    gradient_clip_val=1.0,
    num_sanity_val_steps=0,
    log_every_n_steps=1,
    check_val_every_n_epoch=MAX_EPOCHS,
    limit_val_batches=0,
    callbacks=[PeriodicCheckpoint(CKPT_DIR, every_n_epochs=10),
               BestDiceCheckpoint(CKPT_DIR)],
    precision="bf16-mixed",
)

#trainer.fit(net, ckpt_path=RESUME_CKPT)

trainer.fit(net)

logging.info(f"\nCL retraining complete — whole dataset, no fold split")
print(f"\nDone. Checkpoints saved every 10 epochs + best by train_dice.")