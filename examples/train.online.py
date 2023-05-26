#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import socket
import tqdm
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from min_speckle.datasets.SPI      import DataSampler
from min_speckle.model             import SpeckleEmbedding
from min_speckle.criterion         import TripletLoss
from min_speckle.utils             import init_logger, MetaLog, split_dataset, save_checkpoint, load_checkpoint, set_seed, init_weights
from min_speckle.semihard_selector import OnlineSemiHardSelector

from min_speckle.trans import RandomShift,          \
                              RandomRotate,         \
                              RandomPatch,          \
                              RandomBrightness,     \
                              RandomCenterCropZoom, \
                              RandomBrightness,     \
                              PoissonNoise,         \
                              GaussianNoise,        \
                              Binning,              \
                              Crop

# Set global seed...
seed = 0
set_seed(seed)

torch.autograd.set_detect_anomaly(True)

mpi_comm = None

logger = logging.getLogger(__name__)

# [[[ USER INPUT ]]]
timestamp_prev = None # "2023_0505_1249_26"
epoch          = None # 21

drc_chkpt = "chkpts"
fl_chkpt_prev   = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"
path_chkpt_prev = None if fl_chkpt_prev is None else os.path.join(drc_chkpt, fl_chkpt_prev)

# Set up parameters for an experiment...
drc_dataset   = 'fastdata'
fl_dataset    = '0000.fastdata'    # Raw, just give it a try
path_dataset  = os.path.join(drc_dataset, fl_dataset)

num_sample_per_label_train    = 40
num_sample_train              = num_sample_per_label_train * 100
num_sample_per_label_validate = num_sample_per_label_train // 2
num_sample_validate           = num_sample_train // 2

frac_train    = 0.5
frac_validate = 0.5

uses_mixed_precision = True

alpha        = 0.05565119
lr           = 10**(-3.0)
weight_decay = 1e-5

num_gpu     = 1
size_batch  = 40 * num_gpu
num_workers = 2  * num_gpu    # mutiple of size_sample // size_batch

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            File (Dataset)         : {path_dataset}
            Fraction    (train)    : {frac_train}
            Batch  size            : {size_batch}
            Number of GPUs         : {num_gpu}
            lr                     : {lr}
            weight_decay           : {weight_decay}
            alpha                  : {alpha}
            uses_mixed_precision   : {uses_mixed_precision}
            num_workers            : {num_workers}
            continued training???  : from {fl_chkpt_prev}

            More...

            Sample size (train)                          : {num_sample_train}
            Sample size (validate)                       : {num_sample_validate}
            Sample size (candidates per class, train)    : {num_sample_per_label_train}
            Sample size (candidates per class, validate) : {num_sample_per_label_validate}
            """

timestamp = init_logger(returns_timestamp = True)

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()


# [[[ DATASET ]]]
# Load raw data...
with open(path_dataset, 'rb') as fh:
    dataset_list = pickle.load(fh)
data_train   , data_val_and_test = split_dataset(dataset_list     , frac_train   , seed = None)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = None)

# Set up transformation rules
num_patch                    = 10
size_patch                   = 10
frac_shift_max               = 0.2
angle_max                    = 360
crop_center                  = (525, 525)
crop_window_size             = (195*2, 195*2)
path_brightness_distribution = "image_distribution_by_photon_count.npy"
trim_factor_max              = 0.2
block_size                   = 6

trans_list = (
    Crop(crop_center = crop_center, crop_window_size = crop_window_size),
    Binning(block_size = 6, mask = None),
    RandomRotate(angle_max = angle_max, order = 0), 
    ## RandomShift(frac_shift_max, frac_shift_max),
    RandomPatch(num_patch = num_patch, size_patch_y = size_patch, size_patch_x = size_patch, var_patch_y = 0.2, var_patch_x = 0.2),
    ## RandomCenterCropZoom(trim_factor_max),
)

# Define the training set
dataset_train = DataSampler( dataset_list         = data_train,
                             num_sample           = num_sample_train,
                             num_sample_per_label = num_sample_per_label_train,
                             mpi_comm             = mpi_comm,
                             trans_list           = trans_list, )
dataset_train.report()
dataloader_train = torch.utils.data.DataLoader( dataset_train,
                                                shuffle     = False,
                                                pin_memory  = True,
                                                batch_size  = size_batch,
                                                num_workers = num_workers, )

# Define validation set...
dataset_validate = DataSampler( dataset_list         = data_validate,
                                num_sample           = num_sample_validate,
                                num_sample_per_label = num_sample_per_label_validate,
                                mpi_comm             = mpi_comm,
                                trans_list           = trans_list, )
dataset_validate.report()
dataloader_validate = torch.utils.data.DataLoader( dataset_validate,
                                                   shuffle     = False,
                                                   pin_memory  = True,
                                                   batch_size  = size_batch,
                                                   num_workers = num_workers//2, )


# [[[ MODEL ]]]
# Prerequistite to initialize an embedding model -- image size...
img_sample = dataset_list[0][0][None,]   # 0 : first entry
                                         # 1 : 1st index points to an image
                                         # Image transform requires a batch dimension
                                         # img_sampe shape is (B, H, W)
for trans in trans_list:
    img_sample = trans(img_sample)
size_y, size_x = img_sample.shape[-2:]

# Initialize the embedding model...
dim_emb = 128
model   = SpeckleEmbedding( size_y  = size_y,
                            size_x  = size_x,
                            dim_emb = dim_emb, )

# Initialize weights...
model.apply(init_weights)

# Set device...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to gpu(s)...
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

# Initialize semi hard example selector...
semihard_selector = OnlineSemiHardSelector(model = model)


# [[[ CRITERION ]]]
criterion = TripletLoss(alpha = alpha)

# [[[ OPTIMIZER ]]]
param_iter = model.module.parameters() if hasattr(model, "module") else model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode           = 'min',
                                         factor         = 5e-1,
                                         patience       = 20,
                                         threshold      = 1e-4,
                                         threshold_mode ='rel',
                                         verbose        = True)


# [[[ TRAIN LOOP ]]]
max_epochs = 3000

# From a prev training???
epoch_min = 0
loss_min  = float('inf')
if path_chkpt_prev is not None:
    epoch_min, loss_min = load_checkpoint(model, optimizer, scheduler, path_chkpt_prev)
    ## epoch_min, loss_min = load_checkpoint(model, None, None, path_chkpt_prev)
    epoch_min += 1    # Next epoch
    logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")

print(f"Current timestamp: {timestamp}")

is_init = True
for epoch in tqdm.tqdm(range(max_epochs)):
    epoch += epoch_min

    # Uses mixed precision???
    if uses_mixed_precision: scaler = torch.cuda.amp.GradScaler()

    # Only log the first pass...
    logs_triplets = False
    if is_init:
        logs_triplets = True
        is_init = False

    # ___/ TRAIN \___
    # Turn on training related components in the model...
    model.train()

    # Fetch batches...
    train_loss_list = []
    batch_train = tqdm.tqdm(enumerate(dataloader_train), total = len(dataloader_train))
    for batch_idx, batch_entry in batch_train:
        # Unpack the batch entry and move them to device...
        batch_imgs, batch_labels, batch_metadata = batch_entry

        # Move data to device...
        batch_imgs   = batch_imgs.to(device, dtype = torch.float)
        batch_labels = batch_labels.to(device, dtype = torch.float)

        # Select semi hard examples from candidates...
        batch_a, batch_p, batch_n = semihard_selector(batch_imgs,
                                                      batch_labels,
                                                      batch_metadata,
                                                      logs_triplets = logs_triplets)

        # Forward, backward and update...
        if uses_mixed_precision:
            with torch.cuda.amp.autocast(dtype = torch.float16):
                # Forward pass...
                batch_emb_a = model(batch_a)
                batch_emb_p = model(batch_p)
                batch_emb_n = model(batch_n)

                # Calculate the loss...
                loss = criterion(batch_emb_a, batch_emb_p, batch_emb_n)
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass and optimization...
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass...
            batch_emb_a = model(batch_a)
            batch_emb_p = model(batch_p)
            batch_emb_n = model(batch_n)

            # Calculate the loss...
            loss = criterion(batch_emb_a, batch_emb_p, batch_emb_n)
            loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

            # Backward pass and optimization...
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reporting...
        train_loss_list.append(loss.item())

    train_loss_mean = np.mean(train_loss_list)
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean train loss = {train_loss_mean:.8f}")


    # ___/ VALIDATE \___
    model.eval()

    # Fetch batches...
    validate_loss_list = []
    batch_validate = tqdm.tqdm(enumerate(dataloader_validate), total = len(dataloader_validate))
    for batch_idx, batch_entry in batch_validate:
        # Unpack the batch entry and move them to device...
        batch_imgs, batch_labels, batch_metadata = batch_entry

        # Move data to device...
        batch_imgs   = batch_imgs.to(device, dtype = torch.float)
        batch_labels = batch_labels.to(device, dtype = torch.float)

        # Select semi hard examples from candidates...
        batch_a, batch_p, batch_n = semihard_selector(batch_imgs,
                                                      batch_labels,
                                                      batch_metadata,
                                                      logs_triplets = logs_triplets)

        # Forward only...
        with torch.no_grad():
            if uses_mixed_precision:
                with torch.cuda.amp.autocast(dtype = torch.float16):
                    # Forward pass...
                    batch_emb_a = model(batch_a)
                    batch_emb_p = model(batch_p)
                    batch_emb_n = model(batch_n)

                    # Calculate the loss...
                    loss = criterion(batch_emb_a, batch_emb_p, batch_emb_n)
                    loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus
            else:
                # Forward pass...
                batch_emb_a = model(batch_a)
                batch_emb_p = model(batch_p)
                batch_emb_n = model(batch_n)

                # Calculate the loss...
                loss = criterion(batch_emb_a, batch_emb_p, batch_emb_n)
                loss = loss.mean()    # Collapse all losses if they are scattered on multiple gpus

        # Reporting...
        validate_loss_list.append(loss.item())

    validate_loss_mean = np.mean(validate_loss_list)
    logger.info(f"MSG (device:{device}) - epoch {epoch}, mean val   loss = {validate_loss_mean:.8f}")

    # Report the learning rate used in the last optimization...
    lr_used = optimizer.param_groups[0]['lr']
    logger.info(f"MSG (device:{device}) - epoch {epoch} (lr used = {lr_used})")

    # Update learning rate in the scheduler...
    scheduler.step(validate_loss_mean)


    # ___/ SAVE CHECKPOINT??? \___
    if validate_loss_mean < loss_min:
        loss_min = validate_loss_mean

        fl_chkpt   = f"{timestamp}.epoch_{epoch}.chkpt"
        path_chkpt = os.path.join(drc_chkpt, fl_chkpt)
        save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path_chkpt)
        logger.info(f"MSG (device:{device}) - save {path_chkpt}")
