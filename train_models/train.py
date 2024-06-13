from pathlib import Path
from datetime import datetime

import torch
import torchio as tio
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import pandas as pd
import monai
import json

from tqdm.auto import tqdm

import matplotlib.pyplot as plt 

import sys
sys.path.append("/home/emma/Projets/synthetic-model-brain-segmentation/")
sys.path.append("/home/emma/Projets/synthetic-model-brain-segmentation/config_files/")
from transforms import augment, preprocess
from get_subjects import get_subjects
from load_model import load

#################
#   Parameters  #
#################

config_file = "config004"
data_infos = "dataset_reduced"

with open('./config_files/'+config_file+".json") as f:
        ctx = json.load(f)
        num_workers = ctx["num_workers"]
        num_epochs = ctx["num_epochs"]
        task = ctx["experiment_name"]
        rootdir_train_img = ctx["rootdir_train_img"]
        rootdir_train_labels = ctx["rootdir_train_labels"]
        rootdir_test_img = ctx["rootdir_test"]
        lr = ctx["initial_lr"]
        seed = ctx["seed"]
        net_model = ctx["net_model"]
        batch_size = ctx["batch_size"]
        dropout = ctx["dropout"]
        loss_type = ctx['loss_type']
        if ctx["patch"]:
            patch_size = ctx["patch_size"]
            queue_length = ctx["queue_length"]
            samples_per_volume = ctx["samples_per_volume"] 

with open('./config_files/'+data_infos+".json") as f:
    data_info = json.load(f)
    channel = data_info["channel_names"]["0"]
    suffixe_img = data_info["suffixe_img"]
    suffixe_labels = data_info["suffixe_labels"]
    num_classes = len(data_info["labels"])
    file_ending = data_info["file_ending"]
print(f"{num_classes} classes")

current_dateTime = datetime.now()
id_run = config_file + "_" + str(current_dateTime.day) + "-" + str(current_dateTime.month) + "-" + str(current_dateTime.year) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second) 
save_model_weights_path = "./weights/" + id_run + ".pth"


#################
#   MONITORING  #
#################

logger = TensorBoardLogger("unet_logs", name=config_file)
# tensorboard --logdir /home/emma/Projets/synthetic-model-brain-segmentation/unet_logs/config001

##############
#   Devices  #
##############

if torch.cuda.is_available():
    [print() for i in range(torch.cuda.device_count())]
    [print(f"Available GPUs : \n{i} : {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]
    device = "cuda" 
else:
    device = "cpu"
print(f"device used : {device}")


################
#   DATA PATH  #
################
print("\n# DATA PATH : \n")

img_dir=Path(rootdir_train_img)
print(f"Synthetic images in : {img_dir}")
labels_dir=Path(rootdir_train_labels)
print(f"Labels in : {labels_dir}")
img_test_dir=Path(rootdir_test_img)
print(f"Real test images in : {img_test_dir}")

####################
#   TRAINING DATA  #
####################
print(f"\n# TRAINING DATA : \n")

train_image_paths = sorted(img_dir.glob('**/*_'+suffixe_img+file_ending))
train_label_paths = sorted(labels_dir.glob('**/*_'+suffixe_labels+file_ending))
test_img_paths = sorted(img_test_dir.glob('**/*_T1w'+file_ending))

print(len(train_image_paths))
print(len(train_label_paths))
print(len(test_img_paths))
assert len(train_image_paths) == len(train_label_paths)

train_subjects = get_subjects(train_image_paths, train_label_paths)
print('training dataset size: ', len(train_subjects), ' subjects')

test_subjects = get_subjects(test_img_paths)
test_transform = tio.Compose([
                tio.ZNormalization(masking_method=tio.ZNormalization.mean)])
test_subjects_dataset = tio.SubjectsDataset(test_subjects, transform=test_transform)

print('test dataset size:', len(test_subjects_dataset), 'subjects')

#######################
#   ONE SUBJECT PLOT  #
#######################
print("\n# ONE SUBJECT PLOT\n")

# plot_first_sub(train_dataset)

##########################
#   DATA TRANFORMATION   #
##########################
print("\n# DATA TRANFORMATION\n")

transform_train = tio.Compose([preprocess(num_classes), augment()])
transform_val = tio.Compose([preprocess(num_classes)])

#######################
#   TRAIN VAL SPLIT   #
#######################

print("\n# TRAIN VAL SPLIT\n")
num_subjects = len(train_subjects)

num_val_subjects = 3
num_train_subjects = num_subjects - num_val_subjects
splits = num_train_subjects, num_val_subjects
generator = torch.Generator().manual_seed(seed)
train_subjects, val_subjects = random_split(train_subjects, splits, generator=generator)

train_subjects_dataset = tio.SubjectsDataset(train_subjects, transform=transform_train)
val_subjects_dataset = tio.SubjectsDataset(val_subjects, transform=transform_val)

print(f"Training: {len(train_subjects_dataset)}")
print(f"Validation: {len(val_subjects_dataset)}")     
print(f"Test: {len(test_subjects_dataset)}")    

if ctx["patch"]:
    patch_sampler = tio.data.LabelSampler(
        patch_size=patch_size,
        label_name='seg',
        # label_probabilities=probabilities,
    )

    train_set = tio.Queue(
        train_subjects_dataset,
        queue_length,
        samples_per_volume,
        patch_sampler,
        num_workers=num_workers,
        shuffle_patches=False,
        shuffle_subjects=False
    )

    val_set = tio.Queue(
        val_subjects_dataset,
        queue_length,
        samples_per_volume,
        patch_sampler,
        num_workers=num_workers,
        shuffle_patches=False,
        shuffle_subjects=False
    )
else:
    val_set = val_subjects_dataset
    train_set = train_subjects_dataset



#################
#   LOAD DATA   #
#################
print("\n# LOAD DATA\n")

train_dataloader = DataLoader(train_set, batch_size, num_workers=num_workers, pin_memory=True)
val_dataloader = DataLoader(val_set, batch_size, num_workers=num_workers, pin_memory=True)

#############
#   MODEL   #
#############
print(f"\n# MODEL : {net_model}\n")

# model = load("/home/emma/Projets/synthetic-model-brain-segmentation/weights/config001_11-6-2024-154543.pth", net_model, lr, num_classes)
model = load(None, net_model, lr, dropout, loss_type, num_classes)


## Trainer 

early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss",
    patience = 5,
)

trainer = pl.Trainer(
    max_epochs=num_epochs, # Number of pass of the entire training set to the network
    accelerator=device, 
    devices=1,
    precision=16,
    logger=logger,
    log_every_n_steps=1,
    # limit_train_batches=0.2 # For fast training
)

#################
#   TRAINING    #
#################
print("\n# TRAINING\n")


start = datetime.now()
print("Training started at", start)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)
print("Training duration:", datetime.now() - start)


#################
#   SAVE MODEL  #
#################
print("\n# SAVE MODEL\n")

torch.save(model.state_dict(), save_model_weights_path)

print("model saved in : " + save_model_weights_path)


#################
#   INFERENCE   #
#################
print("# INFERENCE")

def inference(img_set, save = True, metric = False, data = ""):
    get_dice = monai.metrics.DiceMetric(include_background=False, reduction="none")
    get_hd = monai.metrics.HausdorffDistanceMetric(include_background=False, reduction="none")
    subjects_list = []

    for subject in img_set:
        print(subject)
        grid_sampler = tio.inference.GridSampler(
            subject,
            patch_size,
            patch_overlap=10,
        )
        aggregator = tio.inference.GridAggregator(grid_sampler)
        patch_loader = DataLoader(grid_sampler)
        subject.clear_history()

        with torch.no_grad():
            for patches_batch in tqdm(patch_loader, unit='batch'):
                input_tensor = patches_batch['img'][tio.DATA]
                locations = patches_batch[tio.LOCATION]
                logits = model.net(input_tensor)
                labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                outputs = labels
                aggregator.add_batch(outputs, locations)

        output_tensor = aggregator.get_output_tensor()
        print(f"output tensor shape : {output_tensor.shape}")
        pred = tio.LabelMap(tensor=output_tensor, affine=subject.img.affine)
        # plt.imshow(output_tensor.cpu().numpy()[0,64,:,:])
        # plt.show()
        if metric:
            gt_tensor = subject['seg'][tio.DATA].unsqueeze(axis=0)
            print(f"gt tensor shape : {gt_tensor.shape}")
            outputs_one_hot = torch.nn.functional.one_hot(output_tensor.long(), num_classes=num_classes)
            print(f"outputs_one_hot shape : {outputs_one_hot.shape}")
            outputs_one_hot = outputs_one_hot.permute(0, 4, 1, 2, 3)
            print(f"outputs_one_hot shape permuted: {outputs_one_hot.shape}")
            get_dice(outputs_one_hot.to(model.device), gt_tensor.to(model.device))
            get_hd(outputs_one_hot.to(model.device), gt_tensor.to(model.device))
            subjects_list.append(subject.subject)

            print(f"subjects : {subjects_list}")
            dice = get_dice.aggregate()
            print(f"DICE : {dice}")
            get_dice.reset()
            hd = get_hd.aggregate()
            print(f"HD : {hd}")
            get_hd.reset()

            
        subject.add_image(pred, "prediction")
        new_subject = subject.apply_inverse_transform()

        filename = subject.subject
        print(filename)
        meta = {
            'filename_or_obj' : filename
        }
        
        out_file = "./out-predictions/"+data+"/"+subject.subject
        if save :
            monai.data.NiftiSaver(out_file, output_postfix="pred").save(new_subject.prediction.data,meta_data=meta)
            monai.data.NiftiSaver(out_file, output_postfix="img").save(new_subject.img.data,meta_data=meta)
    if metric:
        return dice.cpu().numpy().squeeze(), hd.cpu().numpy().squeeze(), subjects_list

print("# Validation")
dice_val, hd_val, sub_val = inference(val_subjects_dataset, metric = True, data="val")
print("# Test")
inference(test_subjects_dataset, metric = False, data="test")
print("# Training")
dice_train, hd_train, sub_train = inference(train_subjects_dataset, metric = True, data="train")

print(dice_val, hd_val, sub_val)
print(dice_train, hd_train, sub_train)