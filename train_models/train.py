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

import sys
sys.path.append("/home/emma/Projets/Synthetic_images_segmentation/")
sys.path.append("/home/emma/Projets/Synthetic_images_segmentation/config_files/")
from transforms import preprocess, augment
from get_subjects import get_subjects
from load_model import load

#################
#   Parameters  #
#################

config_file = "config001"
data_infos = "dataset.json"

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
        patch_size = ctx["patch_size"]
        queue_length = ctx["queue_length"]
        samples_per_volume = ctx["samples_per_volume"] 

with open('./'+data_infos) as f:
    data_info = json.load(f)
    channel = data_info["channel_names"]["0"]
    suffixe_img = data_info["suffixe_img"]
    suffixe_labels = data_info["suffixe_labels"]
    num_classes = len(data_info["labels"])
    file_ending = data_info["file_ending"]
print(f"{num_classes} classes")

current_dateTime = datetime.now()
id_run = config_file + "_" + str(current_dateTime.day) + "-" + str(current_dateTime.month) + "-" + str(current_dateTime.year) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second) 
save_model_weights_path = "./weights/"+net_model+ "_" + id_run + ".pth"


#################
#   MONITORING  #
#################

logger = TensorBoardLogger("unet_logs", name=net_model+task)

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

train_image_paths = sorted(img_dir.glob('**/*'+suffixe_img+file_ending))
train_label_paths = sorted(labels_dir.glob('**/*'+suffixe_labels+file_ending))
test_img_paths = sorted(img_test_dir.glob('**/*'+file_ending))

print(len(train_image_paths))
print(len(train_label_paths))
print(len(test_img_paths))
assert len(train_image_paths) == len(train_label_paths)

train_subjects = get_subjects(train_image_paths, train_label_paths)
train_dataset = tio.SubjectsDataset(train_subjects)
print('training dataset size:', len(train_dataset), 'subjects')

test_subjects = get_subjects(test_img_paths)
test_set = tio.SubjectsDataset(test_subjects, transform=preprocess())

print('training dataset size:', len(test_set), 'subjects')

#######################
#   ONE SUBJECT PLOT  #
#######################
print("\n# ONE SUBJECT PLOT\n")

# plot_first_sub(train_dataset)

##########################
#   DATA TRANFORMATION   #
##########################
print("\n# DATA TRANFORMATION\n")

transform = tio.Compose([preprocess(num_classes), augment()])

#######################
#   TRAIN VAL SPLIT   #
#######################

print("\n# TRAIN VAL SPLIT\n")
num_subjects = len(train_dataset)

num_val_subjects = 3
num_train_subjects = num_subjects - num_val_subjects
splits = num_train_subjects, num_val_subjects
generator = torch.Generator().manual_seed(seed)
train_subjects, val_subjects = random_split(train_subjects, splits, generator=generator)

train_set = tio.SubjectsDataset(train_subjects, transform=transform)
val_set = tio.SubjectsDataset(val_subjects, transform=preprocess())

print(f"Training: {len(train_set)}")
print(f"Validation: {len(val_set)}")     
print(f"Test: {len(test_set)}")    

patch_sampler = tio.data.LabelSampler(
    patch_size=patch_size,
    label_name='seg',
    # label_probabilities=probabilities,
)

train_patches_queue = tio.Queue(
    train_set,
    queue_length,
    samples_per_volume,
    patch_sampler,
    num_workers=num_workers,
    shuffle_patches=False,
    shuffle_subjects=False
)

val_patches_queue = tio.Queue(
    val_set,
    queue_length,
    samples_per_volume,
    patch_sampler,
    num_workers=num_workers,
    shuffle_patches=False,
    shuffle_subjects=False
)

test_patches_queue = tio.Queue(
    test_set,
    queue_length,
    samples_per_volume,
    patch_sampler,
    num_workers=num_workers,
    shuffle_patches=False,
    shuffle_subjects=False
)

#################
#   LOAD DATA   #
#################
print("\n# LOAD DATA\n")

train_dataloader = DataLoader(train_patches_queue, batch_size, num_workers=num_workers, pin_memory=True)
val_dataloader = DataLoader(val_patches_queue, batch_size, num_workers=num_workers, pin_memory=True)
test_dataloader = DataLoader(test_patches_queue, batch_size, num_workers=num_workers, pin_memory=True)


#############
#   MODEL   #
#############
print(f"\n# MODEL : {net_model}\n")

model = load(None, net_model, lr, num_classes)

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
    # limit_train_batches=0.5 #For fast training
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


#############
#   DICE    #
#############
print("# DICE")

# def get_metrics(batch): # Return DICE score and Hausdorff distance 
#     get_dice = monai.metrics.DiceMetric(include_background=False, reduction="none")
#     get_hd = monai.metrics.HausdorffDistanceMetric(include_background=False, reduction="none")
  
#     inputs, targets, subjects = model.prepare_batch_subject(batch)
#     print(targets.shape)
#     outputs = model.net(inputs.to(model.device)).argmax(dim=1)
#     outputs_one_hot = torch.nn.functional.one_hot(outputs, num_classes=num_classes).permute(0, 4, 1, 2, 3)
#     get_dice(outputs_one_hot.to(model.device), targets.to(model.device))
#     get_hd(outputs_one_hot.to(model.device), targets.to(model.device))
#     subjects_list.append(subjects)

#         print(f"subjects : {subjects_list}")
#         dice = get_dice.aggregate()
#         print(f"DICE : {dice}")
#         get_dice.reset()
#         hd = get_hd.aggregate()
#         print(f"HD : {hd}")
#         get_hd.reset()

#     return dice.cpu().numpy().squeeze(), hd.cpu().numpy().squeeze(), subjects_list


################
#   VALIDATION #
################
# print("# VALIDATION")

# val_dice, val_hd, subject_list = get_metrics(val_dataloader)

# print("Moyenne Dice: ", val_dice.mean())
# print("Moyenne HD: ", val_hd.mean())
# print(subject_list)

# df_val = pd.DataFrame(subject_list, columns=['Subjects'])
# df_val['DICE'] = pd.Series(val_dice)
# df_val['HD'] = pd.Series(val_hd)
# print(df_val)
# df_val.to_csv("val-"+id_run+".csv", sep='\t')
    
################
#   TRAINING   #
################
# print("# TRAINING")

# dice, hd, subject_list = get_metrics(train_dataloader)
# print("Moyenne Dice: ", dice.mean())
# print("Moyenne HD: ", hd.mean())
# print(subject_list)

# df_train = pd.DataFrame(subject_list, columns=['Subjects'])
# df_train['DICE'] = pd.Series(dice)
# df_train['HD'] = pd.Series(hd)
# print(df_train)
# df_train.to_csv("train-"+id_run+".csv", sep='\t')

#################
#   INFERENCE   #
#################
# print("# INFERENCE")

def inference(img_set, save = True, metric = False):
    get_dice = monai.metrics.DiceMetric(include_background=False, reduction="none")
    get_hd = monai.metrics.HausdorffDistanceMetric(include_background=False, reduction="none")
    subjects_list = []
    for subject in img_set:
        grid_sampler = tio.inference.GridSampler(
            subject,
            patch_size,
            patch_overlap=4,
        )
        aggregator = tio.inference.GridAggregator(grid_sampler)
        loader = DataLoader(grid_sampler, batch_size=16)
        with torch.no_grad():
            for batch in loader:
                input_tensor = batch['img'][tio.DATA]
                locations = batch[tio.LOCATION]
                logits = model.net(input_tensor.cuda())
                labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                aggregator.add_batch(labels, locations)

                if metric:
                    gt_tensor = batch['seg'][tio.DATA]
                    subjects_tensor = batch['subject']
                    outputs_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(0, 1, 5, 2, 3, 4)
                    get_dice(outputs_one_hot.to(model.device), gt_tensor.to(model.device))
                    get_hd(outputs_one_hot.to(model.device), gt_tensor.to(model.device))
                    subjects_list.append(subjects_tensor.cpu().numpy())
            if metric:
                print(f"subjects : {subjects_list}")
                dice = get_dice.aggregate()
                print(f"DICE : {dice}")
                get_dice.reset()
                hd = get_hd.aggregate()
                print(f"HD : {hd}")
                get_hd.reset()

            
            output_tensor = aggregator.get_output_tensor()
            subject.add_image(tio.Image(type = tio.LABEL, tensor = output_tensor), "prediction")
            new_subject = subject.apply_inverse_transform(image_interpolation='linear')

            filename = subject.subject
            print(filename)
            meta = {
                'filename_or_obj' : filename
            }
            out_file = "./"+subject.subject
            if save :
                monai.data.NiftiSaver(out_file, output_postfix="pred").save(new_subject.prediction.data.to(torch.int32),meta_data=meta)
            if metric:
                return dice.cpu().numpy().squeeze(), hd.cpu().numpy().squeeze(), subjects_list

inference(val_set, metric = False)
inference(test_set)
inference(train_set)