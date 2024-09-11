from pathlib import Path
from datetime import datetime
import os 

import torch
import torchio as tio
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import pandas as pd
from nilearn import plotting
import nibabel as nib
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

config_file = "config_chalcroft-002"
data_infos = "dataset_reduced"

with open('./config_files/'+config_file+".json") as f:
        ctx = json.load(f)
        num_workers = ctx["num_workers"]
        num_epochs = ctx["num_epochs"]
        task = ctx["experiment_name"]
        rootdir_train_img = ctx["rootdir_train_img"]
        rootdir_train_labels = ctx["rootdir_train_labels"]
        rootdir_test_img = ctx["rootdir_test-cap"]
        rootdir_test_labels = ctx["rootdir_test_labels-cap"]
        rootdir_test_img_healthy = ctx["rootdir_test-healthy"]
        rootdir_test_labels_healthy = ctx["rootdir_test_labels-healthy"]
        lr = ctx["initial_lr"]
        seed = ctx["seed"]
        net_model = ctx["net_model"]
        batch_size = ctx["batch_size"]
        dropout = ctx["dropout"]
        loss_type = ctx['loss_type']
        channels = ctx["channels"]
        n_layers = len(channels)
        if ctx["patch"]:
            patch_size = ctx["patch_size"]
            queue_length = ctx["queue_length"]
            samples_per_volume = ctx["samples_per_volume"] 

with open('./config_files/'+data_infos+".json") as f:
    data_info = json.load(f)
    channel = data_info["channel_names"]["0"]
    suffixe_img_train = data_info["suffixe_img-train"]
    suffixe_labels_train = data_info["suffixe_labels-train"]
    suffixe_img_test = data_info["suffixe_img-test"]
    suffixe_labels_test = data_info["suffixe_labels-test"]
    suffixe_img_test_healthy = data_info["suffixe_img-test-healthy"]
    suffixe_labels_test_healthy = data_info["suffixe_labels-test-healthy"]
    num_classes = len(data_info["labels"])
    file_ending = data_info["file_ending"]
    labels_names = list(data_info["labels"].keys())
print(f"{num_classes} classes : {labels_names}")

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
print(f"Real test stroke images in : {img_test_dir}")
img_test_healthy_dir=Path(rootdir_test_img_healthy)
print(f"Real test healthy images in : {img_test_healthy_dir}")

####################
#   TRAINING DATA  #
####################
print(f"\n# TRAINING DATA : \n")

train_image_paths = sorted(img_dir.glob('**/*'+suffixe_img_train+file_ending))
train_label_paths = sorted(labels_dir.glob('**/*'+suffixe_labels_train+file_ending))
test_img_paths = sorted(img_test_dir.glob('**/*_'+suffixe_img_test+file_ending))
test_label_paths = sorted(img_test_dir.glob('**/*_'+suffixe_labels_test+file_ending))
test_img_healthy_paths = sorted(img_test_healthy_dir.glob('**/*'+suffixe_img_test_healthy+file_ending))
test_label_healthy_paths = sorted(img_test_healthy_dir.glob('**/*'+suffixe_labels_test_healthy+file_ending))

print(len(train_image_paths))
print(len(train_label_paths))
print(len(test_img_paths))
print(len(test_label_paths))
print(len(test_img_healthy_paths))
print(len(test_label_healthy_paths))
assert len(train_image_paths) == len(train_label_paths)
assert len(test_img_paths) == len(test_label_paths)
assert len(test_img_healthy_paths) == len(test_label_healthy_paths)


train_subjects = get_subjects(train_image_paths, train_label_paths)
print('training dataset size: ', len(train_subjects), ' subjects')

test_subjects = get_subjects(test_img_paths, test_label_paths)
test_healthy_subjects = get_subjects(test_img_healthy_paths, test_label_healthy_paths)
remapping_fs = {2:3,3:2,4:1,5:1,7:3,8:2,10:4,11:4,12:4,13:4,14:1,15:1,16:3,17:2,18:2,24:1,26:2,28:2,29:0,30:0,41:3,42:2,43:1,44:1,46:3,47:2,49:4,50:4,51:4,52:4,53:2,54:2,58:2,60:2,61:0,62:0,64:5,72:0,77:3,85:0}
test_transform = tio.Compose([
                tio.Resample(1),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean)])


test_transform_remapping = tio.Compose([
                tio.Resample(1),
                tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                tio.RemapLabels(remapping_fs),
                tio.SequentialLabels()])

test_subjects_dataset = tio.SubjectsDataset(test_subjects, transform=test_transform)
test_healthy_subjects_dataset = tio.SubjectsDataset(test_healthy_subjects, transform=test_transform_remapping)
print('test AVC dataset size:', len(test_subjects_dataset), 'subjects')
print('test healthy dataset size:', len(test_healthy_subjects_dataset), 'subjects')

#######################
#   ONE SUBJECT PLOT  #
#######################
# print("\n# ONE SUBJECT PLOT\n")

# sub_tmp = nib.Nifti1Image(test_healthy_subjects_dataset[1]["img"][tio.DATA].numpy().squeeze(), test_healthy_subjects_dataset[1]["img"].affine)
# plotting.plot_img(sub_tmp, display_mode='mosaic')
# plt.show()


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

model = load("weights/config_chalcroft-002_10-7-2024-18457.pth",net_model, lr, dropout, loss_type, num_classes, channels, num_epochs)

# model = load(None, net_model, lr, dropout, loss_type, num_classes, channels, num_epochs)


## Trainer 

#early_stopping = pl.callbacks.early_stopping.EarlyStopping(
#     monitor="val_loss",
#     patience = 5,
# )

#lr_monitor = LearningRateMonitor(logging_interval='step')

# trainer = pl.Trainer(
#     max_epochs=num_epochs, # Number of pass of the entire training set to the network
#     accelerator=device, 
#     devices=1,
#     precision=16,
#     logger=logger,
#     log_every_n_steps=1,
#     # callbacks=[lr_monitor],
#     # limit_train_batches=0.2 # For fast training
# )

# #################
# #   TRAINING    #
# #################
# print("\n# TRAINING\n")


# start = datetime.now()
# print("Training started at", start)
# trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)
# print("Training duration:", datetime.now() - start)


#################
#   SAVE MODEL  #
#################
# print("\n# SAVE MODEL\n")

# torch.save(model.state_dict(), save_model_weights_path)

# print("model saved in : " + save_model_weights_path)

#################
#   INFERENCE   #
#################
print("# INFERENCE")

# model.eval()
def inference(img_set, save = True, metric = False, df = None, data = ""):
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
        plotting.plot_img(nib.Nifti1Image(input_tensor.cpu().numpy().squeeze(), subject.img.affine), display_mode='tiled')
        if metric:
            if data == "train" or data == "val":
                gt_tensor = subject['seg'][tio.DATA].unsqueeze(axis=0)
            else:
                gt_tensor = torch.nn.functional.one_hot(subject['seg'][tio.DATA].long(), num_classes=num_classes).permute(0, 4, 1, 2, 3)
            print(f"gt tensor shape : {gt_tensor.shape}")
            outputs_one_hot = torch.nn.functional.one_hot(output_tensor.long(), num_classes=num_classes)
            print(f"outputs_one_hot shape : {outputs_one_hot.shape}")
            outputs_one_hot = outputs_one_hot.permute(0, 4, 1, 2, 3)
            print(f"outputs_one_hot shape permuted: {outputs_one_hot.shape}")
            # if data == "test":
            #     outputs_one_hot = outputs_one_hot[:,5,:,:,:].unsqueeze(axis=0)
            get_dice(outputs_one_hot.to(model.device), gt_tensor.to(model.device))
            get_hd(outputs_one_hot.to(model.device), gt_tensor.to(model.device))

            print(f"subjects : {subjects_list}")
            dice = get_dice.aggregate().cpu().numpy()
            print(f"DICE : {dice}")
            get_dice.reset()
            hd = get_hd.aggregate().cpu().numpy()
            print(f"HD : {hd}")
            get_hd.reset()
            
            if data == "test":
                df = df.append(dict(zip(df.columns,[subject.subject] + [dice[0]] + [hd[0]])), ignore_index=True)
            else:
                df = df.append(dict(zip(df.columns,[subject.subject] + list(dice.squeeze()) + list(hd.squeeze()))), ignore_index=True)
            print(df)

        subject.add_image(pred, "prediction")
        new_subject = subject.apply_inverse_transform()

       
        if save :
            out_file = "./out-predictions/"+id_run+"/"+data+"/"+subject.subject
            if not os.path.exists(out_file):
                os.makedirs(out_file)
            pred = nib.Nifti1Image(new_subject.prediction.data.numpy().squeeze(), new_subject.prediction.affine)
            gt = nib.Nifti1Image(new_subject.seg.data.numpy(), new_subject.seg.affine)
            img = nib.Nifti1Image(new_subject.img.data.numpy().squeeze(), new_subject.img.affine)
            nib.save(pred,f"{out_file}/{subject.subject}_pred.nii.gz")
            nib.save(img,f"{out_file}/{subject.subject}_img.nii.gz")
            nib.save(gt,f"{out_file}/{subject.subject}_seg.nii.gz")
    if metric:
        return df

df_labels = pd.DataFrame(columns=["Subjects"] + ["DICE " + label for label in labels_names[1:]] + ["HD " + label for label in labels_names[1:]])

# print("# Validation")
# df_val = inference(val_subjects_dataset, metric = True, df = df_labels, data="val")
# df_val.to_csv("./out-predictions/"+id_run+"/scores_val.csv", index=False)

# print("# Training")
# df_train = inference(train_subjects_dataset, metric = True, df = df_labels, data="train")
# df_train.to_csv("./out-predictions/"+id_run+"/scores_train.csv", index=False)

print("# Test")
df_labels_test = pd.DataFrame(columns=["Subjects", "DICE Stroke-Lesion", "HD Stroke-Lesion"])
df_test = inference(test_subjects_dataset, metric = True, df = df_labels_test, data="test")
df_test.to_csv("./out-predictions/"+id_run+"/scores_test.csv", index=False)

# print("# Test Healthy")
# df_test_healthy = inference(test_healthy_subjects_dataset, metric = True, df = df_labels, data="test-healthy")
# df_test_healthy.to_csv("./out-predictions/"+id_run+"/scores_test-healthy.csv", index=False)

#'''