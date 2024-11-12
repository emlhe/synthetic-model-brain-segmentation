import sys 
from pathlib import Path 
import nibabel as nib 
import numpy as np 
import monai 
import monai.metrics
import torch
import pandas as pd 
import json 
## Get the name of the lesion and json file w/ infos on labels or the id of the label 
## get the paths of the predictions and the paths of the GT 
## associate prediction / GT

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

dir_pred = Path(sys.argv[1])
dir_gt = Path(sys.argv[2])
paths_preds = sorted(dir_pred.glob("**/*.nii.gz"))
paths_gt = sorted(dir_gt.glob("**/*mask.nii.gz"))
dateset_infos = Path(sys.argv[3])
with open(dateset_infos) as f:
    data_info = json.load(f)
    labels = data_info["labels"]
structures = ["left thalamus", "left caudate","left putamen","left pallidum","right thalamus", "right caudate","right putamen","right pallidum", "stroke"]
print(labels)         

scores_df = pd.DataFrame(columns = ["Subjects"] + ["DICE"] + ["HD 95"])
for path_pred in paths_preds:
    subject_pred = str(path_pred).split("_")[-1][:3]
    session_pred = str(path_pred).split("-")[-1][:2]
    print(f"prediction from subject {subject_pred}, session {session_pred}")
    pred_volume = nib.load(path_pred)
    pred_img = pred_volume.get_fdata()
    new_img = np.zeros(pred_img.shape)
    for structure in structures:
        pred_img_struct = np.where(pred_img == int(labels[structure]), pred_img, 0 )
        new_img = new_img + pred_img_struct
    pred_img = consecutive(new_img)
    
    pred_tensor_struct = torch.nn.functional.one_hot(torch.from_numpy(pred_img_struct).long(), num_classes = 2).unsqueeze(axis=0).permute(0, 4, 1, 2, 3)
    for path_gt in paths_gt:
        subject_gt = str(path_gt).split("sub-")[1][:3]
        session_gt = str(path_gt).split("ses-")[1][:2]
        
        if subject_pred == subject_gt and session_pred == session_gt:
            print(f"gt from subject {subject_gt}, session {session_gt}")
            gt_volume = nib.load(path_gt)
            gt_img = gt_volume.get_fdata()
            gt_tensor = torch.nn.functional.one_hot(torch.from_numpy(gt_img).long(), num_classes = 2).unsqueeze(axis=0).permute(0, 4, 1, 2, 3)
            print(pred_tensor_struct.shape)
            print(gt_tensor.shape)

            dsc = monai.metrics.DiceMetric(include_background=False)(pred_tensor_struct, gt_tensor)[0][0]
            # print(f"DSC for prediction sub-{subject_pred}_ses-{session_pred} with gt sub-{subject_gt}_ses-{session_gt} : {score}")
            hd95 = monai.metrics.HausdorffDistanceMetric(include_background=False, percentile=95)(pred_tensor_struct, gt_tensor)[0][0]
            # print(f"HD95 for prediction sub-{subject_pred}_ses-{session_pred} with gt sub-{subject_gt}_ses-{session_gt} : {score}")

            new_data = [[f"sub-{subject_gt}_ses-{session_gt}"] + [dsc.item()] + [hd95.item()]]
            scores_df = pd.concat([scores_df, pd.DataFrame(data=new_data, columns = scores_df.columns)], ignore_index=True)
            print(scores_df)
            break
print(scores_df)
scores_df.to_csv("scores.csv", index=False)
#'''