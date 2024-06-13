from pathlib import Path
import monai
import torchio as tio
import torch
import nibabel as nib
import numpy as np 
import os

from transforms import preprocess_label_map, preprocess
from get_subjects import get_subjects



def remove_labels(img_path, labels_path):
    img_dir=Path(img_path)
    labels_dir=Path(labels_path)

    transform_labels = preprocess_label_map()

    label_paths = sorted(labels_dir.glob('**/*_seg.nii.gz'))

    out_dir = "./data/Dataset_synthCandiStroke_r/Training"

    for path in label_paths : 
        mask_volume = nib.load(path)
        mask = torch.tensor(np.asarray(mask_volume.get_fdata())).type(torch.int16).unsqueeze(dim=0)
        subject = tio.Subject(
            tissues=tio.LabelMap(tensor=mask),
            subject = str(path).split("/")[-1]
        )
        transformed = transform_labels(subject)

        final_mask = nib.Nifti1Image(np.asarray(transformed["tissues"][tio.DATA])[0,:,:,:], mask_volume.affine)
        nib.save(final_mask, os.path.join(out_dir, subject.subject))


remove_labels("/home/emma/Projets/synthetic-model-brain-segmentation/data/synthetic_images_database/bdd_1/out_synthetic_images_v1", "/home/emma/Projets/synthetic-model-brain-segmentation/data/synthetic_images_database/bdd_1/out_synthetic_images_v1")