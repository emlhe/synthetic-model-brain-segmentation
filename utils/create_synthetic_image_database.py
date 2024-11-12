import numpy as np
import random
from pathlib import Path
import nipype
from nipype.interfaces import fsl
import nibabel as nib
import torch 
import torchio as tio 
import os
import copy
import torchio as tio
import json

def get_transform_from_json(json_file):
    with open(json_file) as f:
        transfo_st = json.load(f)
    train_transfo = parse_transform(transfo_st['train_transforms'],'train_transforms')
    motion_transfo = parse_transform(transfo_st['motion_transforms'],'motion_transforms')
    tioremap = tio.RemapLabels({29:0,30:0,41:2,42:3,43:4,44:5,46:7,47:8,49:10,50:11,51:12,52:13,53:17,54:18,58:26,60:28,61:0,62:0,72:0,77:2,85:0})
    return tio.Compose([tioremap, train_transfo, motion_transfo])

def parse_transform(t, transfo_name):
    if isinstance(t, list):
        transfo_list = [parse_transform(tt, transfo_name) for tt in t]
        if transfo_name == 'train_transforms':
            return tio.Compose(transfo_list)
        elif transfo_name == 'motion_transforms':
            return tio.OneOf(transfo_list, p=0.5)
    
    attributes = t.get('attributes') or {}

    t_class = getattr(tio.transforms, t['name'])
    return t_class(**attributes)
    
def get_subjects_list(list_img, list_seg, sessions_dict):

    assert len(list_img) == len(list_seg)

    i=0
    while i < len(list_img):
        path=list_img[i]
        sub = str(path).split("sub-")[-1][:3]
        ses = str(path).split("ses-")[-1][:2]

        # print(f"{i} : sub {sub}, ses {ses} :")

        if not sub in sessions_dict.keys():
            # print("\tSubject not in list")
            list_img.pop(i)
            list_seg.pop(i)
        else:
            if not sessions_dict.get(sub) == ses:
                # print("\tNot the right session")
                list_img.pop(i)
                list_seg.pop(i)
            else:
                # print("\tSubject in list, right session")
                print(f"sub {sub}, ses {ses}")
                i+=1
    return list_img, list_seg

def generate_synth_img(segmentation_paths, out, n, name_db):

    if not os.path.exists(out):
        os.makedirs(out)
    out_synth_img_paths=[]
    out_labels_paths=[]
    subject_list=[]
    for path in segmentation_paths:
        file_id = f'sub-{str(path).split(f"sub-")[-1][:3]}'
        subject = tio.Subject(
                seg=tio.LabelMap(path),
                name=file_id
            )
        subject_list.append(subject)

    tio_transfo = get_transform_from_json("./utils/transforms.json")   

    tioDS = tio.SubjectsDataset(subject_list, transform=tio_transfo)

    for i in range(35,35+n): 
        for sub in tioDS:
            out_s = os.path.join(out, sub.name)
            if not os.path.exists(out_s):
                os.makedirs(out_s)
            out_synth_img_path = os.path.join(out_s, f'{sub.name}_synth-{i}.nii.gz')     
            out_label_path = os.path.join(out_s, f'{sub.name}_label-{i}.nii.gz')     
            out_synth_img_paths_sub=[]
            out_labels_paths_sub=[]
            sub.synth.save(out_synth_img_path)
            sub.seg.save(out_label_path)

            print(f"Synthetic image saved in : {out_synth_img_path}")
            # print(f"With transfo : {sub.history}")

            out_synth_img_paths_sub.append(out_synth_img_path)
            out_labels_paths_sub.append(out_label_path)
        
        out_synth_img_paths_sub.append(out_synth_img_paths_sub)
        out_labels_paths.append(out_labels_paths_sub)

    return out_synth_img_paths, out_labels_paths

def generate_synth_img_stroke(lesion_masks_paths, brain_parc_path, out, n, name_db_1, name_db_2):

    if not os.path.exists(out):
        os.makedirs(out)

    brain_parc_volume = nib.load(brain_parc_path)
    whole_brain_seg = np.asarray(brain_parc_volume.get_fdata())

    synth_img_paths_sub=[]
    label_paths_sub = []
    for i in range(n) : 
        random_index = random.randint(0, len(lesion_masks_paths)-1)
        lesion_mask_path = lesion_masks_paths[random_index]
       
        print(f"Removing index {random_index}")
        
        lesion_subject = str(lesion_mask_path).split(f"sub-")[-1][:3]
        lesion_session = str(lesion_mask_path).split("ses-")[-1][:2]
        brain_parc_subject = str(brain_parc_path).split(f"sub-")[-1][:3]
        
        file_id = f'sub{name_db_2}-{brain_parc_subject}_sub{name_db_1}-{lesion_subject}_ses-{lesion_session}_lesion-overlay'
        out_s = os.path.join(out, f"sub{name_db_2}-{brain_parc_subject}")
        if not os.path.exists(out_s):
            os.makedirs(out_s)
        
        mask_file = f'{file_id}_seg-{i}.nii.gz'
        remmapped_mask_file = f'{file_id}_seg-remmapped-{i}.nii.gz'
        synth_img_file = f'{file_id}_synth-{i}.nii.gz'
        remmapped_mask_path = os.path.join(out_s, remmapped_mask_file)
        mask_path = os.path.join(out_s, mask_file)
        synth_img_path = os.path.join(out_s, synth_img_file)

        lesion_mask_volume = nib.load(lesion_mask_path)
        lesion_mask = np.asarray(lesion_mask_volume.get_fdata()).astype(bool)
        
        whole_brain_and_lesion_seg = np.where(lesion_mask==1, 64, whole_brain_seg)
        whole_brain_and_lesion_seg = torch.tensor(whole_brain_and_lesion_seg).type(torch.int16).unsqueeze(dim=0)

        subject = tio.Subject(
            seg=tio.LabelMap(tensor=whole_brain_and_lesion_seg)
        )

        remapping_left_right = {29:0,30:0,41:2,42:3,43:4,44:5,46:7,47:8,49:10,50:11,51:12,52:13,53:17,54:18,58:26,60:28,61:0,62:0,72:0,77:2,85:0}
        preproc_label_map_transform_left_right = tio.Compose([
            tio.RemapLabels(remapping_left_right),
            tio.SequentialLabels()
        ])
        transformed_im_left_right = preproc_label_map_transform_left_right(subject['seg'])
        subject.add_image(tio.Image(type = tio.LABEL, tensor = transformed_im_left_right.data), "remapped_tissues-left_right")
        
        rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99))
        simulation_transform = tio.RandomLabelsToImage(label_key="remapped_tissues-left_right", image_key='synthetic_mri', ignore_background=True)#, discretize=True)
        blurring_transform = tio.RandomBlur(std=0.3)
        transform = tio.Compose([rescale_transform, simulation_transform, blurring_transform])
        transformed = transform(subject)

        final_img = nib.Nifti1Image(np.asarray(transformed["synthetic_mri"][tio.DATA])[0,:,:,:], brain_parc_volume.affine)  
        transformed_mask = nib.Nifti1Image(np.asarray(subject["remapped_tissues-left_right"][tio.DATA])[0,:,:,:], brain_parc_volume.affine)
        final_mask = nib.Nifti1Image(np.asarray(subject["seg"][tio.DATA])[0,:,:,:], brain_parc_volume.affine)  
        nib.save(final_img, synth_img_path)
        nib.save(transformed_mask, remmapped_mask_path)
        nib.save(final_mask, mask_path)
        print(f"Synthetic image saved in : {synth_img_path}, mask : {mask_path}")

        synth_img_paths_sub.append(synth_img_path)
        label_paths_sub.append(mask_path)

    return synth_img_paths_sub, label_paths_sub

def resample_to_one(path, out_dir):
    if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    image = tio.ScalarImage(path)
    transform = tio.Resample(1)
    output = transform(image)
    path = str(path)
    out_path = os.path.join(out_dir, f"sub-{path.split('sub-')[-1].split('.nii.gz')[0]}_resampled.nii.gz")
    output.save(out_path)
    print(f"Resampled image saved in : {out_path}")
    return out_path

def normalize_mni(path_img, path_seg, path_ref, out, name_db, dict = None):
    if not os.path.exists(out):
            os.makedirs(out)

    subject_id = f"sub-{str(path_img).split(f'sub-')[-1][:3].split('.nii.gz')[0]}"

    # We register a subject to mni and use the transfo mat to register the labels to mni
    print(f"Registering {subject_id} to mni")
    flt_in_file = path_img
    flt_ref_file = path_ref
    flt_in_seg_file = path_seg
    out_path = os.path.join(out,f'{subject_id}')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    flt_out_file = os.path.join(out_path,f'{subject_id}_T1w-norm.nii.gz') # t1 normalized
    flt_out_seg_file = os.path.join(out_path,f'{subject_id}_mask-norm.nii.gz')  # mask registered
    flt_mat_file = os.path.join(out_path,f'{subject_id}_to_mni.mat')
    flt_mat_mask_file = os.path.join(out_path,f'{subject_id}_mask_to_mni.mat')

    # Recalage img1 vers mni
    flt = fsl.FLIRT()
    flt.inputs.in_file = flt_in_file
    flt.inputs.reference = flt_ref_file
    flt.inputs.out_matrix_file = flt_mat_file
    flt.inputs.out_file = flt_out_file
    print(flt.cmdline)
    flt.run() 

    # Recalage mask1 vers img2 à partir de la matrice de transfo
    flt_mask = fsl.FLIRT()
    flt_mask.inputs.in_file = flt_in_seg_file
    flt_mask.inputs.reference = flt_ref_file
    flt_mask.inputs.in_matrix_file = flt_mat_file
    flt_mask.inputs.interp = "nearestneighbour"
    flt_mask.inputs.apply_xfm = True
    flt_mask.inputs.out_matrix_file = flt_mat_mask_file
    flt_mask.inputs.out_file = flt_out_seg_file
    print(flt_mask.cmdline)
    flt_mask.run() 

    return flt_out_file, flt_out_seg_file, flt_mat_file

def invert_transform(img_paths, seg_paths, mat, mni_path, out, name_db, dict=None):
    if not os.path.exists(out):
            os.makedirs(out)

    inverted_img_paths=[]
    inverted_masks_paths=[]
    inverted_mat_paths=[]
    for path_img, path_seg in zip(img_paths, seg_paths):
        print(path_img)
        id_sub = str(path_img).split(f"/")[-1].split(f"_synth-")[0]
        i = str(path_img).split(f"_synth-")[1].split(".nii.gz")[0]
        print(out)
        print(id_sub)
        
        out_s = os.path.join(out, id_sub)
        print(out_s)
        if not os.path.exists(out_s):
            os.makedirs(out_s)

        print(f"Reverting transformation of {id_sub}")
        flt_in_file = path_img
        flt_ref_file = mni_path
        flt_in_seg_file = path_seg
        out_path = os.path.join(out_s,id_sub)
        flt_out_file = out_path + f'_synth-inverse-{i}.nii.gz' # t1 normalized
        flt_out_seg_file = out_path + f"_mask-inverse-{i}.nii.gz"  # mask registered
        flt_out_mat_file = out_path + f"_{i}.mat"
        flt_invt_mat_file = out_path + f"mni2subspace-{i}.mat"
        

        # convert_xfm -omat refvol2invol.mat -inverse invol2refvol.mat
        invt = fsl.ConvertXFM()
        invt.inputs.in_file = mat
        invt.inputs.invert_xfm = True
        invt.inputs.out_file = flt_invt_mat_file
        print(invt.cmdline)
        invt.run()

        flt = fsl.FLIRT()
        flt.inputs.in_file = flt_in_file
        flt.inputs.reference = flt_ref_file
        flt.inputs.in_matrix_file = flt_invt_mat_file
        flt.inputs.out_matrix_file = flt_out_mat_file
        flt.inputs.apply_xfm = True
        flt.inputs.out_file = flt_out_file
        print(flt.cmdline)
        flt.run() 

        # Recalage mask1 vers img2 à partir de la matrice de transfo
        flt_mask = fsl.FLIRT()
        flt_mask.inputs.in_file = flt_in_seg_file
        flt_mask.inputs.reference = flt_ref_file
        flt_mask.inputs.in_matrix_file = flt_invt_mat_file
        flt_mask.inputs.interp = "nearestneighbour"
        flt_mask.inputs.out_matrix_file = flt_out_mat_file
        flt_mask.inputs.apply_xfm = True
        flt_mask.inputs.out_file = flt_out_seg_file
        print(flt_mask.cmdline)
        flt_mask.run() 

        inverted_img_paths.append(flt_out_file)
        inverted_masks_paths.append(flt_out_seg_file)
        inverted_mat_paths.append(flt_invt_mat_file)


# GET ALL IMAGES + MASKS PATH FROM CAP CANDI AND DBB
path_cap = Path("/home/emma/data/IRM-CAP/derivatives/Brain_extraction/FS-SynthStrip/CAP/")
path_cap_normalized = Path("/home/emma/data/IRM-CAP/derivatives/normalization_mni")
path_candi = Path("/home/emma/data/DATABASES/CANDI/derivatives/FREESURFER-synthseg/")
path_dbb = Path("/home/emma/data/DATABASES/DBB/derivatives/DBB-subjects")
paths_img_cap = sorted(path_cap.glob("**/*_synthstripped.nii.gz"))
paths_img_normalized_cap = sorted(path_cap_normalized.glob("**/*_T1w-norm.nii.gz"))
paths_img_candi = sorted(path_candi.glob("**/*_resampled-t1.nii.gz"))
paths_img_dbb = sorted(path_dbb.glob("**/*_synthstripped.nii.gz"))
paths_seg_cap = sorted(path_cap.glob("**/*_mask.nii.gz"))
paths_seg_normalized_cap = sorted(path_cap_normalized.glob("**/*_mask-norm.nii.gz"))
paths_seg_candi = sorted(path_candi.glob("**/*_synthseg-pred.nii.gz"))
paths_seg_dbb = sorted(path_dbb.glob("**/*_parc.nii.gz"))


##### Generate synthetic images from healthy subjects 
# nb_example_per_subject = int(55/ 27)+1
# print(f"{len(paths_seg_candi)} for 1000 imgs : {nb_example_per_subject} per sub")
# generate_synth_img(paths_seg_candi, "/home/emma/data/DATABASES/CANDI/derivatives/synthCANDI/synthhealthy/", nb_example_per_subject, "candi")


nb_example_per_subject = int(500/ len(paths_img_candi))+1
print(f"{nb_example_per_subject} images per subject")
db1 = "cap"
db2 = "candi"

#### On récupère les chemins de toutes les images de CAP en fct du dictionnaire des sessions
sessions_dict = {"001":"02", "002":"01", "003":"01", "005":"01", "006":"02", "008":"01", "009":"02", "010":"01","012":"01", "013":"01", "014":"01", "015":"02", "030":"01","031":"02","032": "01", "036":"01","037": "01", "039": "01", "040":"01","041":"01","043": "01", "044":"01", "063": "01", "066":"01","068":"01","069":"01", "070":"01", "071":"01", "072": "01", "073": "01" }
paths_img_normalized_cap, paths_seg_normalized_cap = get_subjects_list(paths_img_normalized_cap, paths_seg_normalized_cap, sessions_dict)

#### On normalise images + labels sur MNI
path_ref_mni = "/home/emma/data/Atlases-Templates-TPM/MNI/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii"

for image, seg in zip(paths_img_candi, paths_seg_candi):
    out_normalized = "/home/emma/data/DATABASES/CANDI/derivatives/normalization_mni"
    path_t1_normalised_candi, path_parc_normalised_candi, path_mat_candi = normalize_mni(image, seg, path_ref_mni, out_normalized, "candi")

    outpath_synth_mni_candi = "/home/emma/data/DATABASES/CANDI/derivatives/synthCANDI/mni"
    imgs_synth_candi, labels_candi = generate_synth_img_stroke(paths_seg_normalized_cap, path_parc_normalised_candi, outpath_synth_mni_candi, nb_example_per_subject, "cap", "candi")

    outpath_synth_candi = "/home/emma/data/DATABASES/CANDI/derivatives/synthCANDI/subject-space"
    invert_transform(imgs_synth_candi, labels_candi, path_mat_candi, image, outpath_synth_candi, "candi")

#'''