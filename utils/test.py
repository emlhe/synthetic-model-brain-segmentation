import os 
import nibabel as nib
import numpy as np 
import random
import torch
import torchio as tio 
import json 
from nipype.interfaces import fsl

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

def generate_synth_img_stroke(lesion_masks_paths, brain_parc_paths, out, n, name_db_1, name_db_2):

    if not os.path.exists(out):
        os.makedirs(out)

    for brain_parc_path in brain_parc_paths:
        brain_parc_volume = nib.load(brain_parc_path)
        brain_parc = np.asarray(brain_parc_volume.get_fdata())

        for i in range(n) : 
            random_index = random.randint(0, len(lesion_masks_paths)-1)
            lesion_mask_path = lesion_masks_paths[random_index]
            print(f"Removing index {random_index}")
            
            lesion_subject = str(lesion_mask_path).split(f"sub-")[-1][:3]
            lesion_session = str(lesion_mask_path).split("ses-")[-1][:2]
            brain_parc_subject = str(brain_parc_path).split(f"sub-")[-1][:3]         
            file_id = f'sub{name_db_1}-{brain_parc_subject}_sub{name_db_2}-{lesion_subject}_ses-{lesion_session}'

            lesion_mask = np.asarray(nib.load(lesion_mask_path).get_fdata()).astype(bool)
            lesion_and_brain = np.where(brain_parc!=0 , lesion_mask, 0)
            whole_brain_and_lesion_seg = np.where(lesion_and_brain==1, 64, brain_parc)
            whole_brain_and_lesion_seg = torch.tensor(whole_brain_and_lesion_seg).type(torch.int16).unsqueeze(dim=0)

            subject = tio.Subject(
                seg=tio.LabelMap(tensor=whole_brain_and_lesion_seg),
                name=file_id
            )
            out_s = os.path.join(out, f"sub-{brain_parc_subject}")
            if not os.path.exists(out_s):
                os.makedirs(out_s)
            out_synth_img_path = os.path.join(out_s, f'{subject.name}_synth-{i}.nii.gz')     
            out_label_path = os.path.join(out_s, f'{subject.name}_label-{i}.nii.gz')     

            tio_transfo = get_transform_from_json("./utils/transforms.json")
            transformed = tio_transfo(subject)

            transformed.synth.save(out_synth_img_path)
            transformed.seg.save(out_label_path)

            print(f"Synthetic image saved in : {out_synth_img_path}")


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

out_normalized = "./"
path_ref_mni = "/home/emma/data/Atlases-Templates-TPM/MNI/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii"
path_img_candi="/home/emma/data/DATABASES/CANDI/derivatives/FREESURFER-synthseg/sub-001/sub-001_synthstripped.nii.gz"
path_seg_candi="/home/emma/data/DATABASES/CANDI/derivatives/FREESURFER-synthseg/sub-001/sub-001_seg.nii.gz"
path_t1_normalised_candi, path_parc_normalised_candi, path_mat_candi = normalize_mni(path_img_candi, path_seg_candi, path_ref_mni, out_normalized, "candi")

outpath_synth_mni_candi = "./"
paths_seg_normalized_cap="/home/emma/data/IRM-CAP/derivatives/normalization_mni/sub-003/sub-003_ses-01/sub-003_mask-norm.nii.gz"
imgs_synth_candi, labels_candi = generate_synth_img_stroke(paths_seg_normalized_cap, path_parc_normalised_candi, outpath_synth_mni_candi, 3, "cap", "candi")

outpath_synth_candi = "./"
invert_transform(imgs_synth_candi, labels_candi, path_mat_candi, path_img_candi, outpath_synth_candi, "candi")