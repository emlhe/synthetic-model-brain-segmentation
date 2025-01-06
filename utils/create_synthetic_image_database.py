import numpy as np
import random
from pathlib import Path
from nipype.interfaces import fsl
import nibabel as nib
import torch 
import torchio as tio 
import os
import torchio as tio
import json
import argparse


def set_up():
    parser = argparse.ArgumentParser(description='descr')
    parser.add_argument('-lo', '--lesion_overlay', default=True, type=bool, help='If True, the lesion masks to be overlaid must be provided')
    parser.add_argument('-m', '--masks', type=str, help='Stroke lesion masks that will be overlaid on the whole brains parcellations, must end with "mask.nii.gz"')
    parser.add_argument('-st', '--stroke_t1', type=str, help='Associated brains with stroke lesions, must be synthstripped and end with "T1w.nii.gz". Needed only if the masks are not normalized')
    parser.add_argument('-l', '--labels', type=str, help='Base whole brain parcellations on which will be overlaid the stroke lesion, must end with "labels.nii.gz"')
    parser.add_argument('bt', '--base_t1', type=str, help='Associated base brain images on which will be overlaid the stroke lesion, must be synthstripped and end with "T1w.nii.gz". Needed only if not normalized')
    parser.add_argument('n', '--normalized', default=False, type=bool, help='If True, the masks and labels must be pre-normalized to the same template')
    parser.add_argument('ngen', '--n_generations', default=False, type=bool, help='Number of generations per image')
    parser.add_argument('-t', 'template', type=str, help='Template to register the images and masks. Only if normalized is False')
    parser.add_argument('-o', 'out', type=str, help='Directory to save the synthetic images and normalized images if necessary')
    args = parser.parse_args()

    return args


def main():
    args = set_up()

    paths_stroke_masks = sorted(args.masks.glob("**/*mask.nii.gz"))
    paths_labels = sorted(args.labels.glob("**/*labels.nii.gz"))
    
    remap_dict = {4:3,5:1,6:4,7:5,8:6,9:7,10:8,11:3,12:3,13:1,14:9,15:10,16:11,17:12,18:5,19:1,20:2,21:3,22:3,23:1,24:4,25:5,26:6,27:7,28:8,29:9,30:10,31:12,32:5,33:13,34:14} # remap candi
    remap_dict = {1:2,2:1,3:11,4:3,5:4,6:5,7:8,8:7,9:6,10:10,11:12,12:9,13:0,14:0,15:0,16:13,17:14} # remap dbb 
    
    if not args.normalized or not args.lo:
        paths_stroke_t1 = sorted(args.stroke_t1.glob("**/*T1w.nii.gz"))
        paths_base_t1 = sorted(args.base_t1.glob("**/*T1w.nii.gz"))

        if args.lo:
            for path_stroke_mask, path_stroke_t1 in zip(paths_stroke_masks, paths_stroke_t1):
                flt_out_stroke_t1_file, flt_out_stroke_seg_file, flt_stroke_mat_file = normalize_mni(path_stroke_mask, path_stroke_t1, args.template, args.out, "stroke", dict = None)
            
            for path_labels, path_base_t1 in zip(paths_labels, paths_base_t1):
                flt_out_base_t1_file, flt_out_base_seg_file, flt_base_mat_file = normalize_mni(path_labels, path_base_t1, args.template, args.out, "td", dict = None)
            generate_synth_img_stroke(flt_out_stroke_seg_file, flt_out_base_seg_file, args.out, args.n_generations, remap_dict, "stroke", "base")
        else:
            generate_synth_img(paths_labels, args.out, args.n_generations, "synth_base")

    else:
        generate_synth_img_stroke(paths_stroke_masks, paths_labels, args.out, args.n_generations, remap_dict, "stroke", "base")
        


def get_transform_from_json(json_file):
    with open(json_file) as f:
        transfo_st = json.load(f)
    train_transfo = parse_transform(transfo_st['train_transforms'],'train_transforms')
    motion_transfo = parse_transform(transfo_st['motion_transforms'],'motion_transforms')
    return tio.Compose([train_transfo, motion_transfo])

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

def make_consecutive(mask):
    i=0
    print(np.unique(mask))
    labels = np.unique(mask)
    for l in labels:
        mask = np.where(mask==l, i, mask)
        i+=1
    print(np.unique(mask))
    return mask

def get_subjects_list(list_img, list_seg, sessions_dict):
       
    i=0
    while i < len(list_img):
        path=list_img[i]
        sub = str(path).split("sub-")[-1][:3]
        ses = str(path).split("ses-")[-1][:2]

        # print(f"{i} : sub {sub}, ses {ses} :")
        print(f"{i} : sub {sub}, ses {ses} :")
        if not sub in sessions_dict.keys():
            # print("\tSubject not in list")            
            list_img.pop(i)
            
        else:
            if not sessions_dict.get(sub) == ses:
                # print("\tNot the right session")
                list_img.pop(i)
                # list_seg.pop(i)
            else:
                # print("\tSubject in list, right session")
                print(f"sub {sub}, ses {ses}")
                i+=1
    i=0    
    while i < len(list_seg):
        path=list_seg[i]
        sub = str(path).split("sub-")[-1][:3]
        ses = str(path).split("ses-")[-1][:2]

        if not sub in sessions_dict.keys():
            # print("\tSubject not in list")            
            list_seg.pop(i)
            
        else:
            if not sessions_dict.get(sub) == ses:
                # print("\tNot the right session")
                list_seg.pop(i)
                # list_seg.pop(i)
            else:
                # print("\tSubject in list, right session")
                print(f"sub {sub}, ses {ses}")
                i+=1
    
    assert len(list_img) == len(list_seg)
    
    return list_img, list_seg

def generate_synth_img(segmentation_paths, out, n, name_db):

    if not os.path.exists(out):
        os.makedirs(out)
    out_synth_img_paths=[]
    out_labels_paths=[]
    subject_list=[]
    for path in segmentation_paths:
        file_id = f'sub-{str(path).split(f"sub-")[-1].split(f"_")[0]}'
        seg_vol = torch.tensor(nib.load(path).get_fdata()).type(torch.int16).unsqueeze(dim=0)
        subject = tio.Subject(
                seg=tio.LabelMap(tensor=seg_vol),
                name=file_id
            )
        subject_list.append(subject)

    tio_transfo = get_transform_from_json("./utils/transforms.json")   

    tioDS = tio.SubjectsDataset(subject_list, transform=tio_transfo)

    for i in range(15,15+n): 
        for sub in tioDS:
            out_s = os.path.join(out, sub.name)
            if not os.path.exists(out_s):
                os.makedirs(out_s)

            if len(str(i))==1:
                ii=f"0{i}"
            else:
                ii=str(i)

            out_synth_img_path = os.path.join(out_s, f'{sub.name}_synth-{ii}.nii.gz')     
            out_label_path = os.path.join(out_s, f'{sub.name}_label-{ii}.nii.gz')     
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

def add_mask_to_parcellation(brain_parc, lesion_masks_paths):
    random_index = random.randint(0, len(lesion_masks_paths)-1)
    lesion_mask_path = lesion_masks_paths.pop(random_index)
    print(f"Removing index {random_index}")

    lesion_mask = np.asarray(nib.load(lesion_mask_path).get_fdata())
    # On enlève la lésion si background
    lesion_in_brain = np.where(brain_parc==0 , 0, lesion_mask)
    print(np.sum(lesion_in_brain))
    # On test si la lésion est entièrement dans ventricules
    lesion_in_brain = np.where(brain_parc==4, 0, lesion_in_brain) 

    print(np.sum(lesion_in_brain))
    if np.sum(lesion_in_brain) == 0:
        print("no lesion left")
        add_mask_to_parcellation(brain_parc, lesion_masks_paths)

    print(len(np.unique(brain_parc)))
    whole_brain_and_lesion_seg = np.where(lesion_in_brain==1, len(np.unique(brain_parc)), brain_parc)
    whole_brain_and_lesion_seg = np.where(lesion_in_brain==2, len(np.unique(brain_parc))+1, whole_brain_and_lesion_seg)
    whole_brain_and_lesion_seg = torch.tensor(whole_brain_and_lesion_seg).type(torch.int16).unsqueeze(dim=0)
    return whole_brain_and_lesion_seg,  lesion_mask_path, lesion_masks_paths

def generate_synth_img_stroke(lesion_masks_paths, brain_parc_path, out, n, remap_dict, name_db_1, name_db_2):

    if not os.path.exists(out):
        os.makedirs(out)

    # for brain_parc_path in brain_parc_paths:
    brain_parc_volume = nib.load(brain_parc_path)
    brain_parc = make_consecutive(np.asarray(brain_parc_volume.get_fdata()))
    imgs=[]
    labels=[]
    for i in range(n) : 
        print(len(lesion_masks_paths))
        whole_brain_and_lesion_seg, lesion_mask_path, lesion_masks_paths = add_mask_to_parcellation(brain_parc, lesion_masks_paths)
        lesion_subject = str(lesion_mask_path).split(f"/")[-1].split(f"_")[0]
        lesion_th = str(lesion_mask_path).split("th-")[-1][:3]
        brain_parc_subject = str(brain_parc_path).split(f"sub-")[-1].split(f"_")[0]
        file_id = f'sub{name_db_1}-{brain_parc_subject}-sub{name_db_2}-{lesion_subject}-th-{lesion_th}'

        subject = tio.Subject(
            seg=tio.LabelMap(tensor=whole_brain_and_lesion_seg),
            name=file_id
        )
        out_s = os.path.join(out, f"sub-{brain_parc_subject}")
        if not os.path.exists(out_s):
            os.makedirs(out_s)
        if len(str(i))==1:
            ii=f"0{i}"
        else:
            ii=str(i)
        # out_synth_img_path_no_transforms = os.path.join(out_s, f'{subject.name}-{ii}_synth_no_transforms.nii.gz')     
        out_synth_img_path = os.path.join(out_s, f'{subject.name}-{ii}_synth.nii.gz')     
        out_label_path = os.path.join(out_s, f'{subject.name}-{ii}_label.nii.gz')            
        
        # synth_transfo = tio.RandomLabelsToImage(label_key='seg', image_key= "synth", default_mean= [0, 1], default_std=[0.02, 0.1])
        # gen_syn = tio.Compose([tioremap_dbb, synth_transfo])

        # transformed = gen_syn(subject)
        # transformed.synth.save(out_synth_img_path_no_transforms)
        remap=tio.RemapLabels(remap_dict)
        remapped_subject = remap(subject)
        tio_transfo = get_transform_from_json("./utils/transforms.json")
        transformed = tio_transfo(remapped_subject)

        transformed.synth.save(out_synth_img_path)
        tioremap_lesion = tio.RemapLabels({14:13})
        transformed_remaped = tioremap_lesion(transformed)
        transformed_remaped.seg.save(out_label_path)

        print(f"Synthetic image saved in : {out_synth_img_path}")

        imgs.append(out_synth_img_path)
        labels.append(out_label_path)
    return imgs, labels

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

def normalize_mni(path_img, path_seg, path_ref, out):
    if not os.path.exists(out):
            os.makedirs(out)

    subject_id = f"{str(path_img).split(f'/')[-1].split('.nii.gz')[0]}"

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


#'''