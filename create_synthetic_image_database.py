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



def get_subjects_list(list_img, list_seg, sessions_dict):

    assert len(list_img) == len(list_seg)

    i=0
    while i < len(list_img):
        path=list_img[i]
        sub = str(path).split("sub-")[-1][:3]
        ses = str(path).split("ses-")[-1][:2]

        print(f"{i} : sub {sub}, ses {ses} :")

        if not sub in sessions_dict.keys():
            print("\tSubject not in list")
            list_img.pop(i)
            list_seg.pop(i)
        else:
            if not sessions_dict.get(sub) == ses:
                print("\tNot the right session")
                list_img.pop(i)
                list_seg.pop(i)
            else:
                print("\tSubject in list, right session")
                i+=1
    return list_img, list_seg

def register_images(list_img_2_register, list_seg_2_register, list_img_reference, list_seg_reference, out, name_db_1, name_db_2):            
    registered_masks_paths = []
    reference_masks_paths = []
    tmp = copy.deepcopy(list_img_reference)

    if not os.path.exists(out):
        os.makedirs(out)

    for path_img_2_register, path_seg_2_register in zip(list_img_2_register, list_seg_2_register):
        # ASSOCIATE A RANDOM IMAGE FROM CANDI TO THE CAP IMAGE AND DELETE IT FROM THE CANDI LIST 
        print(len(list_img_reference))
        if len(list_img_reference)>0:
            random_index = random.randint(0, len(list_img_reference)-1)
            print(random_index)  
        else:
            list_img_reference=copy.deepcopy(tmp)
        print(f"Removing index {random_index}")
        if len(list_img_reference)==1:
            path_img_reference = list_img_reference[0]
            path_seg_reference = list_seg_reference[0]
        else:
            path_img_reference = list_img_reference.pop(random_index)
            path_seg_reference = list_seg_reference.pop(random_index)
        sub_2_register = str(path_img_2_register).split("sub-")[-1][:3]
        if 'ses' in str(path_img_2_register):
            ses_2_register = str(path_img_2_register).split("ses-")[-1][:2]
            sub_2_register = sub_2_register + "_ses-" + ses_2_register
        
        sub_ref = str(path_img_reference).split("sub-")[-1][:3]
        if 'ses' in str(path_img_reference):
            ses_ref = str(path_img_reference).split("ses-")[-1][:2]
            sub_ref = sub_ref + "_ses-" + ses_ref

        

        # We register a cap subject to a random candi subject and use the transfo mat to register the lesion mask to the candi subject
        print(f"Registering {name_db_1} subject {sub_2_register} to {name_db_2} subject {sub_ref}")
        flt_in_file = path_img_2_register
        flt_ref_file = path_img_reference
        flt_in_seg_file = path_seg_2_register
        out_path = os.path.join(out,f'sub{name_db_1}-{sub_2_register}_to_sub{name_db_2}-{sub_ref}')
        flt_out_file = os.path.join(out_path, f'sub{name_db_1}-{sub_2_register}_to_sub{name_db_2}-{sub_ref}_registered_T1w.nii.gz') # t1 registered
        flt_out_seg_file = os.path.join(out_path, f'sub{name_db_1}-{sub_2_register}_to_sub{name_db_2}-{sub_ref}_registered-mask.nii.gz')  # mask registered
        flt_mat_file = os.path.join(out_path + f"sub{name_db_2}_to_{sub_ref}.mat")
        reference_mask = path_seg_reference

        # Recalage img1 vers img2
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
        flt_mask.inputs.apply_xfm = True
        flt.inputs.out_matrix_file = flt_mat_file
        flt_mask.inputs.out_file = flt_out_seg_file
        print(flt_mask.cmdline)
        flt_mask.run() 


        # The registered mask paths are the lesion masks registered to candi and the associated mask paths are the whole brain seg from candi 
        registered_masks_paths.append(flt_out_seg_file)
        reference_masks_paths.append(reference_mask)

    return registered_masks_paths, reference_masks_paths

# CREATE SYNTHETIC IMAGES FROM THE LESION MASK + WHOLE BRAIN SEG
def generate_synth_img_stroke(lesion_masks_paths, brain_parc_paths, out, n, name_db_1, name_db_2):
    tmp = copy.deepcopy(lesion_masks_paths)
    if not os.path.exists(out):
        os.makedirs(out)

    synth_img_paths=[]
    label_paths=[]
    for brain_parc_path in brain_parc_paths:
        print(len(lesion_masks_paths))

        brain_parc_volume = nib.load(brain_parc_path)
        whole_brain_seg = np.asarray(brain_parc_volume.get_fdata())

        synth_img_paths_sub=[]
        label_paths_sub = []
        for i in range(n) : 
            if len(lesion_masks_paths)>1:
                random_index = random.randint(0, len(lesion_masks_paths)-1)
                lesion_mask_path = lesion_masks_paths.pop(random_index) 
            else:
                random_index=1
                lesion_mask_path = lesion_masks_paths[-1]
                lesion_masks_paths=copy.deepcopy(tmp)

            print(f"Removing index {random_index}")
            
            lesion_subject = str(lesion_mask_path).split(f"sub-")[-1][:3]
            lesion_session = str(lesion_mask_path).split("ses-")[-1][:2]
            if name_db_2 == "dbb":
                brain_parc_subject = str(brain_parc_path).split(f"sub-")[-1][:4]
            else:
                brain_parc_subject = str(brain_parc_path).split(f"sub-")[-1][:3]
            
            file_id = f'sub{name_db_1}-{lesion_subject}_ses-{lesion_session}_to_sub{name_db_2}-{brain_parc_subject}'
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
   
            # remapping_ngc = {2:3,3:2,4:1,5:1,7:6,8:6,10:4,11:4,12:4,13:4,14:1,15:1,16:5,17:2,18:2,24:1,26:2,28:2,29:0,30:0,41:3,42:2,43:1,44:1,46:6,47:6,49:4,50:4,51:4,52:4,53:2,54:2,58:2,60:2,61:0,62:0,64:5,72:0,77:3,85:0}
            # preproc_label_map_transform_ngc = tio.Compose([
            #     tio.RemapLabels(remapping_ngc),
            #     tio.SequentialLabels()
            # ])
            remapping_left_right = {29:0,30:0,41:2,42:3,43:4,44:5,46:7,47:8,49:10,50:11,51:12,52:13,53:17,54:18,58:26,60:28,61:0,62:0,72:0,77:2,85:0}
            preproc_label_map_transform_left_right = tio.Compose([
                tio.RemapLabels(remapping_left_right),
                tio.SequentialLabels()
            ])
            transformed_im_left_right = preproc_label_map_transform_left_right(subject['seg'])
            subject.add_image(tio.Image(type = tio.LABEL, tensor = transformed_im_left_right.data), "remapped_tissues-left_right")
            transformed_mask = nib.Nifti1Image(np.asarray(subject["remapped_tissues-left_right"][tio.DATA])[0,:,:,:], brain_parc_volume.affine)
            
            resample = tio.Resample(1)
            rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99))
            simulation_transform = tio.RandomLabelsToImage(label_key="remapped_tissues-left_right", image_key='synthetic_mri', ignore_background=True)#, discretize=True)
            blurring_transform = tio.RandomBlur(std=0.3)
            transform = tio.Compose([resample, rescale_transform, simulation_transform, blurring_transform])
            transformed = transform(subject)

            final_img = nib.Nifti1Image(np.asarray(transformed["synthetic_mri"][tio.DATA])[0,:,:,:], brain_parc_volume.affine)    
            nib.save(final_img, synth_img_path)
            print(f"Synthetic image saved in : {synth_img_path}, mask : {mask_path}")
            nib.save(transformed_mask, remmapped_mask_path)
            final_mask = nib.Nifti1Image(np.asarray(subject["seg"][tio.DATA])[0,:,:,:], brain_parc_volume.affine)
            nib.save(final_mask, mask_path)

            synth_img_paths_sub.append(synth_img_path)
            label_paths_sub.append(mask_path)

        synth_img_paths.append(synth_img_paths_sub)
        label_paths.append(label_paths_sub)

    return synth_img_paths, label_paths

def generate_synth_img(segmentation_paths, out, n, name_db):

    if not os.path.exists(out):
        os.makedirs(out)

    synth_img_paths=[]
    label_paths=[]
    for path in segmentation_paths:
        if name_db == "dbb":
            subject = str(path).split(f"sub-")[-1][:4]
        else:
            subject = str(path).split(f"sub-")[-1][:3]
        out_s = os.path.join(out, subject)
        if not os.path.exists(out_s):
            os.makedirs(out_s)
        

        segmentation_volume = nib.load(path)

        whole_brain_seg = np.asarray(segmentation_volume.get_fdata())
        whole_brain_seg = torch.tensor(whole_brain_seg).type(torch.int16).unsqueeze(dim=0)
        save_mask_file = f'sub-{subject}_seg.nii.gz'

        subject = tio.Subject(
            seg=tio.LabelMap(tensor=whole_brain_seg)
        )

        remapping_ngc = {2:3,3:2,4:1,5:1,7:3,8:2,10:4,11:4,12:4,13:4,14:1,15:1,16:3,17:2,18:2,24:1,26:2,28:2,29:0,30:0,41:3,42:2,43:1,44:1,46:3,47:2,49:4,50:4,51:4,52:4,53:2,54:2,58:2,60:2,61:0,62:0,64:5,72:0,77:3,85:0}
        remapping_left_right = {29:0,30:0,41:2,42:3,43:4,44:5,46:7,47:8,49:10,50:11,51:12,52:13,53:17,54:18,58:26,60:28,61:0,62:0,72:0,77:2,85:0}
        preproc_label_map_transform_ngc = tio.Compose([
            tio.RemapLabels(remapping_ngc),
            tio.SequentialLabels()
        ])
        preproc_label_map_transform_left_right = tio.Compose([
            tio.RemapLabels(remapping_left_right),
            tio.SequentialLabels()
        ])
        # transformed_im_ngc = preproc_label_map_transform_ngc(subject['seg'])
        # subject.add_image(tio.Image(type = tio.LABEL, tensor = transformed_im_ngc.data), "remapped_tissues-ngc")
        transformed_im_left_right = preproc_label_map_transform_left_right(subject['seg'])
        subject.add_image(tio.Image(type = tio.LABEL, tensor = transformed_im_left_right.data), "remapped_tissues-left_right")
        transformed_mask = nib.Nifti1Image(np.asarray(subject["remapped_tissues-left_right"][tio.DATA])[0,:,:,:], segmentation_volume.affine)
            
        save_mask_file_path = os.path.join(out_s, f"{save_mask_file.split('_seg.nii.gz')[0]}_sef-remmapped.nii.gz")
        nib.save(transformed_mask, save_mask_file_path)
        # tissues_to_transform = ["seg", "remapped_tissues-ngc", "remapped_tissues-left_right"]        

        synth_img_paths_sub=[]
        for i in range(n) : 
            save_generated_image_file_path = f"{save_mask_file.split('_seg.nii.gz')[0]}_synth-{i}.nii.gz"
            resample = tio.Resample(1)
            rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99), )
                
            simulation_transform = tio.RandomLabelsToImage(label_key="remapped_tissues-left_right", image_key='synthetic_mri', ignore_background=True)#, discretize=True)
            blurring_transform = tio.RandomBlur(std=0.3)
            transform = tio.Compose([resample, rescale_transform, simulation_transform, blurring_transform])

            transformed = transform(subject)

            final_img = nib.Nifti1Image(np.asarray(transformed["synthetic_mri"][tio.DATA])[0,:,:,:], segmentation_volume.affine)      
            nib.save(final_img, os.path.join(out_s, save_generated_image_file_path))
            print(f"Synthetic image saved in : {os.path.join(out_s, save_generated_image_file_path)}")

            synth_img_paths_sub.append(save_generated_image_file_path)

        synth_img_paths_sub.append(synth_img_paths_sub)
        label_paths.append(save_mask_file_path)

    return synth_img_paths, label_paths

def normalize_mni(paths_img, paths_seg, path_ref, out, name_db, dict = None):
    if not os.path.exists(out):
            os.makedirs(out)

    normalized_masks_paths=[]
    transfo_mat_paths=[]
    for path_img, path_seg in zip(paths_img, paths_seg):
        if name_db == "dbb":
            subject = str(path_img).split(f"sub-")[-1][:4]
        else:
            subject = str(path_img).split(f"sub-")[-1][:3]

        # We register a subject to mni and use the transfo mat to register the labels to mni
        print(f"Registering {subject} to mni")
        flt_in_file = path_img
        flt_ref_file = path_ref
        flt_in_seg_file = path_seg
        out_path = os.path.join(out,f'sub-{subject}')
        if dict is not None:
            out_path = os.path.join(out_path, f'sub-{subject}_ses-{dict[subject]}')
        flt_out_file = os.path.join(out_path,f'sub-{subject}_T1w-norm.nii.gz') # t1 normalized
        flt_out_seg_file = os.path.join(out_path,f'sub-{subject}_mask-norm.nii.gz')  # mask registered
        flt_mat_file = os.path.join(out_path,f'sub-{subject}_to_mni.mat')

        if not os.path.exists(out_path):
            os.makedirs(out_path)

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
        flt_mask.inputs.apply_xfm = True
        flt.inputs.out_matrix_file = flt_mat_file
        flt_mask.inputs.out_file = flt_out_seg_file
        print(flt_mask.cmdline)
        flt_mask.run() 


        # The registered mask paths are the lesion masks registered to candi and the associated mask paths are the whole brain seg from candi 
        normalized_masks_paths.append(flt_out_seg_file)
        transfo_mat_paths.append(flt_mat_file)

    return normalized_masks_paths, transfo_mat_paths

def invert_transform(img_paths, seg_paths, mat, mni_path, out, name_db, dict=None):
    if not os.path.exists(out):
            os.makedirs(out)

    inverted_img_paths=[]
    inverted_masks_paths=[]
    inverted_mat_paths=[]
    for path_img, path_seg in zip(img_paths, seg_paths):
        print(path_img)
        if name_db == "dbb":
            id_sub = str(path_img).split(f"_synth-")[0]
            i = str(path_img).split(f"_synth-")[1].split(".nii.gz")[0]
        else:
            id_sub = str(path_img).split(f"_synth-")[0]
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
        out_path = os.path.join(out,id_sub)
        flt_out_file = out_path + '_synth-inverse.nii.gz' # t1 normalized
        flt_out_seg_file = out_path + "_mask-inverse.nii.gz"  # mask registered
        flt_invt_mat_file = out_path + "mni2subspace.mat"

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
        flt.inputs.apply_xfm = True
        flt.inputs.out_file = flt_out_file
        print(flt.cmdline)
        flt.run() 

        # Recalage mask1 vers img2 à partir de la matrice de transfo
        flt_mask = fsl.FLIRT()
        flt_mask.inputs.in_file = flt_in_seg_file
        flt_mask.inputs.reference = flt_ref_file
        flt_mask.inputs.in_matrix_file = flt_invt_mat_file
        flt_mask.inputs.apply_xfm = True
        flt_mask.inputs.out_file = flt_out_seg_file
        print(flt_mask.cmdline)
        flt_mask.run() 

        inverted_img_paths.append(flt_out_file)
        inverted_masks_paths.append(flt_out_seg_file)
        inverted_mat_paths.append(flt_invt_mat_file)


# GET ALL IMAGES + MASKS PATH FROM CAP CANDI AND DBB
path_cap = Path("/home/emma/data/IRM-CAP/derivatives/Brain_extraction/FS-SynthStrip/CAP/")
path_candi = Path("/home/emma/data/DATABASES/CANDI/derivatives/FREESURFER-synthseg/")
path_dbb = Path("/home/emma/data/DATABASES/DBB/derivatives/DBB-subjects")
paths_img_cap = sorted(path_cap.glob("**/*_synthstripped.nii.gz"))[:1]
paths_img_candi = sorted(path_candi.glob("**/*_resampled-t1.nii.gz"))[:1]
paths_img_dbb = sorted(path_dbb.glob("**/*_synthstripped.nii.gz"))[:1]
paths_seg_cap = sorted(path_cap.glob("**/*_mask.nii.gz"))[:1]
paths_seg_candi = sorted(path_candi.glob("**/*_synthseg-pred.nii.gz"))[:1]
paths_seg_dbb = sorted(path_dbb.glob("**/*_parc.nii.gz"))[:1]

# Generate synthetic images from healthy subjects 
# n_healthy = 200
# generate_synth_img(paths_seg_candi, n_healthy, "candi")


nb_example_per_subject = nb_example_per_subject = int(1000/ len(paths_img_cap))+1
print(f"{nb_example_per_subject} images per subject")
db1 = "cap"
db2 = "candi"

# outpath_reg = f'/home/emma/data/DATABASES/Synthetic_database/DDB-stroke/out_registered_{db1}_lesions_masks_on_{db2}'
# outpath_synth = os.path.join(outpath_reg.split(f"registered")[0], "out_synthetic_images")

# 1 image par session
sessions_dict = {"001":"02", "002":"01", "003":"01", "005":"01", "006":"02", "008":"01", "009":"02", "010":"01","012":"01", "013":"01", "014":"01", "015":"02", "030":"01","031":"02","032": "01", "036":"01","037": "01", "039": "01", "040":"01","041":"01","043": "01", "044":"01", "063": "01", "066":"01","068":"01","069":"01", "070":"01", "071":"01", "072": "01", "073": "01" }

# On récupère les chemins de toutes les images de CAP en fct du dictionnaire des sessions
paths_img_cap, paths_seg_cap = get_subjects_list(paths_img_cap, paths_seg_cap, sessions_dict)

# On normalise images + labels sur MNI
path_ref_mni = "/home/emma/data/Atlases-Templates-TPM/MNI/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii"

cap_mask_normalised_paths, cap_mat_paths = normalize_mni(paths_img_cap, paths_seg_cap, path_ref_mni, "/home/emma/data/IRM-CAP/derivatives/normalization_mni", "cap", sessions_dict)
dbb_mask_normalised_paths, dbb_mat_paths = normalize_mni(paths_img_dbb, paths_seg_dbb, path_ref_mni, "/home/emma/data/DATABASES/DBB/derivatives/normalization_mni", "dbb")
candi_mask_normalised_paths, candi_mat_paths = normalize_mni(paths_img_candi, paths_seg_candi, path_ref_mni, "/home/emma/data/DATABASES/CANDI/derivatives/normalization_mni", "candi")

# On recale les masques
outpath_synth_dbb = "/home/emma/data/DATABASES/DBB/derivatives/synthDBB/mni"
nb_example_per_subject = 12
dbb_synth_img, dbb_labels = generate_synth_img_stroke(cap_mask_normalised_paths, dbb_mask_normalised_paths, outpath_synth_dbb, nb_example_per_subject, "cap", "dbb")
outpath_candi_dbb = "/home/emma/data/DATABASES/CANDI/derivatives/synthCANDI/mni"
nb_example_per_subject = 12
candi_synth_img, candi_labels = generate_synth_img_stroke(cap_mask_normalised_paths, candi_mask_normalised_paths, outpath_candi_dbb, nb_example_per_subject, "cap", "candi")

# On inverse la transformation pour avoir les images synthétiques dans l'espace du sujet
for i in range(len(dbb_synth_img)):
    invert_transform(dbb_synth_img[i], dbb_labels[i], dbb_mat_paths[i], path_ref_mni, "/home/emma/data/DATABASES/DBB/derivatives/subject-space", "dbb")
for i in range(len(candi_synth_img)):    
    invert_transform(candi_synth_img[i], candi_labels[i], candi_mat_paths[i], path_ref_mni, "/home/emma/data/DATABASES/CANDI/derivatives/synthCANDI/subject-space", "candi")

# distr = 'tissus' # distr = 'tissus' vs 'structure' selon la distribution que l'on veut pour l'image générée

# path_cap = Path(outpath_reg)
# cap_registered_masks_paths = sorted(path_cap.glob("**/*_registered-lesion-mask.nii.gz"))
# candi_initial_masks_paths = []
# for e in cap_registered_masks_paths:
#     subcandi = str(e).split("subcandi-")[1][:3]
#     print("subcandi " + str(subcandi))
#     for f in paths_seg_candi:
#         if subcandi == str(f).split("sub-")[1][:3]:
#             print("subcandi 2 " + str(str(f).split("sub-")[1][:3]))
#             candi_initial_masks_paths.append(f)

# generate_synth_img_stroke(cap_registered_masks_paths, dbb_initial_masks_paths, outpath_synth, nb_example_per_subject, db1, db2)


# out_path = "/home/emma/data/DATABASES/Synthetic_database/CANDI"
# nb_example_per_subject = int(200/ len(paths_seg_candi))+1
# print(f"{nb_example_per_subject} images per subject")
# generate_synth_img(paths_seg_candi, out_path, nb_example_per_subject)