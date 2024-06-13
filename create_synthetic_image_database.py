import numpy as np
import random
from pathlib import Path
from nipype.interfaces import fsl
from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from os.path import join as opj
from nipype.interfaces.io import SelectFiles, DataSink
import nibabel as nib
import torch 
import torchio as tio 
import os
import matplotlib.pyplot as plt 
import sys
sys.path.append("/home/emma/Projets/Synthetic_images_segmentation/")
from transforms import preprocess_label_map

data_format = "1"
def get_subjects_list():
    cap_img = ["001", "005", "006", "008", "009", "010","012", "030","031","036","040","041","044","066","068","069"]
    sessions_dict = {"001":"02", "005":"01", "006":"02", "008":"01", "009":"02", "010":"01","012":"01", "030":"01","031":"02","036":"01","040":"01","041":"01","044":"01","066":"01","068":"01","069":"01"}

    # GET ALL IMAGES + MASKS PATH FROM CAP AND CANDI
    path_cap = Path("/home/emma/data/IRM-CAP/derivatives/Brain_extraction/FS-SynthStrip/CAP/")
    path_candi = Path("/home/emma/data/CANDI/derivatives/Brain_extraction/")
    list_img_cap = sorted(path_cap.glob("**/*_synthstripped.nii.gz"))
    list_img_candi = sorted(path_candi.glob("**/*_synthstripped.nii.gz"))
    list_seg_cap = sorted(path_cap.glob("**/*_mask.nii.gz"))
    list_seg_candi = sorted(path_candi.glob("**/*_seg.nii.gz"))

    assert len(list_img_cap) == len(list_seg_cap)
    # REMOVE IMAGES NOT IN LIST FROM CAP 
    i=0
    while i < len(list_img_cap):
        path=list_img_cap[i]
        sub = str(path).split("sub-")[-1][:3]
        ses = str(path).split("ses-")[-1][:2]

        print(f"{i} : sub {sub}, ses {ses} :")

        if not sub in cap_img:
            print("\tSubject not in list")
            list_img_cap.pop(i)
            list_seg_cap.pop(i)
        else:
            if not sessions_dict.get(sub) == ses:
                print("\tNot the right session")
                list_img_cap.pop(i)
                list_seg_cap.pop(i)
            else:
                print("\tSubject in list, right session")
                i+=1
    return list_img_cap, list_seg_cap, list_img_candi, list_seg_candi

# Apply transformations so that CAP and CANDI img + seg are in the same space 
def register_images(list_img_cap,list_seg_cap, list_img_candi, list_seg_candi, out, context):            
    registered_masks_paths = []
    associated_masks_paths = []
    for path_img_cap, path_seg_cap in zip(list_img_cap, list_seg_cap):
        # ASSOCIATE A RANDOM IMAGE FROM CANDI TO THE CAP IMAGE AND DELETE IT FROM THE CANDI LIST 
        random_candi_index = random.randint(0, len(list_img_candi)-1)
        print(f"Removing index {random_candi_index}")
        path_img_candi = list_img_candi.pop(random_candi_index)
        path_seg_candi = list_seg_candi.pop(random_candi_index)
        sub_cap = str(path_img_cap).split("sub-")[-1][:3]
        ses_cap = str(path_img_cap).split("ses-")[-1][:2]
        sub_cap_mask = str(path_seg_cap).split("sub-")[-1][:3]
        ses_cap_mask = str(path_seg_cap).split("ses-")[-1][:2]

        assert sub_cap == sub_cap_mask
        assert ses_cap == ses_cap_mask

        sub_candi = str(path_img_candi).split("sub-")[-1][:3]

        # First database : we register a cap subject to a rando candi subject and use the transfo mat to register the lesion to the candi subject
        if context == "1" :
            print(f"Registering cap subject {sub_cap} session {ses_cap} and mask from sub {sub_cap_mask}, session {ses_cap_mask}, to candi subject {sub_candi}")
            flt_in_file = path_img_cap
            flt_ref_file = path_img_candi
            flt_in_seg_file = path_seg_cap
            out='./synthetic_images/bdd_'+context+'/out_registered_images/subcap-'+sub_cap+'_ses-'+ses_cap+"_to_subcandi-"+sub_candi
            flt_out_file = out + '_registered_T1w.nii.gz'
            flt_out_seg_file = out + "_registered-lesion-mask.nii.gz"
            flt_mat_file = out + ".mat"
            associated_mask = path_seg_candi

        # 2nd and 3rd databases : we register a random candi subject to a cap subject and use the transfo mat to register the segmentation map to the cap subject 
        elif context == "2" or context == "3" :
            print(f"Registering candi subject {sub_candi} and associated segmentation map to cap subject {sub_cap} session {ses_cap}")
            flt_in_file = path_img_candi
            flt_ref_file = path_img_cap
            flt_in_seg_file = path_seg_candi
            out='./synthetic_images/bdd_'+context+'/out_registered_images/subcandi-'+sub_candi+'_to_subcap-'+sub_cap+'_ses-'+ses_cap
            flt_out_file = out + '_registered_T1w.nii.gz'
            flt_out_seg_file = out + "_registered-whole-brain-mask.nii.gz"
            flt_mat_file = out + ".mat"
            associated_mask = path_seg_cap

        flt_in_file="/home/emma/data/CANDI/derivatives/Brain_extraction/sub-023/sub-023_synthstripped.nii.gz"
        flt_ref_file="/home/emma/data/IRM-CAP/derivatives/Brain_extraction/FS-SynthStrip/CAP/sub-001/ses-02/anat/sub-001_ses-02_synthstripped.nii.gz"
        flt_mat_file="synthetic_images/bdd_3/out_registered_images/subcandi-023_to_subcap-001_ses-02.mat"
        if context == "3":
            fnt = fsl.FNIRT()
            res=fnt.run(in_file=flt_in_file, ref_file=flt_ref_file, affine_file=flt_mat_file, fieldcoeff_file = './synthetic_images/bdd_'+context+'/out_registered_images/subcandi-'+sub_candi+'_to_subcap-'+sub_cap+'_ses-'+ses_cap+'wrapped.nii.gz') 

            aw = fsl.ApplyWarp()
            aw.inputs.in_file = flt_in_seg_file
            aw.inputs.ref_file = flt_ref_file
            aw.inputs.field_file = './synthetic_images/bdd_'+context+'/out_registered_images/subcandi-'+sub_candi+'_to_subcap-'+sub_cap+'_ses-'+ses_cap+'wrapped.nii.gz'
            aw.out_file = './synthetic_images/bdd_'+context+'/out_registered_images/subcandi-'+sub_candi+'_to_subcap-'+sub_cap+'_ses-'+ses_cap+"_fnirt_seg.nii.gz"


        # in database 1 : the registered mask paths are the lesion masks registered to candi and the associated mask paths are the whole brain seg from candi 
        # in databases 2 and 3 : the registered mask paths are the whole brain segmentation registered to cap and the associated mask paths are the lesions masks from cap
        registered_masks_paths.append(flt_out_seg_file)
        associated_masks_paths.append(associated_mask)

    return registered_masks_paths, associated_masks_paths

# CREATE SYNTHETIC IMAGES FROM THE LESION MASK + WHOLE BRAIN SEG
def generate_synth_img(context, registered_masks_paths=[], associated_masks_paths=[], outpath = "./", n=0):

    for r_mask_path, a_mask_path in zip(registered_masks_paths,associated_masks_paths):
        candi_subject = str(r_mask_path).split("subcandi-")[-1][:3]
        cap_subject = str(r_mask_path).split("subcap-")[-1][:3]
        cap_session = str(r_mask_path).split("ses-")[-1][:2]

        r_mask_volume = nib.load(r_mask_path)
        a_mask_volume = nib.load(a_mask_path)
        if len(np.unique(np.asarray(r_mask_volume))) > 2 : # Check which is the whole brain seg and which is the lesion seg (if more than 1 label)
            whole_brain_seg_volume = r_mask_volume
            whole_brain_seg = np.asarray(whole_brain_seg_volume.get_fdata())
            lesion_mask = np.asarray(a_mask_volume.get_fdata()).astype(bool)
            save_mask_file = 'subcandi-'+candi_subject+'_to_subcap-'+cap_subject+'_ses-'+cap_session+'_seg'+n+'.nii.gz'
            save_generated_image_file = save_mask_file.split('_seg.nii.gz')[0]+'_synth'+n+'.nii.gz'

        else:
            whole_brain_seg_volume = a_mask_volume
            whole_brain_seg = np.asarray(whole_brain_seg_volume.get_fdata())
            lesion_mask = np.asarray(r_mask_volume.get_fdata()).astype(bool)
            save_mask_file = 'subcap-'+cap_subject+'_ses-'+cap_session+'_to_subcandi-'+candi_subject+'_seg'+n+'.nii.gz'
            save_generated_image_file = save_mask_file.split('_seg.nii.gz')[0]+'_synth'+n+'.nii.gz'

        whole_brain_and_lesion_seg = np.where(lesion_mask==1, 64, whole_brain_seg)
        whole_brain_and_lesion_seg = torch.tensor(whole_brain_and_lesion_seg).type(torch.int16).unsqueeze(dim=0)

        subject = tio.Subject(
            tissues=tio.LabelMap(tensor=whole_brain_and_lesion_seg)
        )

        resample = tio.Resample(1)
        rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99), )
        simulation_transform = tio.RandomLabelsToImage(label_key='tissues', image_key='generated_mri', ignore_background=True)#, discretize=True)
        blurring_transform = tio.RandomBlur(std=0.3)
        transform = tio.Compose([resample, rescale_transform, simulation_transform, blurring_transform])

        transformed = transform(subject)

        final_img = nib.Nifti1Image(np.asarray(transformed["generated_mri"][tio.DATA])[0,:,:,:], whole_brain_seg_volume.affine)
        final_mask = nib.Nifti1Image(np.asarray(subject["tissues"][tio.DATA])[0,:,:,:], whole_brain_seg_volume.affine)

        
        nib.save(final_mask, os.path.join(outpath, save_mask_file))
        nib.save(final_img, os.path.join(outpath, save_generated_image_file))
        print(f"Synthetic image saved in : {os.path.join(outpath, save_generated_image_file)}")

# CREATE SYNTHETIC IMAGES FROM THE LESION MASK + WHOLE BRAIN SEG
def generate_synth_img_from_list(context, registered_masks_paths=[]):
    out_dir = "./synthetic_images/bdd_"+context+"/out_synthetic_images"
    for r_mask_path in registered_masks_paths:
        candi_subject = str(r_mask_path).split("subcandi-")[-1][:3]
        cap_subject = str(r_mask_path).split("subcap-")[-1][:3]
        cap_session = str(r_mask_path).split("ses-")[-1][:2]

        candi_path = Path("/home/emma/data/CANDI/derivatives/Brain_extraction/")
        candi_seg_path = sorted(candi_path.glob("**/*"+candi_subject+"_seg.nii.gz"))[0]
        print(candi_seg_path)
        r_mask_volume = nib.load(r_mask_path)
        a_mask_volume = nib.load(candi_seg_path)
        if len(np.unique(np.asarray(r_mask_volume))) < 2:
            whole_brain_seg_volume = a_mask_volume
            whole_brain_seg = np.asarray(whole_brain_seg_volume.get_fdata())
            lesion_mask = np.asarray(r_mask_volume.get_fdata()).astype(bool)
            save_mask_file = 'subcap-'+cap_subject+'_ses-'+cap_session+'_to_subcandi-'+candi_subject+'_seg.nii.gz'
            save_generated_image_file = save_mask_file.split('_seg.nii.gz')[0]+'_synthetic.nii.gz'
        else:
            whole_brain_seg_volume = r_mask_volume
            whole_brain_seg = np.asarray(whole_brain_seg_volume.get_fdata())
            lesion_mask = np.asarray(a_mask_volume.get_fdata()).astype(bool)
            save_mask_file = 'subcandi-'+candi_subject+'_to_subcap-'+cap_subject+'_ses-'+cap_session+'_seg.nii.gz'
            save_generated_image_file = save_mask_file.split('_seg.nii.gz')[0]+'_synthetic.nii.gz'
            
        whole_brain_and_lesion_seg = np.where(lesion_mask==1, 64, whole_brain_seg)
        whole_brain_and_lesion_seg = torch.tensor(whole_brain_and_lesion_seg).type(torch.int16).unsqueeze(dim=0)

        subject = tio.Subject(
            tissues=tio.LabelMap(tensor=whole_brain_and_lesion_seg)
        )

        # remapping = {2:3,3:2,4:1,5:1,7:3,8:2,10:2,11:2,12:2,13:2,14:1,15:1,16:3,17:2,18:2,24:1,26:2,28:2,29:0,30:0,41:3,42:2,43:1,44:1,46:3,47:2,49:2,50:2,51:2,52:2,53:2,54:2,58:2,60:2,61:0,62:0,64:4,72:0,77:3,85:0}
        # preproc_label_map_transform = tio.Compose([
        #     tio.RemapLabels(remapping),
        #     tio.SequentialLabels()
        # ])

        # transformed_im = preproc_label_map_transform(subject['tissues'])
        # print(np.unique(transformed_im.data.numpy()))
        # subject.add_image(tio.Image(type = tio.LABEL, tensor = transformed_im.data), "remapped_tissues")

        rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99), )
        simulation_transform = tio.RandomLabelsToImage(label_key='tissues', image_key='generated_mri', ignore_background=True)
        blurring_transform = tio.RandomBlur(std=0.1)
        transform = tio.Compose([rescale_transform, simulation_transform, blurring_transform])

        transformed = transform(subject)

        final_img = nib.Nifti1Image(np.asarray(transformed["generated_mri"][tio.DATA])[0,:,:,:], whole_brain_seg_volume.affine)
        final_mask = nib.Nifti1Image(np.asarray(subject["tissues"][tio.DATA])[0,:,:,:], whole_brain_seg_volume.affine)

        
        nib.save(final_mask, os.path.join(out_dir, save_mask_file))
        nib.save(final_img, os.path.join(out_dir, save_generated_image_file))
        print(f"Synthetic image saved in : {os.path.join(out_dir, save_generated_image_file)}")


img_cap, seg_cap, img_candi, seg_candi = get_subjects_list()

outpath = './data/synthetic_images/bdd_'+data_format+'/out_registered_images'
if not os.path.exists(outpath):
    os.makedirs(outpath)
registered_masks_paths, associated_masks_path = register_images(img_cap,seg_cap, img_candi, seg_candi, outpath, context = data_format)

# registered_masks_paths = ["/home/emma/Projets/MRI_processing_scripts/synthetic_images/bdd_2/out_registered_images/subcandi-013_to_subcap-001_ses-02_registered-whole-brain-mask.nii.gz"]
# associated_masks_path = ["/home/emma/data/IRM-CAP/derivatives/Brain_extraction/FS-SynthStrip/CAP/sub-001/ses-02/anat/sub-001_ses-02_mask.nii.gz"]
# registered_masks_paths = ["./synthetic_images/bdd_1/out_registered_images/subcap-001_ses-02_to_subcandi-028_registered-lesion-mask.nii.gz"]
# associated_masks_path = ["/home/emma/data/CANDI/derivatives/Brain_extraction/sub-028/sub-028_seg.nii.gz"]
# reg_im_path = Path("./generate_synthetic_images/bdd_1/out_registered_images/")
# registered_masks_paths = sorted(reg_im_path.glob("**/*_mask.nii.gz"))

outpath = os.join(outpath.split("out_registered_images")[0], "out_synthetic_images")
if not os.path.exists(outpath):
    os.makedirs(outpath)

nb_example_per_subject = 2

for n in range(nb_example_per_subject):
    generate_synth_img(data_format, registered_masks_paths, associated_masks_path, outpath, n)

