#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Apr 28 2020

@author: mike germuska
"""
code_ver="1.0.0"

import helper_functions as hf
import ML_functions as mf
import nibabel as nib
import numpy as np

from nipype.interfaces import fsl
import CBF0_func as cb

# get command line input parameters
args=hf.parse_cmdln()
# process command line parameters, put in dict and save to csv p_file
print('importing data')
images_dict,d_phys,d_analysis,d_scan_par = hf.process_cmdln(args,code_ver)

d_analysis['M0_cut']=6000  # value depends on image bit-depth and receiver gain ... need to automate this

# calculate CBF0 without smoothing
CBF0_orig=cb.calc_cbf0(images_dict,d_phys,d_scan_par,d_analysis)
print('saving CBF0 data')

empty_header=nib.Nifti1Header()

CBF_orig_img=nib.Nifti1Image(CBF0_orig, images_dict['echo1_img'].affine, empty_header)
nib.save(CBF_orig_img, d_analysis['outpath'] + 'CBF0_orig.nii.gz')

# bet, fast and register CBF0 to structural and make GM mask
anat_brain_fn=d_analysis['struct_fn'][0:d_analysis['struct_fn'].rindex('/')+1]+'anat_brain.nii.gz'

betr = fsl.BET()
print('fsl BET')
betr.inputs.in_file = d_analysis['struct_fn']
betr.inputs.out_file = anat_brain_fn
betr.cmdline
res = betr.run()

fastr = fsl.FAST()
print('fsl FAST')
fastr.inputs.in_files = anat_brain_fn
fastr.inputs.out_basename = 'fast_'
fastr.cmdline
res = fastr.run()

# flirt try registering CBF0 to fast_pveseg (with mutalinfo cost function as contrast is inverted)   
flirtr = fsl.FLIRT(cost_func='mutualinfo')
print('fsl FLIRT')   
flirtr.inputs.in_file = d_analysis['outpath'] + 'CBF0_orig.nii.gz'
flirtr.inputs.reference = 'fast__pveseg.nii.gz'
flirtr.inputs.output_type = "NIFTI_GZ"
flirtr.cmdline
res = flirtr.run()    

import os

cmd = 'fslmaths fast__pve_1.nii.gz -thr 0.5 -bin hr_gm_mask.nii.gz'
os.system(cmd)

# now invert the transform, apply to fast__pve_1, then threshold and binaries to create gm mask
import os
cmd = 'convert_xfm -omat brain_2_func.mat -inverse  CBF0_orig_flirt.mat'
os.system(cmd)

cmd = 'flirt -in fast__pve_1.nii.gz -ref ' + d_analysis['outpath'] + 'CBF0_orig.nii.gz' + ' -applyxfm -init brain_2_func.mat -o pve_1_native.nii.gz'
os.system(cmd)

cmd = 'flirt -in fast__pve_0.nii.gz -ref ' + d_analysis['outpath'] + 'CBF0_orig.nii.gz' + ' -applyxfm -init brain_2_func.mat -o pve_0_native.nii.gz'
os.system(cmd)

cmd = 'flirt -in fast__pve_2.nii.gz -ref ' + d_analysis['outpath'] + 'CBF0_orig.nii.gz' + ' -applyxfm -init brain_2_func.mat -o pve_2_native.nii.gz'
os.system(cmd)

cmd = 'fslmaths pve_1_native.nii.gz -thr 0.5 -bin gm_mask.nii.gz'
os.system(cmd)

try:
    mask_img=nib.load('gm_mask.nii.gz')
    mask_data=mask_img.get_fdata()
except Exception as error:
        print(error)
        raise SystemExit(0)

mask_data[:,:,0:5]=0 # set cerebellum slices to zero in mask (removes low PLD data)

# save mask
empty_header=nib.Nifti1Header()

mask_img=nib.Nifti1Image(mask_data, images_dict['echo1_img'].affine, empty_header)
nib.save(mask_img, 'gm_mask.nii.gz')

# move files to appropriate directory 
cmd = 'cp gm_mask.nii.gz ' + d_analysis['outpath']
os.system(cmd)
cmd = 'cp hr_gm_mask.nii.gz ' + d_analysis['outpath']
os.system(cmd)
cmd = 'cp fast__pve_1.nii.gz ' + d_analysis['outpath']
os.system(cmd)
cmd = 'cp fast__pve_0.nii.gz ' + d_analysis['outpath']
os.system(cmd)
cmd = 'cp fast__pve_2.nii.gz ' + d_analysis['outpath']
os.system(cmd)


# calculate CMRO2, CBF0, and OEF0 using ANN from CO2 ASL and BOLD data

CMRO20,CBF0,OEF0,M,Dc = mf.calc_cmro2(images_dict,d_phys,d_scan_par,d_analysis)

# create a 'brain' mask

try:
    pve_0_img=nib.load('pve_0_native.nii.gz')
    pve_0_data=pve_0_img.get_fdata()
except Exception as error:
        print(error)
        raise SystemExit(0)

try:
    pve_1_img=nib.load('pve_1_native.nii.gz')
    pve_1_data=pve_1_img.get_fdata()
except Exception as error:
        print(error)
        raise SystemExit(0)

try:
    pve_2_img=nib.load('pve_2_native.nii.gz')
    pve_2_data=pve_2_img.get_fdata()
except Exception as error:
        print(error)
        raise SystemExit(0)


brain_mask=pve_1_data+pve_2_data # GM and WM only
brain_mask[brain_mask>0]=1
brain_mask[brain_mask<1]=0

print('saving data')
empty_header=nib.Nifti1Header()


OEF_img=nib.Nifti1Image(OEF0*brain_mask, images_dict['echo1_img'].affine, empty_header)
nib.save(OEF_img, d_analysis['outpath'] + 'OEF0.nii.gz')

CBF_img=nib.Nifti1Image(CBF0*brain_mask, images_dict['echo1_img'].affine, empty_header)
nib.save(CBF_img, d_analysis['outpath'] + 'CBF0.nii.gz')

CMRO2_img=nib.Nifti1Image(CMRO20*brain_mask, images_dict['echo1_img'].affine, empty_header)
nib.save(CMRO2_img, d_analysis['outpath'] + 'CMRO20.nii.gz')

M_img=nib.Nifti1Image(M*brain_mask, images_dict['echo1_img'].affine, empty_header)
nib.save(M_img, d_analysis['outpath'] + 'M.nii.gz')

#CBVv_img=nib.Nifti1Image(CBVv*brain_mask, images_dict['echo1_img'].affine, empty_header)
#nib.save(CBVv_img, d_analysis['outpath'] + 'CBVv.nii.gz')

Dc_img=nib.Nifti1Image(Dc*brain_mask, images_dict['echo1_img'].affine, empty_header)
nib.save(Dc_img, d_analysis['outpath'] + 'Dc.nii.gz')


# now tranform maps to structural space

cmd = 'flirt -in ' + d_analysis['outpath'] + 'OEF0.nii.gz -ref ' + d_analysis['struct_fn'] + ' -applyxfm -init CBF0_orig_flirt.mat -o OEF0_hr.nii.gz'
os.system(cmd)
cmd = 'flirt -in ' + d_analysis['outpath'] + 'CBF0.nii.gz -ref ' + d_analysis['struct_fn'] + ' -applyxfm -init CBF0_orig_flirt.mat -o CBF0_hr.nii.gz'
os.system(cmd)
cmd = 'flirt -in ' + d_analysis['outpath'] + 'CMRO20.nii.gz -ref ' + d_analysis['struct_fn'] + ' -applyxfm -init CBF0_orig_flirt.mat -o CMRO20_hr.nii.gz'
os.system(cmd)


cmd = 'cp OEF0_hr.nii.gz ' + d_analysis['outpath']
os.system(cmd)
cmd = 'cp CBF0_hr.nii.gz ' + d_analysis['outpath']
os.system(cmd)
cmd = 'cp CMRO20_hr.nii.gz ' + d_analysis['outpath']
os.system(cmd)


# delete unused files N.B. very dangerous method of deletion be careful where script is run!!!
cmd = 'rm -R *.nii.gz' 
os.system(cmd)
cmd = 'rm -R *.mat'
os.system(cmd)

cmd = 'fslmaths ' + d_analysis['outpath'] + 'gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'OEF0.nii.gz ' + d_analysis['outpath'] + 'OEF0_masked.nii.gz'
os.system(cmd)
cmd = 'fslmaths ' + d_analysis['outpath'] + 'gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'CBF0.nii.gz ' + d_analysis['outpath'] + 'CBF0_masked.nii.gz'
os.system(cmd)
cmd = 'fslmaths ' + d_analysis['outpath'] + 'gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'CMRO20.nii.gz ' + d_analysis['outpath'] + 'CMRO20_masked.nii.gz'
os.system(cmd)

cmd = 'fslmaths ' + d_analysis['outpath'] + 'gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'M.nii.gz ' + d_analysis['outpath'] + 'M_masked.nii.gz'
os.system(cmd)

#cmd = 'fslmaths ' + d_analysis['outpath'] + 'gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'CBVv.nii.gz ' + d_analysis['outpath'] + 'CBVv_masked.nii.gz'
#os.system(cmd)

cmd = 'fslmaths ' + d_analysis['outpath'] + 'gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'Dc.nii.gz ' + d_analysis['outpath'] + 'Dc_masked.nii.gz'
os.system(cmd)

cmd = 'fslmaths ' + d_analysis['outpath'] + 'gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'CBF0_orig.nii.gz ' + d_analysis['outpath'] + 'CBF0_orig_masked.nii.gz'
os.system(cmd)

# calculate GM averages

cmd = 'fslstats ' + d_analysis['outpath'] + 'OEF0_masked.nii.gz -M'
os.system(cmd)
cmd = 'fslstats ' + d_analysis['outpath'] + 'CBF0_masked.nii.gz -M'
os.system(cmd)
cmd = 'fslstats ' + d_analysis['outpath'] + 'CMRO20_masked.nii.gz -M'
os.system(cmd)
cmd = 'fslstats ' + d_analysis['outpath'] + 'M_masked.nii.gz -M'
os.system(cmd)
#cmd = 'fslstats ' + d_analysis['outpath'] + 'CBVv_masked.nii.gz -M'
#os.system(cmd)
cmd = 'fslstats ' + d_analysis['outpath'] + 'Dc_masked.nii.gz -M'
os.system(cmd)

# cmd = 'fslstats ' + d_analysis['outpath'] + 'CBF0_orig_masked.nii.gz -M'
# os.system(cmd)


cmd = 'fslmaths ' + d_analysis['outpath'] + 'hr_gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'OEF0_hr.nii.gz ' + d_analysis['outpath'] + 'OEF0_hr_masked.nii.gz'
os.system(cmd)
cmd = 'fslmaths ' + d_analysis['outpath'] + 'hr_gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'CBF0_hr.nii.gz ' + d_analysis['outpath'] + 'CBF0_hr_masked.nii.gz'
os.system(cmd)
cmd = 'fslmaths ' + d_analysis['outpath'] + 'hr_gm_mask.nii.gz -mul ' + d_analysis['outpath'] + 'CMRO20_hr.nii.gz ' + d_analysis['outpath'] + 'CMRO20_hr_masked.nii.gz'
os.system(cmd)




