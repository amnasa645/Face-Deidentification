#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_file
import cv2
import os
from skimage import morphology
import math
import numpy as np
from tqdm import tqdm
import glob as glob
import SimpleITK as sitk


def get_full_scan(folder_path):
    files_List = glob.glob(folder_path + '/**/*.dcm', recursive=True)
    itkimage = sitk.ReadImage(files_List[0])
    rows = int(itkimage.GetMetaData('0028|0010'))
    cols = int(itkimage.GetMetaData('0028|0011'))
    mn = 1000
    mx = 0
    for file in tqdm(files_List):
        itkimage = sitk.ReadImage(file)
        mn = np.min([mn, int(itkimage.GetMetaData('0020|0013'))])
        mx = np.max([mx, int(itkimage.GetMetaData('0020|0013'))])
    full_scan = np.ndarray(shape=(mx - mn + 1, rows, cols), dtype=float, order='F')
    new_list = np.ndarray(shape=(mx - mn + 1), dtype=object)
    for file in tqdm(files_List):
        img, n = dcm_image(file)
        n = int(n)
        full_scan[n - mn, :, :] = img[0, :, :]
        new_list[n-mn] = file
    return full_scan,new_list

def dcm_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    ins = float(itkimage.GetMetaData('0020|0013'))
    return numpyImage, ins


def CBCT_blurred(image, prev = 0,s=0):
    gray = np.zeros((image.shape[0], image.shape[1]) , dtype=int)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if image[i,j] > -300 and image[i,j] < 500:
                gray[i,j] = 1
    #plt.imshow(gray , cmap = 'gray')
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    gray = cv2.morphologyEx(np.float32(gray), cv2.MORPH_CLOSE, ker1)
    gray = cv2.morphologyEx(np.float32(gray), cv2.MORPH_CLOSE, ker2)
    #plt.imshow(gray , cmap = 'gray')
    gray = gray>0
    gray = morphology.remove_small_objects(gray, min_size=200)
    #plt.imshow(gray , cmap = 'gray')
    gray = gray.astype(np.uint8)
    kernel = np.ones((30, 30), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = gray.astype(np.uint8)
    #plt.imshow(gray , cmap = 'gray')
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gray = cv2.drawContours(gray, contours,-1, 255, 10)
    #plt.imshow(gray , cmap = 'gray')
    gray[300:,:] = 0
    #plt.imshow(diff, cmap ='gray')
    seg = gray
    seg = np.where((seg==255),seg,0)
    z=0
    if not s==0:
        diff = np.subtract(seg,prev)
        diff = diff>1
        xx = int((diff.shape[1])/2)
        diff[:,0:(xx-100)] =0
        diff[:,(xx+100):] = 0
        diff = morphology.remove_small_objects(diff, min_size=500)
        if np.any(diff):
            for l in range(diff.shape[1]):
                for j in range(diff.shape[0]):
                    if diff[j,l]==True:
                        if np.any(prev[(j-10):(j+10),l]):
                            diff[j,l] = False
            seg = np.where((diff==True),0,seg)
            z=1
    blurred_img = cv2.GaussianBlur(image, (101, 101), 400)
    out = np.where((seg==255), blurred_img, image)
    out = out.astype(np.int16)
    if z==0:
        return out , seg
    else:
        return out, prev

def predictions(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files_list = sorted(glob.glob((input_folder+ "/**/*.dcm"),recursive = True))
    output_list= []
    classUID = []
    initialdata = pydicom.dcmread(files_list[0])
    mode = str(initialdata.Modality)
    if (mode=='CT') or (mode=='ct'):
        i = 0
        scan, names = get_full_scan(input_folder)
        for i in tqdm(range(scan.shape[0])):
            name = names[i]
            img = scan[i, :, :]
            if i == 0:
                out, prev_img = CBCT_blurred(img)
            else:
                out, prev_img = CBCT_blurred(img, prev_img, s=1)
    
            out = out.astype(np.int16)
            dcmData = pydicom.dcmread(name)
            dcmData.PixelData = out.tobytes()
            classUID.append(str(dcmData.SOPClassUID))
            _, tail = os.path.split(name)
            des_path = os.path.join(output_folder, tail)
            output_list.append(des_path)
            dcmData.save_as(des_path)
            i = i + 1
    mimeType = "application/dicom"
    recommendation_string = {"finding": "finding","conclusion":"conclusion","recommendation":"recommendation"} 
    return output_list, classUID, mimeType, recommendation_string





