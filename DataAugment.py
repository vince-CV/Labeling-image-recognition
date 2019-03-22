import numpy as np
import cv2
import random
import threading
import os
import time
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image



def fold(image):
    rows,cols,channel = image.shape
    
    new_row = int (rows/2)
    dst = image[0:new_row, 0:cols]
    return dst
    

def Rotation(image):
    rows,cols,channel = image.shape
    angle = np.random.uniform(low=-20.0, high=20.0)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(image, M, (cols,rows))
    return dst

def Translate(image):
    rows,cols,channel = image.shape
    x_ = cols*0.15
    y_ = rows*0.15
    scale = np.random.uniform(0.80, 1.20)
    x = np.random.uniform(-x_, x_)
    y = np.random.uniform(-y_, y_)
    M = np.float32([[scale, 0, x], [0, scale, y]])
    dst = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return dst

def Affine(image):
    img_info=image.shape
    image_height=img_info[0]
    image_weight=img_info[1]
    mat_src=np.float32([[0,0],[0,image_height-1],[image_weight-1,0]])
    
    x1 = np.random.uniform(0, 50)
    y1 = np.random.uniform(0, 50)
    
    x2 = np.random.uniform(200, 400)
    y2 = np.random.uniform(300, 500)
    
    x3 = np.random.uniform(200, 400)
    y3 = np.random.uniform(300, 500)
    
    mat_dst=np.float32([[x1,y1],[x2,image_height-y2],[image_weight-x3,y3]])
    mat_Affine=cv2.getAffineTransform(mat_src,mat_dst)
    dst=cv2.warpAffine(image,mat_Affine,(image_height,image_weight))
    return dst


def Crop(image):
    rows,cols,channel = image.shape
    L_delta = int(np.random.uniform(1, cols*0.15))
    R_delta = int(np.random.uniform(1, cols*0.15))
    U_delta = int(np.random.uniform(1, rows*0.15))
    D_delta = int(np.random.uniform(1, rows*0.15))
    TOP = 0 + L_delta
    DOWN = rows - R_delta
    LEFT = 0 + U_delta
    RIGHT = cols - D_delta
    crop_img = image[TOP:DOWN, LEFT:RIGHT]
    dst = cv2.copyMakeBorder(crop_img, L_delta, R_delta, U_delta , D_delta, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    return dst

def Hsv(image):
    hue_vari = 1
    sat_vari = 0.5
    val_vari = 0.5
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    
    dst = cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return dst

def Gamma(image):
    gamma_vari = 0.15
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    dst = cv2.LUT(image, gamma_table)
    return dst


def Motion_blur(image):
    image = np.array(image)
    degree_ = 25
    angle_ = 45
    degree = int(np.random.uniform(1, degree_))
    angle = int(np.random.uniform(-angle_, angle_))
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    dst = np.array(blurred, dtype=np.uint8)
    return dst

def Gaussian_blur(image):

    kernel = [random.randint(1, 50) * 2 + 1 for x in range(1)]
    dst = cv2.GaussianBlur(image, ksize=(kernel[0], kernel[0]), sigmaX=0, sigmaY=0)
    return dst


def imageProcessing(image, image_name, function, times, write_path):  
    for _i in range(0, times, 1):
        if (function == 'Rotation'):
            new_image = Rotation(image)
        elif (function == 'Crop'):
            new_image = Crop(image)
        elif (function == 'Hsv'):
            new_image = Hsv(image)
        elif (function == 'Gamma'):
            new_image = Gamma(image)
        elif (function == 'Motion_blur'):
            new_image = Motion_blur(image)
        elif (function == 'Gaussian_blur'):
            new_image = Gaussian_blur(image)
        elif (function == 'Translate'):
            new_image = Translate(image)
        elif (function == 'Affine'):
            new_image = Affine(image)
        else:
            new_image = image
        
        path = write_path+image_name+"_"+function+"_"+str(_i)+".JPG"
        cv2.imwrite(path ,new_image)
