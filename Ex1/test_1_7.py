#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:53:08 2018

"""
import exercise_1 as student
import scipy.ndimage
import matplotlib.pyplot as plt

img=scipy.ndimage.imread('tower.jpg')/255
plt.imshow(img)
img=student.seam_carving(img,it=48)
plt.figure()
plt.imshow(img)
