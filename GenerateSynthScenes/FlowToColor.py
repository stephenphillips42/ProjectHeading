#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
# # 
###function colorwheel = makeColorwheel()

#    color encoding scheme

#    adapted from the color circle idea described at
#    http://members.shaw.ca/quadibloc/other/colint.htm

scale = 1
# Length of transitions between colors
RY = 15 # Red to Yellow
YG =  6 # Yellow to Green
GC =  4 # Green to Cyan
CB = 11 # Cyan to Blue
BM = 13 # Blue to Magenta
MR =  6 # Magenta to Red

# Color ranges
RY_range = np.arange(RY)
YG_range = np.arange(YG)
GC_range = np.arange(GC)
CB_range = np.arange(CB)
BM_range = np.arange(BM)
MR_range = np.arange(MR)

# Create colorwheel
ncols = RY + YG + GC + CB + BM + MR

colorwheel = np.zeros((ncols, 3)) #  r g b

col = 0
# RY
colorwheel[RY_range, 0] = 255
colorwheel[RY_range, 1] = (255*RY_range)/RY
col = col+RY

# YG
colorwheel[col+YG_range, 0] = 255 - (255*YG_range)/YG
colorwheel[col+YG_range, 1] = 255
col = col+YG

# GC
colorwheel[col+GC_range, 1] = 255
colorwheel[col+GC_range, 2] = (255*GC_range)/GC
col = col+GC

# CB
colorwheel[col+CB_range, 1] = 255 - (255*CB_range)/CB
colorwheel[col+CB_range, 2] = 255
col = col+CB

# BM
colorwheel[col+BM_range, 2] = 255
colorwheel[col+BM_range, 0] = (255*BM_range)/BM
col = col+BM

# MR
colorwheel[col+MR_range, 2] = 255 - (255*MR_range)/MR
colorwheel[col+MR_range, 0] = 255

img = np.zeros((128,32*ncols,3))
for x in xrange(ncols):
    img[:,(x*32):((x+1)*32),0] = colorwheel[x,0]/255.0
    img[:,(x*32):((x+1)*32),1] = colorwheel[x,1]/255.0
    img[:,(x*32):((x+1)*32),2] = colorwheel[x,2]/255.0


plt.imshow(img)
plt.show()



