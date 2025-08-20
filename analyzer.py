# RUN IMPORT LIBRARIES
############################################################################################
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tifffile as tiff
# from PIL import Image
import h5py
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max

import brighteyes_ism.analysis.Graph_lib as gra

import scipy as sc

from os import path
import plotly.express as px
from time import time

file = 'Qdots_770nm_34mW_FOV3um_32x32_50rip_5msPixel_ROI7'

# Qdots_37mW_FOV5um_64x64_100rip_5msPixel # fototubo?
# Qdots_770nm_34mW_FOV4um_64x64_100rip_1msPixel_ROI8
# Qdots_770nm_34mW_FOV3um_32x32_50rip_5msPixel_ROI7
# Qdots_770nm_34mW_FOV5um_64x64_30rip_5msPixel_ROI5
# Qdots_770nm_34mW_FOV5um_64x64_50rip_5msPixel_ROI4
# Qdots_770nm_34mW_FOV5um_64x64_60rip_2msPixel_ROI6

####################################################################################
################################################## INPUT SECTION:
####################################################################################

# RUN INPUT
#############################################################################################

#############################################################################################
# FILE IMPORT AND SETUP
############################################################################################

dir = os.getcwd()
filename = path.join(dir,'images','2025 07 07-08 Qdots',file) # input
savename = path.join(dir,'images','output',file) # output
savenameSOFI = path.join(dir,'images','output','SOFI',file)

timeSwitch = True #True if i saved the timetraces
spadFormat=7
Nspad = spadFormat**2

index7=[i for i in range(49) if i != 12]
index5=[8,9,10,11,15,16,17,18,19,22,23,24,25,26,29,30,31,32,33,36,37,38,39,40] #Check if 12 is included or not
index3=[16,17,18,23,24,25,30,31,32]
index1=[17,23,24,25,31]


f = h5py.File(filename + '.h5', 'r') # read h5 file

# ----------------- read header

if timeSwitch:
    group = f['Fluctuations']
    print('Opening time data...')
    Endoftime_h5 = group['IsEndOfTimeBin'][()]
    t_bin=group['Time'][()][0]/1000        #micros
    rep = group.attrs['I_Repetitions'][0]      #repetitions  
else:
    group = f['Image']

imgFormat=group.attrs['I_Height'][0]       #pixels
Depth=group.attrs['I_Depth'][0]            #repetitions/z
ImgFoV=group.attrs['I_XSpan[um]'][0]       #micron
pxsizex=group.attrs['I_XPixelSize[nm]'][0]/1000 #micron         
ZSpan=group.attrs['I_ZSpan[um]'][0]        #micron
pxsizez=group.attrs['I_ZPixelSize[nm]'][0]/1000 #micron
Duration=group.attrs['M_Duration[ms]'][0]  #ms
DwellTime=Duration/(Depth*imgFormat**2)    #ms (approximate)
Fingerprint_h5 = group['Fingerprint'][()]

# ---------------- reading SPAD DATA

print('Opening SPAD data...')
data_h5= group['SPAD'][()]
print('Finished opening')
f.close()

# ----------------- display metadata

print('--------------------METADATA:--------------')
print('imgFormat: ',imgFormat,'x',imgFormat,'px')
print('depth (z): ', Depth)
print('FOV: ', ImgFoV,'μm')
print('pixelsize (x-y): ', pxsizex,'μm')
print('pixelsize (z): ', pxsizez,'μm')
print('ZSpan (z-range):', ZSpan,'μm')
print('Duration frame:', Duration,'ms')
print('Dwell time (time per pixel):', DwellTime,'ms')
print('---------------------------------------------')

# ---------------- fixing binning

# Ch,T,X,Y,R,Z
# print('Full data size: ' + str(data_h5.shape[0]))
# print('Total time bins (no channel distinction): ' +str(Endoftime_h5.shape[0]))

endtime=np.asarray(np.where(Endoftime_h5==1))[0,:]       #Cerco gli indici di Endoftime_h5 dove finiscono i pixels
# print('Scan pixels: ' + str(endtime.shape[0]))
# print('\nendtime: '+str(endtime[:10]))
v = np.max(np.diff(endtime))                             #Cerco la durata massima
# print('maximum temporal timetrace length: ' +str(v))                  
overflow = endtime[np.asarray(np.where(np.diff(endtime)==v))[0,:]+1]  #Indici di cut degli elementi in overflow
# print('\nOverflow: ' +str(overflow))
# print('Pixels in overflow: ' + str(overflow.shape[0]))

over_del = np.zeros(((overflow.shape[0]+1)*Nspad)).astype(np.int64)
for j in range(Nspad):
    over_del[(j)*overflow.shape[0]:overflow.shape[0]*(j+1)]= ((overflow*Nspad)+j)

g=np.linspace(0,48,49).astype(int)
if endtime[0]+1==v: 
    over_del[-49:]=((endtime[0]*Nspad)+g)
else:
    over_del=np.delete(over_del, np.linspace(over_del.shape[0]-Nspad, over_del.shape[0]-1).astype(int) , axis=(0))
    
# print('Bin in data_h5 to be removed: ' +str(over_del))

data_crop = np.delete(data_h5, over_del)
del over_del
del endtime
del g
del overflow
# print('\nFinal size of cropped data: ' +str(data_crop.shape[0]))
# print('Expected size: ' +str(Nspad*Depth*rep*imgFormat**2*(v-1)))

if not data_crop.shape[0]==Nspad*Depth*rep*imgFormat**2*(v-1):
    raise 'data error, bin not corresponding'
# print('size cropped == expected size: ', str(data_crop.shape[0]==Nspad*Depth*rep*imgFormat**2*(v-1)))

# ---------------------------- reshaping (z,rep,y,x,y,ch)

z_i, rep_i, y_i, x_i, t_i, ch_i = 0, 1, 2, 3, 4, 5 # indexes of reshaped data

fingerprint_meta = np.reshape(Fingerprint_h5, (spadFormat,spadFormat))
if not timeSwitch:
    data=np.reshape(data_h5,(Depth,1,imgFormat,imgFormat,1,Nspad)) 
    print('Data format (z,rep,y,x,t,ch) ' +str(data.shape))
else:
    data=np.reshape(data_crop, (Depth,rep,imgFormat,imgFormat,v-1,Nspad))
    print('Data format (z,rep,y,x,t,ch) ' +str(data.shape))
    del data_crop
   
# print('\nCumulative Dwell time (approx.): ' + str(DwellTime) +' ms')
# print('FoV [um]: ' + str(ImgFoV))
# if timeSwitch: print('Time binsize [us]: ' +str(t_bin))
# print('Pixel size [um]: ' +str(pxsizex))
del data_h5 # free the memory
del Endoftime_h5

#############################################################################################
# FIRST PLOT, all the data is in 'data'
############################################################################################

select_channel = 24

CENTRAL_SPAD = data[0, :, :, :, :, 24].sum(axis=(3, 0)).astype(np.uint8)
COFOCAL = data[0, :, :, :, :, :].sum(axis=(3, 0, 4)).astype(np.uint8)

SELECTED_CH = data[0, :, :, :, :, select_channel].sum(axis=(3, 0)).astype(np.uint8)

# im = Image.fromarray(CENTRAL_SPAD) # convert to an image

# -------------------- Plot and analysis

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8, 8)) 
gra.ShowImg(SELECTED_CH[:,:], pxsize_x = pxsizex, fig = fig, ax = ax1)
ax1.set_title(f'Channel: {select_channel}')

def correlazione(t1,m1,t2,m2):
    fft1=sc.fftpack.fft([(t1[i])-m1 for i in range(len(t1))])
    fft2=sc.fftpack.fft([(t2[i])-m2 for i in range(len(t2))])
    fft1=np.ma.conjugate(fft1)
    crosscorr=np.real(sc.fftpack.ifft(fft1*fft2))#/(m1*m2*len(t1)))
    return crosscorr

def onclick(event):
    # Get the x and y coordinates of the click
    if event.inaxes == ax1:
        if event.xdata is None or event.ydata is None:
            return  # click outside axes
        # Convert to integer pixel indices
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        # Make sure indices are inside image bounds
        if 0 <= x < SELECTED_CH.shape[1] and 0 <= y < SELECTED_CH.shape[0]:
            ax2.clear()
            ax2.set_title(f'Pixel ({x}, {y}) time trace')
            ax2.plot(np.reshape(data[0, :, y, x, :, select_channel], -1).astype(np.uint8))
            ax2.set_xlabel('Time (us)')
            ax2.set_ylabel('Intensity (a.u.)')  
            ax2.set_yticks(np.arange(0, 10))

            ax3.clear()
            ax3.set_title(f'Pixel ({x}, {y}) autocorrelation concat')
            t = np.reshape(data[0,:, y, x, :, select_channel], -1) # (rep t) -> (rep*t) concatenation
            tm = np.mean(t)
            ax3.plot(correlazione(t,tm,t,tm))
            ax3.set_xlabel('Lag (us)')
            ax3.set_ylabel('Autocorrelation (a.u.)')

            ax4.clear()
            ax4.set_title(f'Pixel ({x}, {y}) autocorrelation mean')
            ax4.set_xticks(np.arange(0,10))
            ax4.set_xlim(0,10)
            # Extract the slice
            slice_data = data[0, y, x, :, :, select_channel]  # shape: (rep, t)

            # Compute correlation along the last axis for each row
            correlations = [correlazione(rep,np.mean(rep),rep,np.mean(rep)) for rep in slice_data]

            # Take mean across rows (axis=0)
            mean_corr = np.mean(correlations, axis=0)

            ax4.plot(mean_corr)
            ax4.set_xlabel('Lag (us)')
            ax4.set_ylabel('Autocorrelation (a.u.)')
            
            plt.tight_layout()
            fig.canvas.draw()  # update the figure

            # print(f"Clicked pixel at ({x}, {y})")  # update the figure

def on_key(event):
    global select_channel
    if event.key == 'left':
        select_channel -= 1
        if select_channel < 0:
            select_channel = Nspad - 1
    if event.key == 'right':
        select_channel += 1
        if select_channel >= Nspad:
            select_channel = 0

    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    ax1.set_title(f'Channel: {select_channel}')
    SELECTED_CH = data[0, :, :, :, :, select_channel].sum(axis=(3, 0)).astype(np.uint8)
    gra.ShowImg(SELECTED_CH[:,:], pxsize_x = pxsizex, fig = fig, ax = ax1)
    plt.tight_layout()
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.tight_layout()
plt.show()


##################################################################################
###################################################################################
#################################################################################