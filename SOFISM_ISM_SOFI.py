# RUN IMPORT LIBRARIES
############################################################################################
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
import tifffile as tiff
from PIL import Image
import h5py
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max

import scipy as sc

import brighteyes_ism.analysis.APR_lib as apr
import brighteyes_ism.analysis.APR_lib as apr
import brighteyes_ism.analysis.FRC_lib as frc
import brighteyes_ism.analysis.Deconv_lib as dec
import brighteyes_ism.analysis.FocusISM_lib as fism
import brighteyes_ism.analysis.Graph_lib as gra
import brighteyes_ism.analysis.Tools_lib as tool
import brighteyes_ism.dataio.mcs as mcs
import brighteyes_ism.analysis.Deconv_lib as dec
import brighteyes_ism.simulation.PSF_sim as sim
from scipy.ndimage import fourier_shift, shift 
from os import path
from IPython.display import display, HTML
import plotly.express as px

from time import time

dir = os.getcwd()

timeSwitch = True #True if i saved the timetraces
spadFormat=7
Nspad = spadFormat**2

index7=[i for i in range(49) if i != 12]
index5=[8,9,10,11,15,16,17,18,19,22,23,24,25,26,29,30,31,32,33,36,37,38,39,40] #Check if 12 is included or not
index3=[16,17,18,23,24,25,30,31,32]
index1=[17,23,24,25,31]


filelist = [#'Qdots_770nm_34mW_FOV4um_64x64_100rip_1msPixel_ROI8',
            #'Qdots_770nm_34mW_FOV3um_32x32_50rip_5msPixel_ROI7',
            #'Qdots_770nm_34mW_FOV5um_64x64_30rip_5msPixel_ROI5', # slooooow
            # 'Qdots_770nm_34mW_FOV5um_64x64_50rip_5msPixel_ROI4', # sloooooow 
            'Qdots_770nm_34mW_FOV5um_64x64_60rip_2msPixel_ROI6'
]

for file in filelist:
    filename = path.join(dir,'images','2025 07 07-08 Qdots',file) # input
    savename = path.join(dir,'images','output',file) # output
    savenameSOFI = path.join(dir,'images','output','SOFI',file)
    COMPARATION_FILE_SAVING = path.join(dir,'images','output','COMPARISON',f'{file} COMPARATION FULL 2.png')

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

    # ---------------- fixing binning

    endtime=np.asarray(np.where(Endoftime_h5==1))[0,:]       #Cerco gli indici di Endoftime_h5 dove finiscono i pixels

    v = np.max(np.diff(endtime))                             #Cerco la durata massima
                
    overflow = endtime[np.asarray(np.where(np.diff(endtime)==v))[0,:]+1]  #Indici di cut degli elementi in overflow

    over_del = np.zeros(((overflow.shape[0]+1)*Nspad)).astype(np.int64)
    for j in range(Nspad):
        over_del[(j)*overflow.shape[0]:overflow.shape[0]*(j+1)]= ((overflow*Nspad)+j)

    g=np.linspace(0,48,49).astype(int)
    if endtime[0]+1==v: 
        over_del[-49:]=((endtime[0]*Nspad)+g)
    else:
        over_del=np.delete(over_del, np.linspace(over_del.shape[0]-Nspad, over_del.shape[0]-1).astype(int) , axis=(0))
        
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

    del data_h5 # free the memory

    CENTRAL_SPAD = data[0, :, :, :, :, 24].sum(axis=(3, 0)).astype(np.uint8)

    del Endoftime_h5


    ############################ ISM #########################################

        # RUN ISM
    #############################################################################################
    # CHECK FINGERPRINT
    ############################################################################################

    fingerprint = tool.fingerprint(data.sum(axis=(0,1,4)))
    fingerprint[1,5]=(fingerprint[1,4]+fingerprint[1,6])/2 #smooth spad 12
    fingerprint_meta[1,5]=(fingerprint_meta[1,4]+fingerprint_meta[1,6])/2
    c= int((spadFormat-1)/2)  #change it if the maximum is not at the center


    for i in range(7):
        for j in range(7):
            if str(round(fingerprint[i,j]*100/fingerprint[c,c])) != str(round(fingerprint_meta[i,j]*100/fingerprint_meta[c,c])):
                raise 'fingerprint not matching'
            # ax[0].text(j-0.3,i, str(round(fingerprint[i,j]*100/fingerprint[c,c])) + '%', color='limegreen')
            # ax[0].set_title('Experimental fingerprint')
            # ax[1].text(j-0.3,i, str(round(fingerprint_meta[i,j]*100/fingerprint_meta[c,c])) + '%', color='limegreen')
            # ax[1].set_title('Software fingerprint')

    # fig_1 = gra.ShowDataset(np.sum(data,axis=(z_i,rep_i,t_i)),normalize = False, colorbar=False) # SHOW 49 images

    #############################################################################################
    # data initialization
    ############################################################################################

    index7_fix=[i for i in range(48)]
    index5_fix=[8,9,10,11,14,15,16,17,18,21,22,23,24,25,28,29,30,31,32,35,36,37,38,39]
    index3_fix=[15,16,17,22,23,24,29,30,31]
    index1_fix=[16,22,23,24,30]

    # datacut = d7 - spad12
    datacut = np.sum(np.delete(data, 12,5)[0,:,:,:,:],axis=(0,3))[...] #Remove SPAD 12 -> output (y,x,ch)
    d3 = datacut[:,:,index3_fix]
    d5 = datacut[:,:,index5_fix]
    d1 = datacut[:,:,index1_fix]
    d7 = datacut[:,:,index7_fix]

    #############################################################################################
    # APR
    ############################################################################################

    #ISM image
    usf = 10  # upsampling factor = subpixel precision
    ref_D5 = 11 # reference image to compute the shift vectors (center spad)
    ref_D3 = 4  # reference image to compute the shift vectors (center spad)
    ref_D7 = 23 # reference image to compute the shift vectors (center spad)
    ref_D1 = 2 # reference image to compute the shift vectors (center spad)


    start = time()
    shift_vec7, ISM_D7_CH = apr.APR(d7, usf, ref_D7, filter_sigma=1, pxsize = pxsizex*1000) #pxsize in nm
    ISM_D7 = ISM_D7_CH.sum(axis=2)
    end = time()
    TIME_ISM_D7 = end - start
    print('done ISM D7: ', np.round(TIME_ISM_D7,2), 's')

    start = time()
    shift_vec5, ISM_D5_CH = apr.APR(d5, usf, ref_D5, filter_sigma=1, pxsize = pxsizex*1000) #pxsize in nm
    ISM_D5 = ISM_D5_CH.sum(axis=2)
    end = time()
    TIME_ISM_D5 = end - start
    print('done ISM D5: ', np.round(TIME_ISM_D5,2), 's')

    start = time()
    shift_vec3, ISM_D3_CH = apr.APR(d3, usf, ref_D3, filter_sigma=1, pxsize = pxsizex*1000)
    ISM_D3 = ISM_D3_CH.sum(axis=2)
    end = time()
    TIME_ISM_D3 = end - start
    print('done ISM D3: ', np.round(TIME_ISM_D3,2), 's')

    start = time()
    shift_vec1, ISM_D1_CH = apr.APR(d1, usf, ref_D1, filter_sigma=1, pxsize = pxsizex*1000)
    ISM_D1 = ISM_D1_CH.sum(axis=2)
    end = time()
    TIME_ISM_D1 = end - start
    print('done ISM D1: ', np.round(TIME_ISM_D1,2), 's')

    del d1,d3,d5,d7
    del ISM_D1_CH, ISM_D3_CH, ISM_D5_CH, ISM_D7_CH
    del shift_vec1, shift_vec3, shift_vec5, shift_vec7


    ################################### SOFI #####################################

# RUN SOFI
###################################################################################################
#                         FUNCTION DEFINITIONS
##################################################################################################

    def correlazione(t1,m1,t2,m2):
        fft1=scipy.fftpack.fft([(t1[i])-m1 for i in range(len(t1))])
        fft2=scipy.fftpack.fft([(t2[i])-m2 for i in range(len(t2))])
        fft1=np.ma.conjugate(fft1)
        crosscorr=np.real(scipy.fftpack.ifft(fft1*fft2))#/(m1*m2*len(t1)))
        return crosscorr

    def autocorrelazione(t,m):
        return correlazione(t,m,t,m)  #autocorrelation is the correlation of a signal with itself


    def SBR(img,xsi,xsf,xbi,xbf,ysi,ysf,ybi,ybf): # signal to backround ratio
        roi_s = img[xsi:xsf, ysi:ysf]
        roi_b = img[xbi:xbf, ybi:ybf]
        mean_roi_s = np.mean(roi_s)
        mean_roi_b = np.mean(roi_b)
        return mean_roi_s / mean_roi_b



    central=np.asarray(data[0,:,:,:,:,24], dtype='float')

    rep_ic, y_ic, x_ic, t_ic = 0, 1, 2, 3 # indexes of central reshaped data

    ###################################################################################################
    #                         SOFI MEAN
    ##################################################################################################

    start = time()
    SOFI_MEAN=np.zeros((central.shape[y_ic],central.shape[x_ic],central.shape[t_ic]), dtype=np.float64)
    for i in range(central.shape[y_ic]):
        for j in range(central.shape[x_ic]):
            count=0
            for r in range(central.shape[rep_ic]):
                signal_mean = np.mean(central[r,i,j,:])
                if signal_mean != 0: # media sui tempi e skip dove bkg==0
                    count += 1
                    SOFI_MEAN[i,j,:] = SOFI_MEAN[i,j,:] + autocorrelazione(central[r,i,j,:],signal_mean)
            if count!=0:
                SOFI_MEAN[i,j,:] = SOFI_MEAN[i,j,:]/count

    end = time()
    TIME_SOFI_MEAN = end - start
    print('done SOFI MEAN: ', np.round(TIME_SOFI_MEAN,2),'s')
    # SOFI_MEAN_SBR = SBR(SOFI_MEAN[:,:,0], xsi,xsf,xbi,xbf,ysi,ysf,ybi,ybf)
    # print('done SOFI MEAN - SBR: ', np.round(SOFI_MEAN_SBR,2))

    ###################################################################################################
    #                         SOFI SUM
    ##################################################################################################

    start = time()
    central_repsum = central.sum(axis=rep_ic)  # Sum over repetitions
    SOFI_SUM=np.zeros((central.shape[y_ic],central.shape[x_ic],central.shape[t_ic]), dtype=np.float64)
    for i in range(central.shape[y_ic]):
        for j in range(central.shape[x_ic]):
            SOFI_SUM[i,j,:] = autocorrelazione(central_repsum[i,j,:],central_repsum[i,j,:].mean())

    end = time()
    TIME_SOFI_SUM = end - start
    print('done SOFI SUM: ', np.round(TIME_SOFI_SUM,2),'s')
    # SOFI_SUM_SBR = SBR(SOFI_SUM[:,:,0], xsi,xsf,xbi,xbf,ysi,ysf,ybi,ybf)
    # print('done SOFI SUM - SBR: ', np.round(SOFI_SUM_SBR,2))

    ###################################################################################################
    #                         SOFI CONCAT
    ##################################################################################################

    start = time()
    central_concat = np.transpose(central,(1,2,0,3)).reshape(central.shape[y_ic], central.shape[x_ic], -1)  # (y,x,t*rep)

    SOFI_CONCAT=np.zeros((central.shape[y_ic],central.shape[x_ic],central.shape[t_ic]*central.shape[rep_ic]), dtype=np.float64)
    for i in range(central.shape[y_ic]):
        for j in range(central.shape[x_ic]):
            # concateno su tutte le rep
            # plt.plot(central_concat)
            signal_mean = central_concat[i,j,:].mean()
            SOFI_CONCAT[i,j,:] = autocorrelazione(central_concat[i,j,:], signal_mean)

    end = time()
    TIME_SOFI_CONCAT = end - start
    print('done SOFI CONCAT: ', np.round(TIME_SOFI_CONCAT,2),'s')
    # SOFI_CONCAT_SBR = SBR(SOFI_CONCAT[:,:,0], xsi,xsf,xbi,xbf,ysi,ysf,ybi,ybf)
    # print('done SOFI CONCAT - SBR: ', np.round(SOFI_CONCAT_SBR,2))

    del central,central_concat, central_repsum

#####################################################################################################
####################################################################################################
#                        SOFISM
#
#####################################################################################################
#                                                                                                   #
#####################################################################################################

    def batched_fft_correlate(sig1, sig2):
        """
        sig1, sig2: shape (ny, nx, nt)
        Returns correlation with shape (ny, nx, nt)
        mode='same' behavior
        """
        ny, nx, nt = sig1.shape
        # FFT length for 'same' mode
        nfft = nt
        f1 = np.fft.fft(sig1.reshape(-1, nt), n=nfft, axis=1)
        f2 = np.fft.fft(sig2.reshape(-1, nt), n=nfft, axis=1)
        corr = np.fft.ifft(f1 * np.conj(f2), axis=1).real
        # Shift so that center aligns like mode='same'
        corr = np.fft.fftshift(corr, axes=1)
        return corr.reshape(ny, nx, nt)

    # ############################# SOFISM D1

        # # ------------------------ d1 SOFISM CONCAT
    print('working on SOFISM d1... ')
    start = time()
    d1=np.asarray([data[0,:,:,:,:,i] for i in index1], dtype='float') # (z,rep,y,x,t,ch) -> (ch,rep,y,x,t)
    d1=np.transpose(d1,(2,3,4,1,0)) # (ch,rep,y,x,t) -> (y,x,t,rep,ch)

    d1 = d1.reshape(d1.shape[0],d1.shape[1],d1.shape[2]*d1.shape[3],d1.shape[4]) # (y,x,t,rep,ch) ->  (y,x,t*rep,ch)

    ny, nx, nt, nch = d1.shape
    imgN = int(nch * (nch + 1) / 2)
    print(d1.shape)

    SOFISM_CORRELATION_d1 = np.zeros((ny, nx, nt, imgN), dtype=float)

        # Precompute mean-subtracted signals for all channels
    d1_centered = d1 - d1.mean(axis=2, keepdims=True)

    chindex=-1

    import scipy as sc

    for ch1 in range(nch):
        sig1 = d1_centered[:, :, :, ch1]  # shape (ny, nx, nt)
        for ch2 in range(ch1, nch, 1):
            chindex += 1
            print(chindex, 'd1')
            sig2 = d1_centered[:, :, :, ch2]
            # corr = batched_fft_correlate(sig1, sig2)
            SOFISM_CORRELATION_d1[:, :, :, chindex] = batched_fft_correlate(sig1, sig2)

    usf = 10  # upsampling factor = subpixel precision
    ref_d1 = 5+4

    shift_vec_D1, SOFISM_D1_CH = apr.APR( SOFISM_CORRELATION_d1[:,:,0,:], usf, ref_d1, filter_sigma=1, pxsize = pxsizex*1000) #pxsize in nm
    SOFISM_D1 = SOFISM_D1_CH.sum(axis=2)

    end = time()
    TIME_SOFISM_D1 = end - start

    del SOFISM_D1_CH, shift_vec_D1, SOFISM_CORRELATION_d1

    print('done SOFISM D1: ' + str(int(TIME_SOFISM_D1/60)) + 'm' + str(int(TIME_SOFISM_D1%60)) + 's')
    # fig_1 = gra.Show(SOFISM_IMAGES_d1[:,:,0,:],normalize = False, colorbar=True) # SHOW 49 images

    ######################################## SOFISM D3 ####################################
    print('working on SOFISM d3... ')
    start = time()
    d3=np.asarray([data[0,:,:,:,:,i] for i in index3], dtype='float') # (z,rep,y,x,t,ch) -> (ch,rep,y,x,t)
    d3=np.transpose(d3,(2,3,4,1,0)) # (ch,rep,y,x,t) -> (y,x,t,rep,ch)

    d3 = d3.reshape(d3.shape[0],d3.shape[1],d3.shape[2]*d3.shape[3],d3.shape[4]) # (y,x,t,rep,ch) ->  (y,x,t*rep,ch)

    ny, nx, nt, nch = d3.shape
    imgN = int(nch * (nch + 1) / 2)
    print(d3.shape)

    SOFISM_CORRELATION_d3 = np.zeros((ny, nx, nt, imgN), dtype=float)

        # Precompute mean-subtracted signals for all channels
    d3_centered = d3 - d3.mean(axis=2, keepdims=True)

    chindex=-1

    import scipy as sc

    for ch1 in range(nch):
        sig1 = d3_centered[:, :, :, ch1]  # shape (ny, nx, nt)
        for ch2 in range(ch1, nch, 1):
            chindex += 1
            print(chindex, 'd3')
            sig2 = d3_centered[:, :, :, ch2]
            # corr = batched_fft_correlate(sig1, sig2)
            SOFISM_CORRELATION_d3[:, :, :, chindex] = batched_fft_correlate(sig1, sig2)

    usf = 10  # upsampling factor = subpixel precision
    ref_d3 = 9+8+7+6

    shift_vec_D3, SOFISM_D3_CH = apr.APR( SOFISM_CORRELATION_d3[:,:,0,:], usf, ref_d3, filter_sigma=1, pxsize = pxsizex*1000) #pxsize in nm
    SOFISM_D3 = SOFISM_D3_CH.sum(axis=2)

    end = time()
    TIME_SOFISM_D3 = end - start

    del SOFISM_D3_CH, shift_vec_D3, SOFISM_CORRELATION_d3

    print('done SOFISM D3: ' + str(int(TIME_SOFISM_D3/60)) + 'm' + str(int(TIME_SOFISM_D3%60)) + 's')


    #------------------------------ SOFISM D5

    # ------------------------ d5 SOFISM CONCAT
    print('working on SOFISM d5... ')
    start = time()
    d5=np.asarray([data[0,:,:,:,:,i] for i in index5], dtype='float') # (z,rep,y,x,t,ch) -> (ch,rep,y,x,t)
    d5=np.transpose(d5,(2,3,4,1,0)) # (ch,rep,y,x,t) -> (y,x,t,rep,ch)

    d5 = d5.reshape(d5.shape[0],d5.shape[1],d5.shape[2]*d5.shape[3],d5.shape[4]) # (y,x,t,rep,ch) ->  (y,x,t*rep,ch)

    ny, nx, nt, nch = d5.shape
    imgN = int(nch * (nch + 1) / 2)
    print(d5.shape)

    SOFISM_CORRELATION_d5 = np.zeros((ny, nx, nt, imgN), dtype=float)

        # Precompute mean-subtracted signals for all channels
    d5_centered = d5 - d5.mean(axis=2, keepdims=True)

    chindex=-1

    import scipy as sc

    for ch1 in range(nch):
        sig1 = d5_centered[:, :, :, ch1]  # shape (ny, nx, nt)
        for ch2 in range(ch1, nch, 1):
            chindex += 1
            print(chindex, 'd5')
            sig2 = d5_centered[:, :, :, ch2]
            # corr = batched_fft_correlate(sig1, sig2)
            SOFISM_CORRELATION_d5[:, :, :, chindex] = batched_fft_correlate(sig1, sig2)

    usf = 10  # upsampling factor = subpixel precision
    ref_d5 = 25+24+23+22+21+20+19+18+17+16+15

    shift_vec_D5, SOFISM_D5_CH = apr.APR( SOFISM_CORRELATION_d5[:,:,0,:], usf, ref_d5, filter_sigma=1, pxsize = pxsizex*1000) #pxsize in nm
    SOFISM_D5 = SOFISM_D5_CH.sum(axis=2)

    end = time()
    TIME_SOFISM_D5 = end - start

    del SOFISM_D5_CH, shift_vec_D5, SOFISM_CORRELATION_d5

    print('done SOFISM D5: ' + str(int(TIME_SOFISM_D5/60)) + 'm' + str(int(TIME_SOFISM_D5%60)) + 's')


    # # ------------------------ d7 SOFISM CONCAT
    try:
        print('working on SOFISM d7... ')
        start = time()
        d7=np.asarray([data[0,:,:,:,:,i] for i in index7], dtype='float') # (z,rep,y,x,t,ch) -> (ch,rep,y,x,t)
        d7=np.transpose(d7,(2,3,4,1,0)) # (ch,rep,y,x,t) -> (y,x,t,rep,ch)

        d7 = d7.reshape(d7.shape[0],d7.shape[1],d7.shape[2]*d7.shape[3],d7.shape[4]) # (y,x,t,rep,ch) ->  (y,x,t*rep,ch)

        ny, nx, nt, nch = d7.shape
        imgN = int(nch * (nch + 1) / 2)
        print(d7.shape)

        SOFISM_CORRELATION_d7 = np.zeros((ny, nx, nt, imgN), dtype=float)

            # Precompute mean-subtracted signals for all channels
        d7_centered = d7 - d7.mean(axis=2, keepdims=True)

        chindex=-1

        import scipy as sc

        for ch1 in range(nch):
            sig1 = d7_centered[:, :, :, ch1]  # shape (ny, nx, nt)
            for ch2 in range(ch1, nch, 1):
                chindex += 1
                print(chindex, 'd7')
                sig2 = d7_centered[:, :, :, ch2]
                # corr = batched_fft_correlate(sig1, sig2)
                SOFISM_CORRELATION_d7[:, :, :, chindex] = batched_fft_correlate(sig1, sig2)

        usf = 10  # upsampling factor = subpixel precision
        ref_d7 = 25+24+23+22+21+20+19+18+17+16+15

        shift_vec_D7, SOFISM_D7_CH = apr.APR( SOFISM_CORRELATION_d7[:,:,0,:], usf, ref_d7, filter_sigma=1, pxsize = pxsizex*1000) #pxsize in nm
        SOFISM_D7 = SOFISM_D7_CH.sum(axis=2)

        end = time()
        TIME_SOFISM_D7 = end - start

        del SOFISM_D7_CH, shift_vec_D7, SOFISM_CORRELATION_d7

        print('done SOFISM D7: ' + str(int(TIME_SOFISM_D7/60)) + 'm' + str(int(TIME_SOFISM_D7%60)) + 's')
    except Exception as e:
        print(repr(e))

    # DISPLAY ##############################################################################################################
    fig, ax = plt.subplots(3, 4, figsize=(5*4,15))

    gra.ShowImg(SOFISM_D1, pxsize_x = pxsizex, fig = fig, ax = ax[0,0])
    ax[0,0].set_title('SOFISM D1 ' + str(int(TIME_SOFISM_D1/60)) + 'm' + str(int(TIME_SOFISM_D1%60)) + 's')

    gra.ShowImg(SOFISM_D3, pxsize_x = pxsizex, fig = fig, ax = ax[0,1])
    ax[0,1].set_title('SOFISM D3 ' + str(int(TIME_SOFISM_D3/60)) + 'm' + str(int(TIME_SOFISM_D3%60)) + 's')

    gra.ShowImg(SOFISM_D5, pxsize_x = pxsizex, fig = fig, ax = ax[0,2])
    ax[0,2].set_title('SOFISM D5 ' + str(int(TIME_SOFISM_D5/60)) + 'm' + str(int(TIME_SOFISM_D5%60)) + 's')

    try:
        gra.ShowImg(SOFISM_D7, pxsize_x = pxsizex, fig = fig, ax = ax[0,3])
        ax[0,3].set_title('SOFISM D7 ' + str(int(TIME_SOFISM_D7/60)) + 'm' + str(int(TIME_SOFISM_D7%60)) + 's')
    except:
        pass

    gra.ShowImg(SOFI_CONCAT[:,:,0], pxsize_x = pxsizex, fig = fig, ax = ax[1,0])
    ax[1,0].set_title('SOFI CONCAT ' + str(np.round(TIME_SOFI_CONCAT,2)) + 's')

    gra.ShowImg(SOFI_MEAN[:,:,0], pxsize_x = pxsizex, fig = fig, ax = ax[1,1])
    ax[1,1].set_title('SOFI MEAN ' + str(np.round(TIME_SOFI_MEAN,2)) + 's')

    gra.ShowImg(SOFI_SUM[:,:,0], pxsize_x = pxsizex, fig = fig, ax = ax[1,2])
    ax[1,2].set_title('SOFI SUM ' + str(np.round(TIME_SOFI_SUM,2)) + 's')

    gra.ShowImg(CENTRAL_SPAD, pxsize_x = pxsizex, fig = fig, ax = ax[1,3])
    ax[1,3].set_title('CENTRAL SPAD ')

    gra.ShowImg(ISM_D1, pxsize_x = pxsizex, fig = fig, ax = ax[2,0])
    ax[2,0].set_title('ISM D1 ' + str(np.round(TIME_ISM_D1,2)) + 's')

    gra.ShowImg(ISM_D3, pxsize_x = pxsizex, fig = fig, ax = ax[2,1])
    ax[2,1].set_title('ISM D3 ' + str(np.round(TIME_ISM_D3,2)) + 's')

    gra.ShowImg(ISM_D5, pxsize_x = pxsizex, fig = fig, ax = ax[2,2])
    ax[2,2].set_title('ISM D5 ' + str(np.round(TIME_ISM_D5,2)) + 's')

    gra.ShowImg(ISM_D7, pxsize_x = pxsizex, fig = fig, ax = ax[2,3])
    ax[2,3].set_title('ISM D7 ' + str(np.round(TIME_ISM_D7,2)) + 's')

    fig.savefig(COMPARATION_FILE_SAVING, dpi=300)

    try:
        tiff.imwrite(path.join(dir,'images','output','COMPARISON',file + ' stack 2.tiff'),
                    [CENTRAL_SPAD, SOFI_CONCAT, SOFI_MEAN, SOFI_SUM, ISM_D1, ISM_D3, ISM_D5, ISM_D7, SOFISM_D1, SOFISM_D3, SOFISM_D5, SOFISM_D7])
    except:
        pass
