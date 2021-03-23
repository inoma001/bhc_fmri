
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 2020

@author: mike germuska
"""
import numpy as np
import joblib
import pickle
from scipy import linalg
from scipy import sparse
import pyfftw

def create_HP_filt(flength,cutoff,TR):
    cut=cutoff/TR
    sigN2=(cut/np.sqrt(2))**2
    K=linalg.toeplitz(1/np.sqrt(2*np.pi*sigN2)*np.exp(-np.linspace(0,flength,flength)**2/(2*sigN2)))
    K=sparse.spdiags(1/np.sum(K,axis=0),0,flength,flength)*K
    H=np.zeros([flength,flength])
    X=np.array([np.ones(flength), range(1,flength+1)])
    X=np.transpose(X)
    for i in range(flength):
        W=np.diag(K[i])
        Hat=np.dot(np.dot(X,linalg.pinv(np.dot(W,X))),W)
        H[i]=Hat[i]
    HPfilt=np.eye(flength)-H
    return HPfilt


def calc_cmro2(images_dict,d_phys,d_scan_par,d_analysis):

    print('pre-processing ASL and BOLD data')


    #scale echo1 1 data by M0 and threshold out low M0 values (also scale by 100)
    x_axis,y_axis,no_slices,datapoints=np.shape(images_dict['echo1_data'])
    image_data=np.zeros([x_axis,y_axis,no_slices,datapoints])
    for i in range(datapoints):
        with np.errstate(divide='ignore',invalid='ignore'):
            image_data[:,:,:,i]=100*(np.divide(images_dict['echo1_data'][:,:,:,i],images_dict['M0_data']))
            image_data[:,:,:,i][images_dict['M0_data']<d_analysis['M0_cut']]=0
     
    flow_data=np.empty([x_axis,y_axis,no_slices,datapoints-2]) # pre-allocate array
    # matrix surround subtraction for both c-(t0+t2)/2  and t+(c0+c2) to get perfusion data
    # for even data points
    flow_data=image_data[:,:,:,1:-1]-(image_data[:,:,:,0:-2]+image_data[:,:,:,2:])/2
    # for odd data points
    flow_odd=-image_data[:,:,:,1:-1]+(image_data[:,:,:,0:-2]+image_data[:,:,:,2:])/2
    # add in odd data points
    flow_data[:,:,:,1::2]=flow_odd[:,:,:,1::2]



    # surround average to get BOLD data
    bold_data=(images_dict['echo2_data'][:,:,:,1:-1]+(images_dict['echo2_data'][:,:,:,0:-2]+images_dict['echo2_data'][:,:,:,2:])/2)/2

    # mask BOLD data (should make dimensinality reduction work better - apart from divide by zero problems)
    for i in range(datapoints-2):    
        bold_data[:,:,:,i][images_dict['M0_data']<d_analysis['M0_cut']]=0

  
   # convert into percent signal change
    per_bold=np.empty([x_axis,y_axis,no_slices,datapoints-2]) # pre-allocate array
    baseline=np.mean(bold_data[:,:,:,0:4],axis=3)
    for i in range(datapoints-2):
        with np.errstate(divide='ignore', invalid='ignore'):
            per_bold[:,:,:,i]=np.divide(bold_data[:,:,:,i],baseline)
            per_bold[:,:,:,i][baseline==0]=0
    per_bold=(per_bold-1)

    cut=300; 
    HPfilt=create_HP_filt(117,cut,4.4)
        
    #     HP filter data
    print('HP filt BOLD data')
    for i in range(x_axis):    
        for j in range(y_axis): 
            for k in range(no_slices): 
                per_bold[i,j,k,:]=per_bold[i,j,k,:]-np.mean(per_bold[i,j,k,0:4])
                per_bold[i,j,k,:]=np.dot(HPfilt,per_bold[i,j,k,:])
                per_bold[i,j,k,:]=per_bold[i,j,k,:]-np.mean(per_bold[i,j,k,0:4])


  
    per_bold=np.nan_to_num(per_bold)

    print('pyfftw FFT')
    
        
#    
    # calculate the FFT of BOLD and ASL data
    # FFTW is faster than numpy fft so use this.

#    import pre-computed fftw wisdom for these datasets for significant speed-up
#    fft_wisdom=pickle.load(open('fft_wisdom.sav', 'rb')) 
#    pyfftw.import_wisdom(fft_wisdom)

    pyfftw.interfaces.cache.enable()
    
    BOLD_fft=pyfftw.interfaces.numpy_fft.fft(per_bold)
    ASL_fft=pyfftw.interfaces.numpy_fft.fft(flow_data)

    # Now calculate CBF0
    print('predicting CBF0')
    PLD_vect=np.linspace(d_scan_par['PLD'],d_scan_par['PLD']+no_slices*d_scan_par['slice_delay'], num=no_slices)
    PLD_mat=np.tile(PLD_vect, (x_axis,y_axis,1))

    array_elements=15;
    ML_array=np.empty([x_axis,y_axis,no_slices,3+2*array_elements])
    ML_array[:,:,:,0]=d_phys['Hb']
    ML_array[:,:,:,1]=d_phys['CaO20']
    ML_array[:,:,:,2]=PLD_mat
    ML_array[:,:,:,3:3+array_elements]=np.absolute(ASL_fft[:,:,:,0:array_elements])
    ML_array[:,:,:,3+array_elements:3+2*array_elements]=np.absolute(BOLD_fft[:,:,:,0:array_elements])

    ML_array=np.reshape(ML_array,(x_axis*y_axis*no_slices, 3+2*array_elements))

  
    filename='CBF0_lightGBM_no_noise_50k_model.pkl'
    net=joblib.load(filename)  
    filename='CBF0_lightGBM_no_noise_50k_scaler.pkl'
    scaler=joblib.load(filename)

    X_train_scaled=scaler.transform(ML_array)
   
    CBF0_vect=net.predict(X_train_scaled)*150
    CBF0=np.reshape(CBF0_vect, (x_axis,y_axis,no_slices))  

    CBF0[CBF0<0]=0
    CBF0[CBF0>250]=250
    CBF0[images_dict['M0_data']<d_analysis['M0_cut']]=0

    array_elements=15;
    ML_array=np.empty([x_axis,y_axis,no_slices,3+2*array_elements -1])
    ML_array[:,:,:,0]=d_phys['Hb']
    ML_array[:,:,:,1]=d_phys['CaO20']
    ML_array[:,:,:,2]=PLD_mat
    ML_array[:,:,:,3:3+array_elements]=np.absolute(ASL_fft[:,:,:,0:array_elements])
    ML_array[:,:,:,3+array_elements:3+2*array_elements - 1]=np.absolute(BOLD_fft[:,:,:,1:array_elements])
    ML_array=np.reshape(ML_array,(x_axis*y_axis*no_slices, 3+2*array_elements -1))


    print('Calculating CBF tSNR')

    CBFsd=np.copy(CBF0)
    CBFmean=np.copy(CBF0)
    CBFsd=np.std(flow_data[:,:,:,0:20],axis=3)
    CBFmean=np.mean(flow_data[:,:,:,0:20],axis=3)
    CBFsd[CBF0<60]=np.nan
    CBFmean[CBF0<60]=np.nan
    print('ASL tSNR')
    print(np.nanmean( np.divide(CBFmean, CBFsd) ))

    print('Calculating BOLD tSNR')

    BOLDsd=np.copy(CBF0)
    BOLDmean=np.copy(CBF0)
    BOLDsd=np.std(bold_data[:,:,:,0:20],axis=3)
    BOLDmean=np.mean(bold_data[:,:,:,0:20],axis=3)
    BOLDsd[CBF0<60]=np.nan
    BOLDmean[CBF0<60]=np.nan
    print('BOLD tSNR')
    print(np.nanmean( np.divide(BOLDmean, BOLDsd) ))


    print('predicting OEF')

#   ensemble fitting
    import os 
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    ensembleDirectory=os.path.join(scriptDirectory, 'OEF_ensemble_sav/')  
    file_list=os.listdir(ensembleDirectory) 

    OEF_array=np.zeros([x_axis,y_axis,no_slices,int(len(file_list)/2)])
    
    array_counter=-1
    for i in range(len(file_list)):
        current_file=file_list[i]
        if current_file[-9:-4]=='model':
            filename='OEF_ensemble_sav/' + current_file
            
            print(filename)
            
            net=joblib.load(filename)
            filename='OEF_ensemble_sav/' + current_file[0:-9] + 'scaler.pkl'
            scaler=joblib.load(filename) 
            X_train_scaled=scaler.transform(ML_array)

 # use CMRO2 regressor
            CMRO2_vect=net.predict(X_train_scaled)*500
            with np.errstate(divide='ignore',invalid='ignore'):
                OEF_vect = np.divide ( CMRO2_vect , (d_phys['CaO20']*39.34*CBF0_vect) )

          #  OEF_vect= net.predict(X_train_scaled)
            OEF_local=np.reshape(OEF_vect, (x_axis,y_axis,no_slices))

            array_counter+=1
            print(array_counter)
            OEF_array[:,:,:,array_counter]=OEF_local
            
                        
    OEF_array[np.isnan(OEF_array)]=0
    OEF_array[np.isinf(OEF_array)]=0   
    OEF=np.mean(OEF_array,3) 
    
#   limit impossible answers 
    OEF[OEF>=1]=1
    OEF[OEF<0]=0

    
#   calculate CMRO2 
    OEF[images_dict['M0_data']<d_analysis['M0_cut']]=0    
    CMRO2=OEF*39.34*d_phys['CaO20']*CBF0

#  rational equation solution from CBF0 and CMRO2 to M (estimate with R^2=0.9)
    with np.errstate(divide='ignore',invalid='ignore'):
        M = (-0.04823*CMRO2 + 0.01983*CMRO2*CMRO2) / (29.19*CBF0 + 0.9426*CBF0*CBF0)

    M[images_dict['M0_data']<d_analysis['M0_cut']]=0  

# calculate Dc, O2 diffusivity from CBF0 and M (R^2 = 0.88 and RMSE = 0.33)
    Dc=0.1728 + 0.03024*CBF0 + 8.4*M - 0.0003404*CBF0*CBF0 + 1.101*CBF0*M - 36.44*M*M +4.559E-6*CBF0*CBF0*CBF0 -0.01734*CBF0*CBF0*M - 1.725*CBF0*M*M - 1.755E-8*CBF0*CBF0*CBF0*CBF0 + 6.407E-5*CBF0*CBF0*CBF0*M + 0.03734*CBF0*CBF0*M*M

    Dc[images_dict['M0_data']<d_analysis['M0_cut']]=0      

    return CMRO2, CBF0, OEF, M, Dc

