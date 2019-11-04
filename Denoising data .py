#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pywt
import sys
import pandas as pd
import numpy as np
from statsmodels.robust import mad


# In[2]:


input_data = pd.read_csv(r'C:\Users\apr0329\Desktop\files\apr\data\trainA\Train_A_002.csv')
input_data=input_data.values
input_data


# In[3]:


fx=np.asarray(input_data[:,0])
fy=np.asarray(input_data[:,1])
fz=np.asarray(input_data[:,2])
vx=np.asarray(input_data[:,3])
vy=np.asarray(input_data[:,4])
vz=np.asarray(input_data[:,5])
acc=np.asarray(input_data[:,6])


# for i in range(1, len(coeffs)):
#     plt.subplot(maxlev, 1, i)
#     plt.plot(coeffs[i])
#     coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
#     plt.plot(coeffs[i])

# In[4]:


def denoise(input_data,wavelet_type,threshold,mode):
    w=pywt.Wavelet(wavelet_type)
    maxlev = pywt.dwt_max_level(len(input_data), w.dec_len)
    coeffs = pywt.wavedec(input_data, wavelet_type , level=maxlev)
    #sigma=np.median(input_data)/0.6745
    #uthresh = sigma * np.sqrt( 2*np.log( len( input_data ) ) )
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]),mode)
        plt.plot(coeffs[i])
    denoised_data = pywt.waverec(coeffs, wavelet_type)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(input_data,'r')
    plt.xlabel('time (s)')
    plt.title("Input signal with noise")
    #plt.subplot(2, 1, 2)
    plt.plot(denoised_data)
    plt.xlabel('time (s)')
    plt.title("De-noised signal using wavelet techniques")
    plt.tight_layout()
    plt.show()


    


# In[5]:


deno_fx=denoise(fx[0:2000],wavelet_type='sym4',threshold=0.4,mode='soft')


# In[6]:


deno_fy=denoise(fy,wavelet_type='sym4',threshold=0.4,mode='soft')


# In[7]:


deno_fz=denoise(fz,wavelet_type='sym4',threshold=0.4,mode='soft')


# In[8]:


deno_vx=denoise(vx,wavelet_type='sym4',threshold=0.4,mode='hard')


# In[9]:


deno_vy=denoise(vy,wavelet_type='sym4',threshold=0.4,mode='hard')


# In[10]:


deno_vz=denoise(vz,wavelet_type='sym4',threshold=0.4,mode='hard')


# In[11]:


deno_acc=denoise(acc,wavelet_type='sym4',threshold=0.4,mode='soft')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




