#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:16:45 2022

@author: yangweian
"""

# import os
import torch
# from torch import nn
import numpy as np
import numpy.ma as ma #masked array
import cv2
import matplotlib.pyplot as plt
import copy
#import utils
import time
import torchvision
from scipy.ndimage.morphology import distance_transform_edt
# %matplotlib
# %matplotlib inline

class fDDFT():
    def __init__(self, I):
        #normalized data
        # self.I = (I/255) ** 2
        #we use this (v) because we have already used fermi, which already normalized our input data
        self.I = I ** 2
        # self.I = torchvision.transforms.Resize(
        #     (int(I.shape[-2]/2), int(I.shape[-1])))(I ** 2)        
        self.PI = torch.acos(torch.zeros(1)).item() * 2
        self.KE = 2 * self.PI * self.I
        # self.VE = self.V()
        self.VE = self.Vr() #calling function
        self.gamma = 0.5 * torch.mean(self.VE)/torch.mean(self.KE)
        #print("initial gamma=", self.gamma)
        # self.gamma = torch.tile(0.5 * torch.mean(self.VE)/torch.mean(self.KE), I.shape[-2::])
    
    def Vr(self): #more reduce vector
        '''
        You can add more dimension to the images
        '''
        I = torchvision.transforms.Resize(
            (int(self.I.shape[-2]/2), int(self.I.shape[-1]/2)))(self.I)
        # you can change /'2' into bigger number.
        # dtype : torch.tensor shape[channel, height, width]
        h, w = I.shape[-2::] 
        I = I/2        
        # h, w = self.I.shape[-2::] 
        # I = self.I/2
        # dist
        dist = torch.ones((h*2-1, w*2-1)) # (h*2-1, w*2-1)
        dist[h-1, w-1] = 0
        dist = distance_transform_edt(dist)  #edt = euclidean distance transform       
        dist[h-1, w-1] = 1
        dist_inv = torch.tensor(1/dist)
        dist_inv[h-1,w-1] = 0 # distance map done


        # FFT
        # image_shape(h,w) + dist_shape(2h-1,2w-1) -1 ----> (3*h-2, 3*w-2)
        fft_shape = (3*h-2, 3*w-2) 
        dist_fft = torch.fft.rfftn(dist_inv, fft_shape)#.cuda() # distance_fft2

        I_fft = torch.fft.rfftn(I, fft_shape)#.cuda() # img_fft2
        # multiply img_fft2 and distance_fft2 then do rfft2
        VE = torch.fft.irfftn(I_fft * dist_fft, fft_shape) 
        # pick middle of the result
        VE = VE[..., h-1: 2*h-1, w-1: 2*w-1] 
        VE = torchvision.transforms.Resize(
            (self.I.shape[-2], self.I.shape[-1]))(VE)

        return VE

    def V(self):
        '''
        You can add more dimension to the images
        '''

        h, w = self.I.shape[-2::] 
        I = self.I/2
        # dist
        dist = torch.ones((h*2-1, w*2-1)) # (h*2-1, w*2-1)
        dist[h-1, w-1] = 0
        dist = distance_transform_edt(dist)        
        dist[h-1, w-1] = 1
        dist_inv = torch.tensor(1/dist)
        dist_inv[h-1,w-1] = 0 # distance map done


        # FFT
        # image_shape(h,w) + dist_shape(2h-1,2w-1) -1 ----> (3*h-2, 3*w-2)
        fft_shape = (3*h-2, 3*w-2) 
        dist_fft = torch.fft.rfftn(dist_inv, fft_shape)#.cuda() # distance_fft2

        I_fft = torch.fft.rfftn(I, fft_shape)#.cuda() # img_fft2
        # multiply img_fft2 and distance_fft2 then do rfft2
        VE = torch.fft.irfftn(I_fft * dist_fft, fft_shape) 
        # pick middle of the result
        VE = VE[..., h-1: 2*h-1, w-1: 2*w-1] 


        return VE

    def KED(self): #kinetic energy density
        KED = self.KE * self.gamma ** 2
        return KED

    def VD(self): #potential energy density
        VD = self.VE * self.gamma
        return VD

    def HED(self):
        HED = self.KED() + self.VD()
        return HED / torch.max(HED)

    def LED(self):
        LED = self.KED() - self.VD()
        return LED / torch.max(LED)

    def thre(self):
        thre = self.LED().mean()
        return thre

    def out(self, thre = None):
        if thre == None:
            thre = self.thre()
        out = (self.LED() > thre)# * self.LED()    
        return out

    def Update(self): #code can be changed
        #LEDloss = self.thre()
        ledstep = 0
        
        while self.thre() < 0:     
            meanVD = torch.mean(self.VD()) * 2
            meanKED = torch.mean(self.KED())    
            # grad = torch.mean(torch.gradient(torch.mean(self.LED(), axis = 0), dim = -2)[0])
            # tmp = (meanVD / meanKED) * grad # 加入邊界資訊更新
            # self.gamma += torch.mean(tmp) # 加入邊界資訊更新
            self.gamma += 0.5*(meanVD / meanKED)
            ledstep += 1
            #print("LEDloss = {}".format(self.thre()))

        #print("Update Complete, gamma = {}, step={}".format(float(self.gamma), ledstep))

    # plot
    def plot(self, z, name):
        # z is the 2D energy map 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = [i for i in range(self.I.shape[-1])]
        y = [i for i in range(self.I.shape[-2])]
        x, y = np.meshgrid(x, y)
        z = np.array(z) 
        # z = z/np.max(z) # normalize to plot the results
        ax.plot_surface(x, y, z, cmap='coolwarm')
        plt.title(name)
        plt.show()
    
def compare_plot(*data):
    num = len([*data])
    plt.figure(figsize=(num*5, 5))
    for i, data in enumerate([*data]):
        plt.subplot(1, num, i+1)
        plt.imshow(data)
    plt.show()
        

#%% Read images
if __name__ == '__main__':      
#    data = cv2.imread("/Users/yangweian/Documents/BIOMIL/DDFT/BraTS2020/patient_1/img_1_3_78.png")
    data = cv2.imread("D:\Vin\BraTS22_Modified\Raw_png_100\Patient_1\Patient_1_1_93.png")
#    gt = cv2.imread("/Users/yangweian/Documents/BIOMIL/DDFT/BraTS2020/patient_1/img_1_5_78.png")
    gt = cv2.imread("D:\Vin\BraTS22_Modified\Raw_png_100\Patient_1\Patient_1_5_93.png")
#%% 
    #Delete modalities which gt is all zero
    if gt.max() != 0:             
        img = data.copy()
        zp = np.sum(~(img>0))/(240*240*3)
        if img.max() == 0 or zp >= 0.87:
            print("Nothing in the image...")
        else:    
#%% Preprocessing ==> Fermi Normalization
            h, w = img.shape[:2]
            # resized = cv2.resize(img, (h//2, w//2))
            resized = cv2.resize(img, (h, w))    
            masked = ma.masked_where(resized < resized.mean(), resized).astype(np.float32)
#            ft = utils.FermiTransform(masked)
#            img_fermi = ft.fermi()
            
            def fermi(image_data):
                return 1/(np.exp(-1*(image_data-np.mean(image_data))/round(np.std(image_data),2))+1)

            # img_fermi = fermi(masked)
            
            # # Fermi Update process
            # # img_fermi = ft.update(5)
            # img_n = img_fermi/img_fermi.max() #every cell divide by max number in the entire image
            # #img_fermi.max() = 0.9648
            # compare_plot(img, img_n, img_fermi) #compare 2 plot
#%%  fDDFT concept  
            a = torch.tensor(masked.copy())
            a = a.permute(2,0,1)
            model = fDDFT(a)
            #model.plot(torch.mean(a, axis = 0), "input")
            # Plot Energy        
            # model.plot(torch.mean(model.KED(), axis = 0), "KED")# KED           
            # model.plot(torch.mean(model.VD(), axis = 0), "VD")# VD           
            # model.plot(torch.mean(model.HED(), axis = 0), "HED")# HED           
            # model.plot(torch.mean(model.LED(), axis = 0), "LED")# LED     
            
            # Update model
            model.Update()
            
            # Plot Energy     
            # model.plot(torch.mean(model.KED(), axis = 0), "KED")# KED           
            # model.plot(torch.mean(model.VD(), axis = 0), "VD")# VD           
            # model.plot(torch.mean(model.HED(), axis = 0), "HED")# HED           
            # model.plot(torch.mean(model.LED(), axis = 0), "LED")# LED 

            attention = model.VD()/torch.max(model.VD()) # self-attention
            # model.plot(torch.mean(attention, axis = 0), "self-attention")
            # model.plot(torch.mean(attention * model.out(0), axis = 0), "self-attention*Mask")
            a = (attention * model.out(0)) * a 
            # system weineng * mask model * raw data
            # only the first a is the input, the next one is updtaed version
            #model.plot(torch.mean(a, axis = 0), "Next input")
            #compare_plot(a.permute(1,2,0)/torch.max(a))            
            
#%%        
            a = torch.tensor(masked.copy())
            a = a.permute(2,0,1)
            models = []
            HED = []
            #start = time.time()
            #compare_plot(masked)
            #print("DDFT update process...") 
            for i in range(3): #this is the sum of a, here loop 3
            # while True:    
                print(f"iter: {i}")
                model = fDDFT(a)              
                models.append(copy.copy(model))
                model.Update()
                                 
                HED.append(torch.mean(model.HED()))
                print(HED[-1])
                
                attention = model.VD()/torch.max(model.VD()) # 用位能更新
                a = (attention * model.out(0)) * a #changable
                compare_plot(a.permute(1,2,0)/torch.max(a))
                # a = (attention - torch.mean(attention)) * a 

                if i > 0:
                    if abs(HED[i]-HED[i-1]) < 1e-2 and torch.mean(model.HED()) < 1e-2:
                        #print("System Reaches Stable")
                        break        
                
            #end = time.time()
            #model.plot(torch.mean(model.out(0)*1.0, axis = 0), "output Mask")
            #print('elapsed time = {:.2f}'.format(end-start))
            
            out = model.out(thre=0.3).permute(1,2,0)#/out_mean
                    #MaskedRaw
            out = np.multiply(out,data)
            #compare_plot(out)
            # show HED difference between each model
            # for i in range(len(models)-1):
            #     print(HED[i+1] - HED[i])
            
            # Plot Energy     
            # model.plot(torch.mean(model.KED(), axis = 0), "KED")# KED           
            # model.plot(torch.mean(model.VD(), axis = 0), "VD")# VD           
            # model.plot(torch.mean(model.HED(), axis = 0), "HED")# HED           
            # model.plot(torch.mean(model.LED(), axis = 0), "LED")# LED 
        
            # for m in models:
            #     m.plot(torch.mean(m.LED(), axis = 0), "LED")
        
            # plot iter HE
            # plt.figure()
            # plt.plot(HED)
        
            # plot results
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ax1 = fig.add_subplot(132)
            ax2 = fig.add_subplot(133)
            ax.imshow(img)
            ax.set_title('Raw data')
            out = model.out().permute(1,2,0)
            tmp = out[..., 0] * 255
            out = torch.tensor(resized.copy()) * ~out
            out[...,0] +=  tmp
            ax1.imshow(out)
            ax1.set_title('Prediction')
            # gt = np.stack((gt_resized, gt_resized, gt_resized), axis = -1)
            # gt = braTS2020.edema[braTS2020.idx].copy()
            gtvis = torch.tensor(img) * ~(gt>0)
            gtvis[...,2] += gt[...,0] * 255
            ax2.imshow(gtvis.numpy(), cmap='gray')
            ax2.set_title('GT')
            plt.show()
            compare_plot(img, out, gtvis.numpy())
            
            # print dice score
            gt_resized = cv2.resize(gt.copy(), (h//2, w//2))
            # dice = utils.dice_coef2(gt_resized[...,0]>0, model.out().permute(1,2,0).numpy()[...,0])
            #dice = utils.dice_coef2(gt[...,0]>0, model.out().permute(1,2,0).numpy()[...,0])

            #use code below bcs there is no utils.dice_coef2
            def dice_coeff(pred,target):
                pred = torch.tensor(pred)
                target = torch.tensor(target)
                numerator = 2 * torch.sum(pred * target)
                denominator = torch.sum(pred + target)
                return 1 - (numerator + 1) / (denominator + 1)    
            
            dice = dice_coeff(gt[...,0]>0, model.out().permute(1,2,0).numpy()[...,0])

            print('dice = {}'.format(dice))
    else:
         print("Nothing in the image... bro")