import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from nibabel.affines import apply_affine
 
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
import pandas as pd
from nilearn import masking
 
import seaborn as sns
 

from openpyxl import load_workbook,Workbook 
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


if __name__ == '__main__':
    
    correlation=np.zeros((9,55))
    
    ICA= nib.load('/Users/zhuyuling/Downloads/UKBiobank_BrainImaging_GroupMeanTemplates 2/rfMRI_ICA_d100.nii.gz')
    ICA_data= ICA.get_fdata()
    #ICA_1=ICA_data[:,:,:,0]
    print(ICA_data.shape)
    #men=nib.load('/Users/zhuyuling/Desktop/men.nii.gz')
    #men_img_data=men.get_fdata()
    #print(men_img_data.shape)
    #idx = np.argwhere(men_img_data == 1) 
    #ICA_1_index=np.where(np.rint(MNI152_1_ICA)==1)
    #ICA_1_1=MNI152_1_ICA[ICA_1_index]
    Yeo_img = nib.load('/Users/zhuyuling/Desktop/mentor/Yeo2011_7Networks_MNI152_FreeSurferConformed2mm_LiberalMask.nii.gz')
    Yeo_img_data = Yeo_img.get_fdata()
    print(Yeo_img_data.shape)
    subcortical=nib.load('/Users/zhuyuling/Desktop/mentor/subcortical_mask.nii.gz')
    subcortical_data=subcortical.get_fdata()
    cerebellum=nib.load('/Users/zhuyuling/Desktop/mentor/cerebellum_mask.nii.gz')
    cerebellum_data=cerebellum.get_fdata()
    cerebellum_data_1=cerebellum_data[:,:,:,0]
    print(subcortical_data.shape)
    ICA_21_index=[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    ICA_index=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,
               34,35,36,37,38,39,40,41,42,43,45,46,48,49,50,52,53,57,58,60,63,64,93]
    
    Yeo_4=list()
    for i in range(1,8):    
        yeo_index_1=np.where(np.rint(Yeo_img_data)==i)
        yeo_1=Yeo_img_data[yeo_index_1]
        mask = np.rint(Yeo_img_data[:,:,:])==i
        masked = Yeo_img_data.copy()
        masked[mask]=1
        masked[~mask] = 0
        Yeo_4.append(masked)
    c=np.stack(Yeo_4,axis=0)
    for i in range(1,8):
        m=0
        for j in ICA_index:
            correlation[i-1][m]=np.corrcoef(c[i-1,:,:,:].flatten(),ICA_data[:,:,:,j-1].flatten())[0][1]
            m+=1
    m=0
    for j in ICA_index: 
        correlation[7][m]=np.corrcoef(subcortical_data.flatten(),ICA_data[:,:,:,j-1].flatten())[0][1]
        print(correlation[7][m])
        correlation[8][m]=np.corrcoef(cerebellum_data_1.flatten(),ICA_data[:,:,:,j-1].flatten())[0][1]
        print(correlation[8][m])
        m+=1
    res=pd.DataFrame(correlation) 
    res.to_excel('/Users/zhuyuling/Desktop/mentor/correlation_100.xlsx')
    y_label=['Yeo_1','Yeo_2','Yeo_3','Yeo_4','Yeo_5','Yeo_6','Yeo_7','subcortical','cerebellum']
    sns_plot=sns.heatmap(correlation,xticklabels=ICA_index,yticklabels=y_label,annot=False)
    sns_plot.set_title('heatmap for correlation 21 out of 25 components')
    plt.xlabel('ICA')
 
    plt.show()
#######sum of value
'''
    ICA_index=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,
               34,35,36,37,38,39,40,41,42,43,45,46,48,49,50,52,53,57,58,60,63,64,93]
    arr_1=np.zeros(shape=(40000,3))
    
    for i in range(1,8):
        index_yeo=np.argwhere(np.rint(Yeo_img_data)==i)
        m=0
        for j in ICA_index:
            print(j)
            ICA_data_1=ICA_data[:,:,:,j-1]
            sum=0
            for k in range(0,len(index_yeo)):
                x,y,z=index_yeo[k,]
                sum+=ICA_data_1[x][y][z]
            correlation[i-1][m]=sum
            m+=1
    index_cortical=np.argwhere(np.rint(subcortical_data)==1)
    m=0
    for j in ICA_index:
        ICA_data_1=ICA_data[:,:,:,j-1]
        sum=0
        for k in range(0,len(index_cortical)):
            x,y,z=index_cortical[k,]
            sum+=ICA_data_1[x][y][z]
        correlation[7][m]=sum
        m+=1
    cerebellum_data_1=cerebellum_data[:,:,:,0]
    index_cerebellum=np.argwhere(np.rint(cerebellum_data_1)==1)
    print(index_cerebellum[1,])
    m=0
    for j in ICA_index:
        ICA_data_1=ICA_data[:,:,:,j-1]
        sum=0
        for k in range(0,len(index_cerebellum)):
            x,y,z=index_cerebellum[k,]
            sum+=ICA_data_1[x][y][z]
        correlation[8][m]=sum
        m+=1
    res=pd.DataFrame(correlation)   
    res.to_excel('/Users/zhuyuling/Desktop/mentor/sum_100.xlsx')
    y_label=['Yeo_1','Yeo_2','Yeo_3','Yeo_4','Yeo_5','Yeo_6','Yeo_7','subcortical','cerebellum']
 
    sns_plot=sns.heatmap(correlation,xticklabels=ICA_index,yticklabels=y_label,annot=False)
    sns_plot.set_title('heatmap for 100 ICA sum of value')
    plt.xlabel('ICA')
    #plt.ylabel('Yeo')
    plt.show()
    '''
    