# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 00:13:21 2019

@author: hacker 1
"""
# import libs
import torch
import random

# define constants
lr = 1e-4
nEpoch = 1500
gpu = 2 #Which gpu should use
seed = 1
sampleSeqLength = 16 #length of sequence to train network
momentum = 0.9
samplingEpochs = 100 #how often to compute the CMC curve'

print('starting train')

#torch.cuda.set_device(gpu)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
random.seed(0)

seqRootRGB = 'iLIDS-VID//i-LIDS-VID//sequences//'
seqRootOF = 'iLIDS-VID//i-LIDS-VID-OF-HVP//sequences//'

print('loading Dataset - ',seqRootRGB,seqRootOF)

from DataPrepare import prepareDataset
dataset = prepareDataset(seqRootRGB, seqRootOF, '.png',False)

print('dataset loaded')

print('randomizing test/training split')

from DataUtils import partitionDataset,feature_extraction
trainInds,testInds = partitionDataset(len(dataset),0.5)

# load model
from model import *
import timeit
from DataUtils import getPosSample,getNegSample,prepare_person_seq


model = FCN32()
model.cuda()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
nTrainPersons = len(trainInds)

for epoch in range(1, nEpoch+1):
    batchError = 0
    order = torch.randperm(nTrainPersons)
    tic = timeit.default_timer()
    
    for i in range((2*nTrainPersons)):
        #positive pair
        if i%2 == 0:
            pushPull = 1
            camA = 0
            camB = 1
            
            startA, startB, seq_length = getPosSample(dataset, trainInds, order[i/2], sampleSeqLength)
            netInA = dataset[trainInds[order[i/2]]][camA][startA:(startA + seq_length),:,:,:].squeeze()
            netInB = dataset[trainInds[order[i/2]]][camB][startB:(startB + seq_length),:,:,:].squeeze()
            netTarget = [1, (order[i/2]), (order[i/2])]
            
        else:
            pushPull = -1
            seqA,seqB,camA,camB,startA,startB,seq_length = getNegSample(dataset,trainInds,sampleSeqLength)
            netInA = dataset[trainInds[seqA]][camA][startA:(startA + seq_length),:,:,:].squeeze()
            netInB = dataset[trainInds[seqB]][camB][startB:(startB + seq_length),:,:,:].squeeze()
            netTarget = [-1,seqA,seqB]
            
            
            




import cv2
import torchvision.transforms as transforms
def doDataAug(seq,cropx,cropy,flip):
    seqLen = seq.shape[0]
    seqChnls = seq.shape[1]
    seqDim1 = seq.shape[2]
    seqDim2 = seq.shape[3]
    
    cropx=int(cropx[0])
    cropy=int(cropy[0])
    flip=flip[0]
	
    daData = torch.zeros(seqLen,seqChnls,seqDim1-8,seqDim2-8)
	#to_pillow_image=transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    for i in range(seqLen): 
		#import ipdb;ipdb.set_trace()
		#do the data augmentation here
        thisFrame = seq[i,:,:,:].squeeze().clone()
        if flip == 1:
            img_np = thisFrame.numpy()
            img_np = cv2.flip(img_np,0)
            thisFrame=torch.from_numpy(img_np) #to_tensor(img_np)
			
        thisFrame = thisFrame[:, cropx: (56 + cropx), cropy:(40 + cropy)]
        thisFrame = thisFrame - torch.mean(thisFrame)
        daData[i,:,:,:] = thisFrame
	
    return daData	



















