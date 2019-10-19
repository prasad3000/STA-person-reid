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
from torch.autograd import Variable
import torch.nn as nn
import math


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
            
            startA, startB, seq_length = getPosSample(dataset, trainInds, order[int(i/2)], sampleSeqLength)
            netInA = dataset[trainInds[order[int(i/2)]]][camA][startA:(startA + seq_length),:,:,:].squeeze()
            netInB = dataset[trainInds[order[int(i/2)]]][camB][startB:(startB + seq_length),:,:,:].squeeze()
            netTarget = [1, (order[int(i/2)]), (order[int(i/2)])]
            
        else:
            pushPull = -1
            seqA,seqB,camA,camB,startA,startB,seq_length = getNegSample(dataset,trainInds,sampleSeqLength)
            netInA = dataset[trainInds[seqA]][camA][startA:(startA + seq_length),:,:,:].squeeze()
            netInB = dataset[trainInds[seqB]][camB][startB:(startB + seq_length),:,:,:].squeeze()
            netTarget = [-1,seqA,seqB]
        
        cropxA = torch.floor(torch.rand(1).squeeze() * 8) + 1
        cropyA = torch.floor(torch.rand(1).squeeze() * 8) + 1
        cropxB = torch.floor(torch.rand(1).squeeze() * 8) + 1
        cropyB = torch.floor(torch.rand(1).squeeze() * 8) + 1
        
        flipA = torch.floor(torch.rand(1).squeeze() * 2) + 1
        flipB = torch.floor(torch.rand(1).squeeze() * 2) + 1
        
        netInA = doDataAug(netInA, cropxA, cropyA, flipA)
        netInB = doDataAug(netInB, cropxB, cropyB, flipB)
        
        netInA = Variable(netInA).cuda()
        netInB = Variable(netInB).cuda()
        
        optimizer.zero_grad()
        
        # losses
        hingeLoss = nn.HingeEmbeddingLoss()
        criterion1 = nn.NLLLoss()
        criterion2 = nn.NLLLoss()
        
        # pass the data to model
        comb_features1, lsmax1 = model(netInA)
        comb_features2, lsmax2 = model(netInB)
        
        # calculate 12 pairwise distance
        distance = nn.PairwiseDistance(p = 2)
        target1 = Variable(torch.Tensor([netTarget[0]]).cuda()).type(torch.cuda.LongTensor)
        target2 = Variable(torch.Tensor([netTarget[1]]).cuda()).type(torch.cuda.LongTensor)
        target3 = Variable(torch.Tensor([netTarget[2]]).cuda()).type(torch.cuda.LongTensor)
        
        dist = distance(comb_features1, comb_features2)
        loss = hingeLoss(dist, target1) + criterion1(lsmax1, target2) + criterion2(lsmax2, target3)
        
        batchError = batchError + loss
        
        loss.backward()
        optimizer.step()
        
    
        
    toc = timeit.default_timer()
    print(epoch)
    
    if epoch == 1300:
        lr = lr/10
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
    if (epoch % 100 == 0):
        print('__________________________________________________________________')
        print(epoch)
        '''cmcTest, simMatTest = computeCMC(dataset, testInds, model, sampleSeqLength)
        cmcTrain, simMatTrain = computeCMC(dataset, trainInds, model, sampleSeqLength)
        
        strTest = 'Test'
        strTrain = 'Train'
        
        inds = [0,1,2,3,4,5,6,7,8,9,10]
        
        for c in range(len(inds)):
            if c< nTrainPersons:
                sTest = strTest + str(int(math.floor(cmcTest[inds[c]]))) + '   '
                sTrain = strTrain + str(int(math.floor(cmcTrain[inds[c]]))) + '   '
    
                print(sTest)
                print(sTrain)'''
                
        model.train()
        
            
torch.save(model, 'trained_model/model.pt')            
            




import cv2
import torchvision.transforms as transforms
def doDataAug(seq,cropx,cropy,flip):
    seqLen = seq.shape[0]
    seqChnls = seq.shape[1]
    seqDim1 = seq.shape[2]
    seqDim2 = seq.shape[3]
    
    cropx=int(cropx)
    cropy=int(cropy)
    flip=flip
	
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



















