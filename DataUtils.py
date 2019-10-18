# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:33:20 2019

@author: hacker 1
"""
import torch

def partitionDataset(nPersons,split):
    split_point = int(nPersons * split)
    inds = torch.randperm(nPersons)
    trainInds = inds[0:split_point]
    testInds = inds[split_point:nPersons]

    print('N train = '+str(trainInds.shape[0]))
    print('N test  = '+str(testInds.shape[0]))

    return trainInds,testInds


import numpy as np
from torch.autograd import Variable

def feature_extraction(feature_net,dataset):
    nPersons = len(dataset)
    extracted_feature = np.array(np.zeros(nPersons).astype(int)).tolist()

    #set the parameters for data augmentation. Note that we apply the same augmentation
    #to the whole sequence, rather than individual images
    cropX = torch.floor(torch.rand(2).squeeze() * 8) + 1
    cropY = torch.floor(torch.rand(2).squeeze() * 8) + 1
    flip = torch.floor(torch.rand(2).squeeze() * 2) + 1

    letter = ['cam_a','cam_b']
    for i in range(nPersons):
        extracted_feature[i] = [0 for _ in range(len(letter))]
        for cam in range(len(letter)):
            data = doDataAug(dataset[i][cam], cropX[cam], cropY[cam], flip[cam])
            input_video=Variable(data,volatile=True)
            video_feature=feature_net(input_video.cuda())
            extracted_feature[i][cam]=video_feature
    return extracted_feature

import torchvision.transforms as transforms
from PIL import Image

def doDataAug(seq,cropx,cropy,flip):
    flip=int(flip)
    cropx=int(cropx)
    cropy=int(cropy)
    seqLen = seq.shape[0]
    seqChnls = seq.shape[1]
    seqDim1 = seq.shape[2]
    seqDim2 = seq.shape[3]
	
    Data = torch.zeros(seqLen,seqChnls,seqDim1-8,seqDim2-8)
    to_pil_image=transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    for i in range(seqLen): 
        currentFrame = seq[i,:,:,:].squeeze().clone()
        if flip == 1:
            pil_image=to_pil_image(currentFrame)
            flip_image=pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            currentFrame=to_tensor(flip_image)
        currentFrame = currentFrame[:, cropx: (56 + cropx), cropy:(40 + cropy)]
        currentFrame = currentFrame - torch.mean(currentFrame)
        Data[i,:,:,:] = currentFrame
	
    return Data



def getPosSample(dataset,trainInds,person,sampleSeqLen):
	# we have 2 cam
    camA = 0
    camB = 1
    actualSampleSeqLen = sampleSeqLen
    nSeqA = len(dataset[trainInds[person]][camA])
    nSeqB = len(dataset[trainInds[person]][camB])

	#what to do if the sequence is shorter than the sampleSeqLen 
    if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen:
        if nSeqA < nSeqB:
            actualSampleSeqLen = nSeqA
        else:
            actualSampleSeqLen = nSeqB

    startA = int(torch.rand(1)[0] * ((nSeqA - actualSampleSeqLen) + 1))
    startB = int(torch.rand(1)[0] * ((nSeqB - actualSampleSeqLen) + 1)) 

    return startA,startB,actualSampleSeqLen


def getNegSample(dataset,trainInds,sampleSeqLen):
    permAllPersons = torch.randperm(len(trainInds))
    personA = permAllPersons[0]
    personB = permAllPersons[1]
	#choose the camera, ilids video only has two, but change this for other datasets
    camA = 0
    camB = 1

    actualSampleSeqLen = sampleSeqLen
    nSeqA = len(dataset[trainInds[personA]][camA]) #len(dataset[trainInds[personA]][camA-1])
    nSeqB = len(dataset[trainInds[personB]][camB]) #len(dataset[trainInds[personB]][camB-1])

	#what to do if the sequence is shorter than the sampleSeqLen 
    if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen:
        if nSeqA < nSeqB:
            actualSampleSeqLen = nSeqA
        else:
            actualSampleSeqLen = nSeqB

    startA = int(torch.rand(1)[0] * ((nSeqA - actualSampleSeqLen) + 1)) #+ 1
    startB = int(torch.rand(1)[0] * ((nSeqB - actualSampleSeqLen) + 1)) #+ 1

    return personA,personB,camA,camB,startA,startB,actualSampleSeqLen


def prepare_person_seq(personImages,trainInds,order,cam,start,seq_length):
	#pdb.set_trace()
	person_seq_original=torch.FloatTensor(seq_length,128)
	for i in range(seq_length): 
		person_seq_original[i]=personImages[trainInds[order]][cam][start+i].data.cpu()
	person_seq=person_seq_original.transpose(0,1)
	person_seq=person_seq.unsqueeze(0)
	person_seq=person_seq.unsqueeze(2)
	return person_seq_original, person_seq