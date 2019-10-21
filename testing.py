nPerson = len(Inds)

similarityMatrix = torrch.zero(nPerson, nPerson)

for shift in range(7):
    for flip in range(1):
        shift_y = shift
        
        camAFeatures = torch.DoubleTensor(nPerson, 128)
        
        for person in range(nPerson):
            seqLen = dataset[Inds[person]][0].shape[0]
            
            seq=dataset[Inds[person]][0][0:seqLen,:,:].squeeze().clone()
            
            # Augmentation
            augSeq = torch.zeros(seqLen, 5, 56, 40)
            
            for k in range(seqLen):
                temp = seq[k,:,:,:].squeeze().clone()
                
                if flip == 1:
                    temp = torch.from_numpy(cv2.flip(temp.numpy(), 0))
                    
                temp = temp[:, shift: (56 + shift), shift_y:(40 + shift_y)]
                temp = temp - torch.mean(temp)
                augSeq[k,:,:,:] = temp.cuda().clone()
                
            test_seq = Variable(augSeq).cuda()
            output_features_camA, classifierA = model(test_seq)
            
            camAFeatures[person,:]=output_features_camA.data.squeeze(0)
            
        camBFeatures = torch.DoubleTensor(nPerson, 128)
        
        for person in range(nPerson):
            seqLen = dataset[Inds[person]][0].shape[0]
            
            seq=dataset[Inds[person]][0][0:seqLen,:,:].squeeze().clone()
            
            # Augmentation
            augSeq = torch.zeros(seqLen, 5, 56, 40)
            
            for k in range(seqLen):
                temp = seq[k,:,:,:].squeeze().clone()
                
                if flip == 1:
                    temp = torch.from_numpy(cv2.flip(temp.numpy(), 0))
                    
                temp = temp[:, shift: (56 + shift), shift_y:(40 + shift_y)]
                temp = temp - torch.mean(temp)
                augSeq[k,:,:,:] = temp.cuda().clone()
                
            test_seq = Variable(augSeq).cuda()
            output_features_camB, classifierB = model(test_seq)
            
            camBFeatures[person,:]=output_features_camB.data.squeeze(0)
            
            
        for i in range(nPerson):
            for j in range(nPerson):
                fa = camAFeatures[i, :]
                fb = camBFeatures[j, :]
                
                distance = math.sqrt(torch.sum(torch.pow(fa - fb,2)))
                similarityMatrix[i][j] = similarityMatrix[i][j] + distance
