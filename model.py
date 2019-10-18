# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:14:32 2019

@author: hacker 1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN8(nn.Module):
    
    def __init__(self, class_ = 1):
        super(FCN8, self).__init__()
        
        self.pad1 = nn.Zeropad2d(4)
        self.cnn_conv1 = nn.Conv2d(5, 16, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))
        
        self.pad2 = nn.Zeropad2d(4)
        self.cnn_conv2 = nn.Conv2d(16, 32, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))

        self.pad1 = nn.Zeropad2d(4)
        self.cnn_conv3 = nn.Conv2d(32, 32, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool3 = nn.MaxPool2d((2, 2), (2, 2))
        
        self.cnn_dropout = nn.Dropout2d(p = 0.6)
        self.fc = nn.Linear(32*10*8, 128)
        
        # convolution layer 1 
        self.conv11 = nn.Conv2d(128, 64, (1, 3), padding = (0, 100))
        self.bn11 = nn.BatchNorm2d(64)
        self.tanh = nn.Tanh()
        self.conv12 = nn.Conv2d(64, 64, (1, 3), padding = (0, 1))
        self.bn12 = nn.BatchNorm2d(64)
        self.tanh = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/2

        # convolution layer 2
        self.conv21 = nn.Conv2d(64, 128, (1, 3), padding = (0, 1))
        self.bn21 = nn.BatchNorm2d(128)
        self.tanh = nn.Tanh()
        self.conv22 = nn.Conv2d(128, 128, (1, 3), padding = (0, 1))
        self.bn22 = nn.BatchNorm2d(128)
        self.tanh = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/4

        # convolution layer 3
        self.conv31 = nn.Conv2d(128, 256, (1, 3), padding = (0, 1))
        self.bn31 = nn.BatchNorm2d(256)
        self.tanh = nn.Tanh()
        self.conv32 = nn.Conv2d(256, 256, (1, 3), padding = (0, 1))
        self.bn32 = nn.BatchNorm2d(256)
        self.tanh = nn.Tanh()
        self.conv33 = nn.Conv2d(256, 256, (1, 3), padding = (0, 1))
        self.bn33 = nn.BatchNorm2d(256)
        self.tanh = nn.Tanh()
        self.maxpool3 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/8

        # convolution layer 4
        self.conv41 = nn.Conv2d(256, 512, (1, 3), padding = (0, 1))
        self.bn41 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.conv42 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn42 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.conv43 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn43 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.maxpool4 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/16
        
        # convolution layer 5
        self.conv51 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn51 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.conv52 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn52 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.conv53 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn53 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.maxpool5 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/32
        
        # fc layer 6
        self.fc6 = nn.Conv2d(512, 4096, (1, 7))
        self.tanh = nn.Tanh()
        self.drop6 = nn.Dropout2d()
        
        # fc layer 7
        self.fc7 = nn.Conv2d(4096, 4096, (1, 1))
        self.tanh = nn.Tanh()
        self.drop7 = nn.Dropout2d()
        
        # score
        self.score1 = nn.Conv2d(4096, class_, (1, 1))
        self.score2 = nn.Conv2d(512, class_, (1, 1))
        self.score3 = nn.Conv2d(256, class_, (1, 1))
        
        self.upscore2 = nn.ConvTranspose2d(class_, class_, (1, 4), stride = (1, 2), bias = False)
        self.upscore8 = nn.ConvTranspose2d(class_, class_, (1, 16), stride = (1, 8), bias = False)
        self.upscore_pool4 = nn.ConvTranspose2d(class_, class_, (1, 4), stride = (1, 2), bias = False)
        
        self.sig = nn.Sigmoid()
        self.clfy = nn.Linear(128, 150)
        self.lsm = nn.LogSoftmax()

        
        
    def forward(self, x):
        cnn = self.pool1(self.tanh(self.cnn_conv1(self.pad1(x))))
        cnn = self.pool2(self.tanh(self.cnn_conv2(self.pad2(cnn))))
        cnn = self.pool3(self.tanh(self.cnn_conv3(self.pad3(cnn))))
        cnn = self.fc(self.cnn_dropout(cnn.view(-1, 32*10*8)))
        
        features = cnn.transpose(0, 1).unsqueeze(0).unsqueeze(2)
        
        hidden = self.tanh(self.bn11(self.conv11(features)))
        hidden = self.tanh(self.bn12(self.conv12(hidden)))
        hidden = self.maxpool1(hidden)
        
        hidden = self.tanh(self.bn21(self.conv21(hidden)))
        hidden = self.tanh(self.bn22(self.conv22(hidden)))
        hidden = self.maxpool2(hidden)
        
        hidden = self.tanh(self.bn31(self.conv31(hidden)))
        hidden = self.tanh(self.bn32(self.conv32(hidden)))
        hidden = self.tanh(self.bn33(self.conv33(hidden)))
        hidden = self.maxpool3(hidden)
        pool3 = hidden
        
        hidden = self.tanh(self.bn41(self.conv41(hidden)))
        hidden = self.tanh(self.bn42(self.conv42(hidden)))
        hidden = self.tanh(self.bn43(self.conv43(hidden)))
        hidden = self.maxpool4(hidden)
        pool4 = hidden
        
        hidden = self.tanh(self.bn51(self.conv51(hidden)))
        hidden = self.tanh(self.bn52(self.conv52(hidden)))
        hidden = self.tanh(self.bn53(self.conv53(hidden)))
        hidden = self.maxpool5(hidden)
        
        hidden = self.drop6(self.tanh(self.fc6(hidden)))
        
        hidden = self.drop7(self.tanh(self.fc7(hidden)))
        
        hidden = self.upscore2(self.score1(hidden))
        upscore2 = hidden #1/16
        
        hidden = self.score3(pool4)
        hidden = hidden[:, :, :, 5:5 + upscore2.size()[3]]
        score_pool4c = hidden #1/16
        
        hidden = self.upscore_pool4(upscore2 + score_pool4c)
        upscore_pool4 = hidden #1/8
        
        hidden = self.score3(pool3)
        hidden = hidden[:, :, :, 9:9 + upscore_pool4.size()[3]]
        score_pool3c = hidden  # 1/8
        
        hidden = upscore_pool4 + score_pool3c  # 1/8
        hidden = self.upscore8(hidden)
        hidden = hidden[:, :, :, 31:31 + features.size()[3]].contiguous()
        
        att_score=self.sig(hidden.squeeze(0).squeeze(0).transpose(0,1))
        
        cnn_trans = cnn.transpose(0, 1)
        
        att_feature = torch.mm(cnn_trans, att_score).transpose(0, 1)
        
        combined = F.normalize(torch.mean(torch.cat((cnn, att_feature), 0), 0).unsqueeze(0), p = 2, dim = 1)
        
        classified = self.clfy(combined)
        lsmax = self.lsm(classified)
        
        return combined, lsmax


class FCN16(nn.Module):
    
    def __init__(self, class_ = 1):
        super(FCN16, self).__init__()
        
        self.pad1 = nn.Zeropad2d(4)
        self.cnn_conv1 = nn.Conv2d(5, 16, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))
        
        self.pad2 = nn.Zeropad2d(4)
        self.cnn_conv2 = nn.Conv2d(16, 32, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))

        self.pad1 = nn.Zeropad2d(4)
        self.cnn_conv3 = nn.Conv2d(32, 32, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool3 = nn.MaxPool2d((2, 2), (2, 2))
        
        self.cnn_dropout = nn.Dropout2d(p = 0.6)
        self.fc = nn.Linear(32*10*8, 128)
        
        # convolution layer 1 
        self.conv11 = nn.Conv2d(128, 64, (1, 3), padding = (0, 100))
        self.bn11 = nn.BatchNorm2d(64)
        self.tanh = nn.Tanh()
        self.conv12 = nn.Conv2d(64, 64, (1, 3), padding = (0, 1))
        self.bn12 = nn.BatchNorm2d(64)
        self.tanh = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/2

        # convolution layer 2
        self.conv21 = nn.Conv2d(64, 128, (1, 3), padding = (0, 1))
        self.bn21 = nn.BatchNorm2d(128)
        self.tanh = nn.Tanh()
        self.conv22 = nn.Conv2d(128, 128, (1, 3), padding = (0, 1))
        self.bn22 = nn.BatchNorm2d(128)
        self.tanh = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/4

        # convolution layer 3
        self.conv31 = nn.Conv2d(128, 256, (1, 3), padding = (0, 1))
        self.bn31 = nn.BatchNorm2d(256)
        self.tanh = nn.Tanh()
        self.conv32 = nn.Conv2d(256, 256, (1, 3), padding = (0, 1))
        self.bn32 = nn.BatchNorm2d(256)
        self.tanh = nn.Tanh()
        self.conv33 = nn.Conv2d(256, 256, (1, 3), padding = (0, 1))
        self.bn33 = nn.BatchNorm2d(256)
        self.tanh = nn.Tanh()
        self.maxpool3 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/8

        # convolution layer 4
        self.conv41 = nn.Conv2d(256, 512, (1, 3), padding = (0, 1))
        self.bn41 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.conv42 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn42 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.conv43 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn43 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.maxpool4 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/16
        
        # convolution layer 5
        self.conv51 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn51 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.conv52 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn52 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.conv53 = nn.Conv2d(512, 512, (1, 3), padding = (0, 1))
        self.bn53 = nn.BatchNorm2d(512)
        self.tanh = nn.Tanh()
        self.maxpool5 = nn.MaxPool2d((1, 2), stride = (1, 2), ceil_mode = True)     #1/32
        
        # fc layer 6
        self.fc6 = nn.Conv2d(512, 4096, (1, 7))
        self.tanh = nn.Tanh()
        self.drop6 = nn.Dropout2d()
        
        # fc layer 7
        self.fc7 = nn.Conv2d(4096, 4096, (1, 1))
        self.tanh = nn.Tanh()
        self.drop7 = nn.Dropout2d()
        
        self.score_fr = nn.Conv2d(4096, class_, (1, 1))
        self.score_fr_bn = nn.BatchNorm2d(class_)
        self.score_pool4 = nn.Conv2d(512, class_, (1, 1))
        self.score_pool4_bn = nn.BatchNorm2d(class_)
        
        self.upscore2 = nn.ConvTranspose2d(class_, class_, (1, 4), stride = (1, 2), bias = False)
        self.upscore16 = nn.ConvTranspose2d(class_, class_, (1, 32), stride = (1, 16), bias = False)
        
        self.sig = nn.Sigmoid()
        self.clfy = nn.Linear(128, 150)
        self.lsm = nn.LogSoftmax()
        
    def forward(self, x):
        
        cnn = self.pool1(self.tanh(self.cnn_conv1(self.pad1(x))))
        cnn = self.pool2(self.tanh(self.cnn_conv2(self.pad2(cnn))))
        cnn = self.pool3(self.tanh(self.cnn_conv3(self.pad3(cnn))))
        cnn = self.fc(self.cnn_dropout(cnn.view(-1, 32*10*8)))
        
        features = cnn.transpose(0, 1).unsqueeze(0).unsqueeze(2)
        
        hidden = self.tanh(self.bn11(self.conv11(features)))
        hidden = self.tanh(self.bn12(self.conv12(hidden)))
        hidden = self.maxpool1(hidden)
        
        hidden = self.tanh(self.bn21(self.conv21(hidden)))
        hidden = self.tanh(self.bn22(self.conv22(hidden)))
        hidden = self.maxpool2(hidden)
        
        hidden = self.tanh(self.bn31(self.conv31(hidden)))
        hidden = self.tanh(self.bn32(self.conv32(hidden)))
        hidden = self.tanh(self.bn33(self.conv33(hidden)))
        hidden = self.maxpool3(hidden)
        
        hidden = self.tanh(self.bn41(self.conv41(hidden)))
        hidden = self.tanh(self.bn42(self.conv42(hidden)))
        hidden = self.tanh(self.bn43(self.conv43(hidden)))
        hidden = self.maxpool4(hidden)
        pool4 = hidden
        
        hidden = self.tanh(self.bn51(self.conv51(hidden)))
        hidden = self.tanh(self.bn52(self.conv52(hidden)))
        hidden = self.tanh(self.bn53(self.conv53(hidden)))
        hidden = self.maxpool5(hidden)
        
        hidden = self.drop6(self.tanh(self.fc6(hidden)))
        
        hidden = self.drop7(self.tanh(self.fc7(hidden)))
        
        upscore2 = self.upscore2(self.score_fr(hidden))
        
        score_pool4c = self.score_pool4(pool4)[:, :, :, 5:5 + upscore2.size()[3]]
        
        hidden = self.upscore16(upscore2 + score_pool4c)[:, :, :, 27:27 + features.size()[3]].contiguous()
        
        att_score = self.sig(hidden.squeeze(0).squeeze(0).transpose(0,1))
        
        cnn_trans = cnn.transpose(0, 1)
        
        att_feature = torch.mm(cnn_trans, att_score).transpose(0, 1)
        
        combined = F.normalize(torch.mean(torch.cat((cnn, att_feature), 0), 0).unsqueeze(0), p = 2, dim = 1)
        
        classified = self.clfy(combined)
        lsmax = self.lsm(classified)
        
        return combined, lsmax
        
class FCN32(nn.Module):
    
    def __init__(self, class_ = 1):
        super(FCN32, self).__init__()
        
        self.pad1 = nn.ZeroPad2d(4)
        self.cnn_conv1 = nn.Conv2d(5, 16, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))
        
        self.pad2 = nn.ZeroPad2d(4)
        self.cnn_conv2 = nn.Conv2d(16, 32, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))

        self.pad1 = nn.ZeroPad2d(4)
        self.cnn_conv3 = nn.Conv2d(32, 32, (5, 5), (1, 1))
        self.tanh = nn.Tanh()
        self.pool3 = nn.MaxPool2d((2, 2), (2, 2))
        
        self.sp_conv_att = nn.Conv2d(32, 3, (5, 5), padding = (2, 2))
        self.sp_conv = nn.Conv2d(32, 32, (5, 5), padding = (2, 2))
        
        self.cnn_dropout = nn.Dropout2d(p = 0.6)
        self.fc = nn.Linear(32*10*8, 128)
        
        self.fc6 = nn.Conv2d(128, 1, (1, 1))
        self.drop = nn.Dropout2d()
        
        self.sig = nn.Sigmoid()
        self.clfy = nn.Linear(128, 150)
        self.lsm = nn.LogSoftmax()
        
    def forward(self, x):
        
        cnn = self.pool1(self.tanh(self.cnn_conv1(self.pad1(x))))
        cnn = self.pool2(self.tanh(self.cnn_conv2(self.pad2(cnn))))
        cnn = self.pool3(self.tanh(self.cnn_conv3(self.pad3(cnn))))

        sp_cnn = cnn
        
        sp_att = self.sp_conv_att(sp_cnn)
        
        sp_att1=sp_att[:,0].unsqueeze(1)
        sp_att2=sp_att[:,1].unsqueeze(1)
        sp_att3=sp_att[:,2].unsqueeze(1)
        
        sp_sig1 = self.sig(sp_att1)
        sp_sig2 = self.sig(sp_att2)
        sp_sig3 = self.sig(sp_att3)
    
        cnn = self.fc(self.cnn_dropout(cnn.view(-1, 32*10*8)))
         
        features = cnn.tranpose(0, 1).unsqueeze(0).unsqueeze(2) #(1L, 128L, 1L, 16L)
        
        hidden = self.drop(self.fc6(features))
        
        att_score = self.sig(hidden).squeeze(0).squeeze(1).transpose(0,1)
        
        cnn_trans = cnn.transpose(0, 1)
        
        att_score = torch.mm(cnn_trans, att_score)
        att_score = att_score.transpose(0, 1)
        
        spatial_att1 = torch.mul(sp_cnn, sp_sig1).view(-1, 32*10*8)
        spatial_att2 = torch.mul(sp_cnn, sp_sig2).view(-1, 32*10*8)
        spatial_att3 = torch.mul(sp_cnn, sp_sig3).view(-1, 32*10*8)
        
        spatial_att1 = self.fc(self.cnn_dropout(spatial_att1))
        spatial_att2 = self.fc(self.cnn_dropout(spatial_att2))
        spatial_att3 = self.fc(self.cnn_dropout(spatial_att3))
        
        comb_feature1 = torch.add(spatial_att1, att_score)
        comb_feature2 = torch.add(spatial_att2, att_score)
        comb_feature3 = torch.add(spatial_att3, att_score)
        
        comb_feature1 = torch.cat((cnn, comb_feature1), 0)
        comb_feature2 = torch.cat((cnn, comb_feature2), 0)
        comb_feature3 = torch.cat((cnn, comb_feature3), 0)
        
        comb_feature1 = torch.mean(comb_feature1,0).unsqueeze(0)
        comb_feature2 = torch.mean(comb_feature2,0).unsqueeze(0)
        comb_feature3 = torch.mean(comb_feature3,0).unsqueeze(0)
        
        comb_feature1 = F.normalize(comb_feature1, p=2, dim=1)
        comb_feature2 = F.normalize(comb_feature2, p=2, dim=1)
        comb_feature3 = F.normalize(comb_feature3, p=2, dim=1)
        
        comb_features = torch.add(comb_feature1, comb_feature2)
        comb_features = torch.add(comb_features, comb_feature3)
        
        classified = self.clfy(comb_features)
        lsmax = self.lsm(classified)
        
        return comb_features, lsmax