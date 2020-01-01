'''
利用卷积神经网络来预测每个网格的流入流出量
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#-------------------------------------------------------
targets = np.loadtxt('./data/targets.txt').reshape(1294, 175, 2)
closeness_features = np.loadtxt('./data/closeness_features.txt').reshape(1294, 175, 10)
period_features = np.loadtxt('./data/period_features.txt').reshape(1294, 175, 10)
trend_features = np.loadtxt('./data/trend_features.txt').reshape(1294, 175, 10)

targets = targets[:,:169,:].reshape(1294, 13, 13, 2)
closeness_features = closeness_features[:,:169,:].reshape(1294, 13, 13, 10)
period_features = period_features[:,:169,:].reshape(1294, 13, 13, 10)
trend_features = trend_features[:,:169,:].reshape(1294, 13, 13, 10)
#-------------------------------------------------------

test_num = 300

train_targets = targets[test_num:]
test_targets = targets[:test_num]

train_closeness_features = closeness_features[test_num:]
test_closeness_features = closeness_features[:test_num]

train_period_features = period_features[test_num:]
test_period_features = period_features[:test_num]

train_trend_features = trend_features[test_num:]
test_trend_features = trend_features[:test_num]
#-------------------------------------------------------
class myDataSet(Dataset):
    def __init__(self, closeness_features, period_features, trend_features, targets, feature_transform=None, target_transform=None):
        self.closeness_features = closeness_features
        self.period_features = period_features
        self.trend_features = trend_features
        self.targets = targets
        self.feature_transform = feature_transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.closeness_features)
    def __getitem__(self, idx):
        if self.feature_transform and self.target_transform:
            data = (self.feature_transform(self.closeness_features[idx]), 
                    self.feature_transform(self.period_features[idx]), 
                    self.feature_transform(self.trend_features[idx]), 
                    self.target_transform(self.targets[idx]))
        else:
            data = (self.closeness_features[idx], 
                    self.period_features[idx], 
                    self.trend_features[idx], 
                    self.targets[idx])
        return data

feature_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((2.1977,2.1978,2.1995,2.2033,2.2084,1.6551,1.6551,1.6562,1.6584,1.6619), 
                          (3.6767,3.6834,3.7022,3.7063,3.7053,2.8211,2.8195,2.8321,2.8340,2.8307))
    ])

target_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((2.2003,1.6571), 
                          (3.6455,2.8027))
    ])

trainset = myDataSet(train_closeness_features, 
    train_period_features, 
    train_trend_features, 
    train_targets, 
    feature_transform=feature_transform, 
    target_transform=target_transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = myDataSet(test_closeness_features, 
    test_period_features, 
    test_trend_features, 
    test_targets, 
    feature_transform=feature_transform, 
    target_transform=target_transform)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

#-------------------------------------------------------

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x  # shape: [batch, 2, 13, 13]

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.closeness_net = MyNet()
        self.period_net = MyNet()
        self.trend_net = MyNet()
        self.c = nn.Parameter(torch.FloatTensor([1]))
        self.p = nn.Parameter(torch.FloatTensor([1]))
        self.t = nn.Parameter(torch.FloatTensor([1]))
        self.bias = nn.Parameter(torch.FloatTensor([1]))
    def forward(self, cx, px, tx):
        out = self.c * self.closeness_net(cx) + self.p * self.period_net(px) + self.t * self.trend_net(tx) + self.bias
        return F.tanh(out)
   
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  
criteon = nn.MSELoss()


for epoch in range(1000):
    for idx, (cx_features, px_features, tx_features, y) in enumerate(trainloader):
        optimizer.zero_grad()
        out = model(cx_features.float(), px_features.float(), tx_features.float())
        loss = criteon(out, y.float())
        loss.backward()
        optimizer.step()

    print ('Epoch: %03d | loss: %.4f' % (epoch+1, loss))










