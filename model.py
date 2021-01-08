import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Model(nn.Module):
    def __init__(self, width, height, max_pooling=[1,(2,3),(2,3),(2,3)], mode=0):
        super(Model, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.MaxPool2d(max_pooling[0]),
            nn.Conv2d(32,32,3),
            nn.MaxPool2d(max_pooling[1]),
            nn.Dropout(inplace=True),
            nn.Conv2d(32,64,3),
            nn.MaxPool2d(max_pooling[2]),
            nn.Conv2d(64,64,3),
            nn.MaxPool2d(max_pooling[3])
        )
        self.conv.apply(init_weights)

        self.d_conv = torch.flatten(self.conv(torch.zeros(1,1,height,width))).size()[0]

        self.siamese_feed_forward = nn.Sequential(
            nn.Linear(self.d_conv, 512),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Sigmoid()
        )
        self.siamese_feed_forward.apply(init_weights)

        self.feed_forward = nn.Sequential(
            nn.Linear(2*self.d_conv, 512),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.feed_forward.apply(init_weights)

        self.mode = mode
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    def forward_one(self, x):
        return torch.flatten(self.conv(x), start_dim=1)
    
    def forward(self, x1, x2):
        if self.mode==0:
            y1 = self.forward_one(x1)
            y2 = self.forward_one(x2)
            return (self.cos(y1, y2) + 1)/2

        elif self.mode==1:
            y1 = self.siamese_feed_forward(self.forward_one(x1))
            y2 = self.siamese_feed_forward(self.forward_one(x2))
            return self.cos(y1, y2)

        else:
            out = torch.cat((self.forward_one(x1), self.forward_one(x2)), dim=1)
            return self.feed_forward(out).squeeze()