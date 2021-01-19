import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Model(nn.Module):
    def __init__(self, width, height, mode=0, conv_filters=[32,32,64,64], max_pooling=(2,3),kernel=3):
        super(Model, self).__init__()

        conv_filters.insert(0,1)
        n = len(conv_filters)
        i, prev_filter_size, conv_layers = 1, 1, []
        while(i<n):
            conv_layers.append(nn.Conv2d(conv_filters[i-1],conv_filters[i],kernel))
            conv_layers.append(nn.ReLU(inplace=True))
            i+=1
            if(i%2==1 or i==n):
                conv_layers.append(nn.MaxPool2d(max_pooling))


        self.conv = nn.Sequential(*conv_layers)
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