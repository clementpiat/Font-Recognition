import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, width, height, use_cosine_similiarity=True):
        super(Model, self).__init__()
        
        self.conv_width = width // 4 - 3
        self.conv_height = height // 4 - 3
        self.d_conv = 64 * self.conv_width * self.conv_height
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.Conv2d(32,32,3),
            nn.MaxPool2d(2)
        )
        self.conv_dropout = nn.Dropout(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3),
            nn.Conv2d(64,64,3),
            nn.MaxPool2d(2)
        )
        
        self.feed_forward1 = nn.Linear(2*self.d_conv, 512)
        self.feed_forward_dropout = nn.Dropout(inplace=True)
        self.feed_forward2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.use_cosine_similiarity = use_cosine_similiarity
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.use_dropout = True

    def skip_dropout(self):
        self.use_dropout = False

    def forward_one(self, x):
        x1 = self.conv1(x)
        if self.use_dropout:
            x1 = self.conv_dropout(x1)

        return torch.flatten(self.conv2(x1), start_dim=1)
    
    def forward(self, x1, x2):
        if self.use_cosine_similiarity:
            return (self.cos(self.forward_one(x1), self.forward_one(x2)) + 1)/2

        out = torch.cat((self.forward_one(x1), self.forward_one(x2)), dim=1)
        out1 = self.feed_forward1(out)
        if self.use_dropout:
            out1 = self.feed_forward_dropout(out1)

        return self.feed_forward2(out1).squeeze()