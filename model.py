import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, width, height, use_cosine_similiarity=True):
        super(Model, self).__init__()
        
        self.conv_width = width // 4 - 3
        self.conv_height = height // 4 - 3
        self.d_conv = 64 * self.conv_width * self.conv_height
        
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.Conv2d(32,32,3),
            nn.MaxPool2d(2),
            nn.Dropout(inplace=True),
            nn.Conv2d(32,64,3),
            nn.Conv2d(64,64,3),
            nn.MaxPool2d(2)
        )

        self.siamese_feed_forward = nn.Sequential(
            nn.Linear(self.d_conv, 512),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.Sigmoid()
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(2*self.d_conv, 512),
            nn.Dropout(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.use_cosine_similiarity = use_cosine_similiarity
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    def forward_one(self, x):
        return torch.flatten(self.conv(x), start_dim=1)
    
    def forward(self, x1, x2):
        if self.use_cosine_similiarity:
            y1 = self.siamese_feed_forward(self.forward_one(x1))
            y2 = self.siamese_feed_forward(self.forward_one(x2))
            return self.cos(y1, y2)

        out = torch.cat((self.forward_one(x1), self.forward_one(x2)), dim=1)
        return self.feed_forward(out).squeeze()