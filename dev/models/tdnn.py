import torch
import torch.nn as nn 
from dev.transforms import Preprocessor
import torch.nn.functional as F


class TDNN(nn.Module):

    def __init__(self, num_classes):
        super(TDNN, self).__init__()
        
        dropout = 0.0

        self.prep = Preprocessor()
        
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(32, 512, kernel_size=5, dilation=1),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),

            nn.Conv1d(512, 512, kernel_size=5, dilation=2),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),

            nn.Conv1d(512, 512, kernel_size=7, dilation=3),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),

            nn.Conv1d(512, 512, kernel_size=1, dilation=1),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),

            nn.Conv1d(512, 1500, kernel_size=1, dilation=1),
            nn.BatchNorm1d(1500),
            nn.Dropout(p=dropout)
        )

        self.fc1 = nn.Linear(3000,512)
        self.batchnorm_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(512,512)
        self.batchnorm_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(512,num_classes)

    def encode(self, x):
        x = self.prep(x.squeeze(1))

        x = self.cnn_layers(x)

        #if self.training:
        #    #this is for stability purpose only
        #    x += 0.001 * torch.rand_like(x)

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.batchnorm_fc1(F.relu(self.fc1(stats))))
        x = self.fc2(x)
        return x
    
    def predict_from_embeddings(self, x):
        x = self.dropout_fc2(self.batchnorm_fc2(F.relu(x)))
        x = self.fc3(x)
        return x

    def forward(self, x):
        """
        Inputs:
            x: [B, 1, T] waveform
        Outputs:
            x: [B, 1, T] waveform
        """
        x = self.encode(x)
        x = self.predict_from_embeddings(x)
        return x

