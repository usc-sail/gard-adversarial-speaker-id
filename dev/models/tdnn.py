import torch
import torch.nn as nn
from dev.transforms import Preprocessor
import torch.nn.functional as F

class TDNN(nn.Module):
    def __init__(self, num_classes):
        super(TDNN, self).__init__()

        self.prep = Preprocessor()

        num_filters = 512
        stats_dim = 1500
        embed_dim = 512

        self.cnn_layers = nn.Sequential(
            # input normalization
            nn.BatchNorm1d(32),

            nn.Conv1d(32, num_filters, kernel_size=5, dilation=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            nn.Conv1d(num_filters, num_filters, kernel_size=5, dilation=2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            nn.Conv1d(num_filters, num_filters, kernel_size=7, dilation=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            nn.Conv1d(num_filters, num_filters, kernel_size=1, dilation=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),

            nn.Conv1d(num_filters, stats_dim, kernel_size=1, dilation=1),
            nn.BatchNorm1d(stats_dim),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(stats_dim*2, embed_dim)
        self.batchnorm_fc1 = nn.BatchNorm1d(embed_dim)

        self.fc2 = nn.Linear(embed_dim,embed_dim)
        self.batchnorm_fc2 = nn.BatchNorm1d(embed_dim)

        self.fc3 = nn.Linear(embed_dim,num_classes)

    def encode(self, x):
        x = self.prep(x.squeeze(1))
        x = self.cnn_layers(x)
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.fc1(stats) # "segment6" in x-vector paper
        return x

    def predict_from_embeddings(self, x):
        x = F.relu(self.batchnorm_fc1(x))
        x = F.relu(self.batchnorm_fc2(self.fc2(x))) # "segment7 in x-vector paper"
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




