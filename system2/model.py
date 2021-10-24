import torch
from torch import nn


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.bn: nn.BatchNorm1d = nn.BatchNorm1d(774)
        self.lstm: nn.LSTM = nn.LSTM(774, 1024, bidirectional=True)

        self.fc1: nn.Sequential = nn.Sequential(
            nn.BatchNorm1d(2059),
            nn.Linear(2059, 256),
            nn.ReLU(inplace=True)
        )

        self.fc2: nn.Sequential = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )

        self.fc3: nn.Sequential = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64, 4)
        )

    def forward(self, user_data: torch.FloatTensor,
                tweet_data: torch.FloatTensor) -> torch.FloatTensor:
        tweet_data = self.bn.forward(tweet_data.transpose(1, 2)).transpose(1, 2)
        tweet_data = self.lstm.forward(tweet_data.transpose(0, 1))[0][-1]
        user_data = torch.cat((user_data, tweet_data), dim=1)
        user_data = self.fc1.forward(user_data)
        user_data = self.fc2.forward(user_data)
        user_data = self.fc3.forward(user_data)
        return user_data
