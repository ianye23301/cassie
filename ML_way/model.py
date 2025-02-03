import torch
import torch.nn as nn

# very simple network
class CarPriceNN(nn.Module):
    def __init__(self, input_dim):
        super(CarPriceNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)  # dropout to prevent overfitting

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)  
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)  
        return x