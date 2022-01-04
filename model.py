from torch import nn
import torch.nn.functional as F

#Just starting with vanilla model
class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        #We should use convolutional layers because they are known to be translational invariant
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # fully connected layers
        self.fc1 = nn.Linear(320, 50)
        self.output = nn.Linear(50, 10)


    def forward(self, x):
        # Convolutional and maxpoolinf
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)
