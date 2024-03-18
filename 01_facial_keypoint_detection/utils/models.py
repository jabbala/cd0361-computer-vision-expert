import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
# import torch.nn.init as I

class Net(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 36, kernel_size=5)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        # Define max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Define dropout layers
        self.dropout = nn.Dropout(p=0.2)
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 136)

    def forward(self, input_tensor):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Apply convolutional layers with ReLU activation and max-pooling
        input_tensor = self.pool(F.relu(self.conv1(input_tensor)))
        input_tensor = self.dropout(input_tensor)
        input_tensor = self.pool(F.relu(self.conv2(input_tensor)))
        input_tensor = self.dropout(input_tensor)
        input_tensor = self.pool(F.relu(self.conv3(input_tensor)))
        input_tensor = self.dropout(input_tensor)
        input_tensor = self.pool(F.relu(self.conv4(input_tensor)))
        input_tensor = self.dropout(input_tensor)
        input_tensor = self.pool(F.relu(self.conv5(input_tensor)))

        # Flatten the output for the fully connected layer
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        # Apply fully connected layer
        input_tensor = self.fc1(input_tensor)
        return input_tensor
