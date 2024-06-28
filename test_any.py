import torch
import torch.nn as nn
import torch.nn.functional as F


import torch

# # Assuming your data is in a 3D numpy array of shape (num_samples, num_time_steps, num_features)
# num_samples = 100
# num_time_steps = 50
# num_features = 3
# data = torch.randn(num_samples, num_time_steps, num_features)
#
# # Reshape the data to (N, Cin, H, W)
# data = data.permute(0, 2, 1).unsqueeze(-1)
# # data now has shape (100, 3, 50, 1)
#
# # Now you can feed this data into a Conv2D layer
# conv_layer = torch.nn.Conv2d(in_channels=num_features, out_channels=16, kernel_size=(3, 1))
# output = conv_layer(data)
# print(output)


# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2, 1))
#         self.dropout1 = nn.Dropout(0.1)
#
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 1))
#         self.dropout2 = nn.Dropout(0.2)
#
#         self.flatten = nn.Flatten()
#
#         # Calculate the flattened dimension after the convolutional layers
#         # The height will reduce from 80 to 79 due to the kernel size of 2 and stride of 1.
#         # The width will reduce from 3 to 2 for the same reason.
#         self.fc1_input_dim = 64 * (80 - 2 + 1) * (3 - 2 + 1)  # 64 * 79 * 2
#         self.fc1 = nn.Linear(self.fc1_input_dim, 128)
#         self.dropout3 = nn.Dropout(0.5)
#
#         self.fc2 = nn.Linear(128, 6)
#
#     def forward(self, x):
#         # x = x.unsqueeze(-1)
#         x = F.relu(self.conv1(x))
#         x = self.dropout1(x)
#
#         x = F.relu(self.conv2(x))
#         x = self.dropout2(x)
#
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout3(x)
#
#         x = F.softmax(self.fc2(x), dim=1)
#
#         return x
#
#
# # Create the model
# model = CNNModel()
#
# # Example input
# x_train = torch.randn(425, 3, 80)  # Original input shape
# # x_train = x_train.permute(0, 2, 1)  # Change shape to (425, 3, 80)
#
# # Forward pass
# output = model(x_train)
# print(output.shape)


rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output)