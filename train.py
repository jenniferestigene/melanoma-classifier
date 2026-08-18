import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Net



# 100x100 pixels
img_size = 100


training_data = np.load("melanoma_training_data.npy", allow_pickle=True)


# for row in training_data:
#     print(row[0])
#     print(row[1])
#     print()
#     print()
#     input()


# Putting all the image arrays into this tensor
train_X = torch.Tensor(np.array([item[0] for item in training_data]))
train_X = train_X / 255

# for row in train_X:
#     print(row)
#     print()
#     input()


# one-hot vector labels tensor
train_y = torch.Tensor( [item[1] for item in training_data] )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
print(f"Using device: {device}")


optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_function = nn.CrossEntropyLoss()


# how many images you're passing through at once
batch_size = 100


# how many times we are passing through the training data
epochs = 20


for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(train_X), batch_size):

        print(f"EPOCH {epoch+1}, fraction complete: {i/len(train_X)}")

        batch_X = train_X[i: i+batch_size].view(-1, 3, img_size, img_size)
        batch_y = torch.argmax(train_y[i:i+batch_size], dim=1)

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # resets gradients of model parameters to zero before this pass
        optimizer.zero_grad()
        outputs = net(batch_X)
        # calculates the loss between the predicted outputs and the actual image one-hot vector labels
        loss = loss_function(outputs, batch_y)

        # real label:
                # [0,1]
        # model guess (example):
                # [0.34, 0.66]


        # backpropagation calculates the gradients of the loss with respect to model parameters
        loss.backward()
        # optimizer updates the model parameters based on the gradient we just calculated
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - avg loss: {epoch_loss/(len(train_X)/batch_size):.4f}")



torch.save(net.state_dict(), "saved_model.pth")