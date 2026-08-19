import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Net


img_size = 100

training_data = np.load("melanoma_training_data.npy", allow_pickle=True)

train_X = torch.Tensor(np.array([item[0] for item in training_data]))

# ImageNet normalization: 
train_X = train_X / 255.0

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

train_y = torch.Tensor(np.array([item[1] for item in training_data]))

# Validation set
val_size = int(0.1 * len(train_X))
val_X, val_y = train_X[:val_size], train_y[:val_size]
train_X, train_y = train_X[val_size:], train_y[val_size:]

print(f"Training on {len(train_X)} images, validating on {len(val_X)} images")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
print(f"Using device: {device}")

# Confirm only the final layer's parameters are trainable
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in net.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

mean = mean.to(device)
std = std.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001) 
loss_function = nn.CrossEntropyLoss()

batch_size = 32  # smaller batch size - ResNet18 uses more memory than the small CNN
epochs = 10       # fewer epochs needed - pretrained weights converge faster


for epoch in range(epochs):
    epoch_loss = 0
    net.train()

    for i in range(0, len(train_X), batch_size):

        print(f"EPOCH {epoch+1}, fraction complete: {i/len(train_X)}")

        batch_X = train_X[i: i+batch_size].view(-1, 3, img_size, img_size)
        batch_y = torch.argmax(train_y[i:i+batch_size], dim=1)

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        batch_X = (batch_X - mean) / std  # apply ImageNet normalization

        optimizer.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (len(train_X) / batch_size)

    # Validation accuracy check after every epoch
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(val_X)):
            real_class = torch.argmax(val_y[i])
            img = val_X[i].view(-1, 3, img_size, img_size).to(device)
            img = (img - mean) / std
            output = net(img)[0]
            predicted_class = torch.argmax(output)

            if predicted_class.cpu() == real_class:
                correct += 1
            total += 1

    val_accuracy = correct / total
    print(f"Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.4f} - val accuracy: {val_accuracy:.4f}")


torch.save(net.state_dict(), "saved_model.pth")
print("Model saved to saved_model.pth")