import numpy as np
import torch 
from model import Net 


# 100x100 pixels 
img_size = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
net.load_state_dict(torch.load("saved_model.pth", map_location=device))
net.eval()


testing_data = np.load("melanoma_testing_data.npy", allow_pickle=True)


# for row in testing_data:
#     print(row[0])
#     print(row[1])
#     print()
#     print()
#     input()


# Putting all the image arrays into this tensor
test_X = torch.Tensor(np.array([item[0] for item in testing_data]))
test_X = test_X / 255

# for row in test_X:
#     print(row)
#     print()
#     input()


# one-hot vector labels tensor
test_y = torch.Tensor( [item[1] for item in testing_data] )


correct = 0
total = 0

# PyTorch will automatically keep track of the gradients unless you tell it not to (could be wasting compute)
with torch.no_grad():
    # tells PyTorch to not automatically keep track of gradients

    for i in range(len(test_X)):

        # real label:
            # [0,1]
        # model guess (example):
            # [0.34, 0.66]

        output = net(test_X[i].view(-1, 3, img_size, img_size).to(device))[0]

        if output[0] >= output[1]:
            guess = "Benign"
        else:
            guess = "Malignant"

        real_label = test_y[i]

        if real_label[0] >= real_label[1]:
            real_class = "Benign"
        else:
            real_class = "Malignant"


        if guess == real_class:
            correct += 1

        total += 1


print(f"Accuracy: {round(correct/total,3)}")