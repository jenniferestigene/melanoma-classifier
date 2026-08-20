import numpy as np
import torch
from model import Net


img_size = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
net.load_state_dict(torch.load("saved_model.pth", map_location=device))
net.eval()


testing_data = np.load("melanoma_testing_data.npy", allow_pickle=True)

test_X = torch.Tensor(np.array([item[0] for item in testing_data]))
test_X = test_X / 255.0

test_y = torch.Tensor(np.array([item[1] for item in testing_data]))


correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_X)):

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