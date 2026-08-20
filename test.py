import numpy as np
import torch
import matplotlib.pyplot as plt
import os
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

# Confusion matrix counts
# true_positive: actually malignant, predicted malignant
# true_negative: actually benign, predicted benign
# false_positive: actually benign, predicted malignant
# false_negative: actually malignant, predicted benign  <- the dangerous error
tp = tn = fp = fn = 0

with torch.no_grad():
    for i in range(len(test_X)):

        output = net(test_X[i].view(-1, 3, img_size, img_size).to(device))[0]

        guess = "Benign" if output[0] >= output[1] else "Malignant"

        real_label = test_y[i]
        real_class = "Benign" if real_label[0] >= real_label[1] else "Malignant"

        if guess == real_class:
            correct += 1
        total += 1

        if real_class == "Malignant" and guess == "Malignant":
            tp += 1
        elif real_class == "Benign" and guess == "Benign":
            tn += 1
        elif real_class == "Benign" and guess == "Malignant":
            fp += 1
        elif real_class == "Malignant" and guess == "Benign":
            fn += 1


accuracy = correct / total

recall_malignant = tp / (tp + fn) if (tp + fn) > 0 else 0.0
precision_malignant = tp / (tp + fp) if (tp + fp) > 0 else 0.0
specificity_benign = tn / (tn + fp) if (tn + fp) > 0 else 0.0

print(f"Accuracy: {round(accuracy, 3)}")
print(f"Malignant recall (sensitivity): {round(recall_malignant, 3)}")
print(f"Malignant precision: {round(precision_malignant, 3)}")
print(f"Benign specificity: {round(specificity_benign, 3)}")
print()
print(f"True positives (malignant correctly caught): {tp}")
print(f"False negatives (malignant missed, called benign): {fn}")
print(f"True negatives (benign correctly cleared): {tn}")
print(f"False positives (benign flagged as malignant): {fp}")

with open("test_accuracy.txt", "w") as f:
    f.write(str(accuracy))

os.makedirs("assets", exist_ok=True)

matrix = np.array([[tn, fp],
                    [fn, tp]])
labels = ["Benign", "Malignant"]

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(matrix, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")

for i in range(2):
    for j in range(2):
        color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                color=color, fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=200, bbox_inches="tight")
print("\nSaved assets/confusion_matrix.png")