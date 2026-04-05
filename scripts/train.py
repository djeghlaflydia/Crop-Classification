import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def load(area):
    return (
        np.load(f"X_train_{area}.npy"),
        np.load(f"y_train_{area}.npy"),
        np.load(f"X_val_{area}.npy"),
        np.load(f"y_val_{area}.npy"),
        np.load(f"X_test_{area}.npy"),
        np.load(f"y_test_{area}.npy"),
    )

class SimpleCNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def main():
    area = "California"
    Xtr, ytr, Xv, yv, Xt, yt = load(area)

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)

    Xv = torch.tensor(Xv, dtype=torch.float32)
    yv = torch.tensor(yv, dtype=torch.long)

    Xt = torch.tensor(Xt, dtype=torch.float32)
    yt = torch.tensor(yt, dtype=torch.long)

    model = SimpleCNN(Xtr.shape[1], len(np.unique(ytr)))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=32)

    train_accs = []
    val_accs = []

    for epoch in range(50):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        # accuracy train
        model.eval()
        pred_tr = model(Xtr).argmax(1)
        pred_v = model(Xv).argmax(1)

        acc_tr = accuracy_score(ytr, pred_tr)
        acc_v = accuracy_score(yv, pred_v)

        train_accs.append(acc_tr)
        val_accs.append(acc_v)

        print(f"Epoch {epoch+1} | train={acc_tr:.3f} val={acc_v:.3f}")

    # 📈 GRAPH
    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="val")
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    # 🧪 TEST
    pred_test = model(Xt).argmax(1)
    acc = accuracy_score(yt, pred_test)

    print("Test accuracy:", acc)

    # 📊 CONFUSION MATRIX
    cm = confusion_matrix(yt, pred_test)
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    main()