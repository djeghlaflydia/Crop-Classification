import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model import MCTNet

def load_and_prepare_data(area_name):
    print(f"Loading files for {area_name}...")
    X_train = np.load(f'X_train_{area_name}.npy')
    y_train = np.load(f'y_train_{area_name}.npy')
    mask_train = np.load(f'mask_train_{area_name}.npy')
    X_val = np.load(f'X_val_{area_name}.npy')
    y_val = np.load(f'y_val_{area_name}.npy')
    mask_val = np.load(f'mask_val_{area_name}.npy')
    X_test = np.load(f'X_test_{area_name}.npy')
    y_test = np.load(f'y_test_{area_name}.npy')
    mask_test = np.load(f'mask_test_{area_name}.npy')

    return (
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(mask_train, dtype=torch.float32).unsqueeze(-1), torch.tensor(y_train, dtype=torch.long)),
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(mask_val, dtype=torch.float32).unsqueeze(-1), torch.tensor(y_val, dtype=torch.long)),
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(mask_test, dtype=torch.float32).unsqueeze(-1), torch.tensor(y_test, dtype=torch.long))
    )

def train_area(area_name):
    print(f"\n=== Training on {area_name} ===")
    try:
        train_ds, val_ds, test_ds = load_and_prepare_data(area_name)
    except Exception as e:
        print(f"Error loading {area_name}: {e}")
        return

    num_classes = int(torch.cat([train_ds.tensors[2], val_ds.tensors[2], test_ds.tensors[2]]).max().item() + 1)
    input_dim = train_ds.tensors[0].shape[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MCTNet(input_dim=input_dim, d_model=64, n_stages=2, nhead=4, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    best_val_acc = 0.0
    for epoch in range(30): # Reduced epochs for rapid Part 1 verification
        model.train()
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad(); loss = criterion(model(x, m), y); loss.backward(); optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, m, y in val_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                pred = model(x, m).argmax(dim=1)
                total += y.size(0); correct += (pred == y).sum().item()
        
        val_acc = correct / total if total > 0 else 0
        scheduler.step(val_acc)
        if (epoch+1) % 5 == 0: print(f"Epoch {epoch+1}/30 - Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{area_name}.pth')

    # Test
    model.load_state_dict(torch.load(f'best_model_{area_name}.pth'))
    test_loader = DataLoader(test_ds, batch_size=32)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for x, m, y in test_loader:
            out = model(x.to(device), m.to(device))
            preds.extend(out.argmax(dim=1).cpu().numpy()); trues.extend(y.numpy())
    
    print(f"Results for {area_name}: OA={accuracy_score(trues, preds):.4f} | Kappa={cohen_kappa_score(trues, preds):.4f}")

def main():
    for area in ['Arkansas', 'California']:
        train_area(area)

if __name__ == '__main__':
    main()