import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import sys
sys.path.append('..')
from scripts.model import MCTNet

def load_data(area_name):
    X_train = np.load(f'X_train_{area_name}.npy')
    y_train = np.load(f'y_train_{area_name}.npy')
    mask_train = np.load(f'mask_train_{area_name}.npy')
    # idem pour val et test
    return (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test)

def main():
    area_name = 'Arkansas'  # ou 'California'
    (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test) = load_data(area_name)

    # Conversion en tenseurs
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    mask_train_t = torch.tensor(mask_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    # idem val, test

    # Paramètres
    num_classes = len(np.unique(y_train))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MCTNet(input_dim=10, d_model=64, n_stages=3, nhead=5, kernel_size=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_t, mask_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, mask_val_t, y_val_t), batch_size=batch_size)

    best_val_acc = 0.0
    for epoch in range(200):
        # Entraînement
        model.train()
        train_loss = 0.0
        for x, m, y in train_loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, m)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, m, y in val_loader:
                x, m, y = x.to(device), m.to(device), y.to(device)
                out = model(x, m)
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        val_acc = correct / total
        scheduler.step(val_acc)
        print(f'Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{area_name}.pth')

    # Test
    model.load_state_dict(torch.load(f'best_model_{area_name}.pth'))
    test_loader = DataLoader(TensorDataset(X_test_t, mask_test_t, y_test_t), batch_size=batch_size)
    preds, trues = [], []
    with torch.no_grad():
        for x, m, y in test_loader:
            x, m = x.to(device), m.to(device)
            out = model(x, m)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            trues.extend(y.numpy())
    oa = accuracy_score(trues, preds)
    kappa = cohen_kappa_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    print(f'Test: OA={oa:.4f}, Kappa={kappa:.4f}, F1={f1:.4f}')

if __name__ == '__main__':
    main()