"""
train.py - Model training with early stopping and checkpointing
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.model import MCTNet

def load_data(area_name):
    """Load preprocessed data"""
    print(f"\nLoading data for {area_name}...")
    
    X_train = np.load(f'{config.DATA_DIR}/X_train_{area_name}.npy')
    X_val = np.load(f'{config.DATA_DIR}/X_val_{area_name}.npy')
    X_test = np.load(f'{config.DATA_DIR}/X_test_{area_name}.npy')
    
    y_train = np.load(f'{config.DATA_DIR}/y_train_{area_name}.npy')
    y_val = np.load(f'{config.DATA_DIR}/y_val_{area_name}.npy')
    y_test = np.load(f'{config.DATA_DIR}/y_test_{area_name}.npy')
    
    mask_train = np.load(f'{config.DATA_DIR}/mask_train_{area_name}.npy')
    mask_val = np.load(f'{config.DATA_DIR}/mask_val_{area_name}.npy')
    mask_test = np.load(f'{config.DATA_DIR}/mask_test_{area_name}.npy')
    
    # Load class info
    with open(f'{config.DATA_DIR}/class_info_{area_name}.json', 'r') as f:
        class_info = json.load(f)
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Classes: {class_info['n_classes']}")
    
    return (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test), class_info

def create_dataloaders(X_train, y_train, mask_train, X_val, y_val, mask_val, batch_size=32):
    """Create PyTorch dataloaders"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(mask_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    
    return train_loader, val_loader

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, m, y in loader:
        x, m, y = x.to(device), m.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x, m)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, m, y in loader:
            x, m, y = x.to(device), m.to(device), y.to(device)
            output = model(x, m)
            loss = criterion(output, y)
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

def plot_training_curves(history, area_name):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Loss Curves - {area_name}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Accuracy Curves - {area_name}', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(f'results/{area_name}', exist_ok=True)
    plt.savefig(f'results/{area_name}/training_curves.png', dpi=150)
    plt.close()

def train_area(area_name):
    """Train model for one area"""
    print(f"\n{'='*60}")
    print(f"Training MCTNet on {area_name}")
    print(f"{'='*60}")
    
    # Load data
    (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test), class_info = load_data(area_name)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, mask_train,
        X_val, y_val, mask_val,
        batch_size=config.BATCH_SIZE
    )
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    input_dim = X_train.shape[-1]
    num_classes = class_info['n_classes']
    
    model = MCTNet(
        input_dim=input_dim,
        d_model=config.MODEL_CONFIG['d_model'],
        n_stages=config.MODEL_CONFIG['n_stages'],
        nhead=config.MODEL_CONFIG['nhead'],
        kernel_size=config.MODEL_CONFIG['kernel_size'],
        num_classes=num_classes,
        dropout=config.MODEL_CONFIG['dropout']
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\n  Starting training...")
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping and checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_info': class_info
            }, f'results/{area_name}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Save training history
    with open(f'results/{area_name}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_curves(history, area_name)
    
    print(f"\n  ✅ Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    return model, history, best_val_acc

def main():
    """Main training function"""
    print("\n" + "="*70)
    print("MODEL TRAINING - Part 1")
    print("="*70)
    
    results = {}
    
    for area_name in config.STUDY_AREAS.keys():
        try:
            model, history, best_acc = train_area(area_name)
            results[area_name] = {'best_val_acc': best_acc, 'history': history}
        except Exception as e:
            print(f"Error training {area_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for area, res in results.items():
        print(f"  {area}: Best Val Acc = {res['best_val_acc']:.4f} ({res['best_val_acc']*100:.2f}%)")
    
    print("\n✅ Training complete! Check 'results/{area_name}/' for outputs.")

if __name__ == "__main__":
    # Create results directories
    for area_name in config.STUDY_AREAS.keys():
        os.makedirs(f'results/{area_name}', exist_ok=True)
    
    main()