"""
train.py - Model training with early stopping and checkpointing (AMÉLIORÉ)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.model import MCTNet

def load_data(area_name):
    """Load preprocessed data"""
    print(f"\n📂 Loading data for {area_name}...")
    
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
    
    print(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"    X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"    Classes: {class_info['n_classes']}")
    
    # Afficher distribution des classes
    train_dist = Counter(y_train)
    print(f"    Distribution train: {dict(sorted(train_dist.items())[:5])}...")
    
    return (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test), class_info

def create_dataloaders(X_train, y_train, mask_train, X_val, y_val, mask_val, batch_size=32):
    """Create PyTorch dataloaders with class weighting"""
    
    # Calculer les poids pour équilibrer les classes
    class_counts = Counter(y_train)
    num_classes = len(class_counts)
    
    # Poids par classe (inverse de la fréquence)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in y_train]
    
    # Convertir en tensor
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    
    # Créer sampler pondéré
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,  # Utiliser sampler au lieu de shuffle
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, class_counts

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, m, y in tqdm(loader, desc="      Training", leave=False):
        x, m, y = x.to(device), m.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x, m)
        loss = criterion(output, y)
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, balanced_acc

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, m, y in tqdm(loader, desc="      Validation", leave=False):
            x, m, y = x.to(device), m.to(device), y.to(device)
            output = model(x, m)
            loss = criterion(output, y)
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, balanced_acc

def plot_training_curves(history, area_name, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title(f'Loss Curves - {area_name}', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title(f'Accuracy Curves - {area_name}', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Balanced Accuracy
    axes[1, 0].plot(history['train_balanced_acc'], label='Train Balanced Acc', linewidth=2)
    axes[1, 0].plot(history['val_balanced_acc'], label='Val Balanced Acc', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Balanced Accuracy', fontsize=12)
    axes[1, 0].set_title(f'Balanced Accuracy Curves - {area_name}', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], label='Learning Rate', linewidth=2, color='green')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title(f'Learning Rate Schedule - {area_name}', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=150)
    plt.close()

def train_area(area_name):
    """Train model for one area"""
    # Créer le dossier de sauvegarde
    save_dir = f'results/train/{area_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 Training MCTNet on {area_name}")
    print(f"   Results will be saved to: {save_dir}")
    print(f"{'='*60}")
    
    # Load data
    (X_train, y_train, mask_train), (X_val, y_val, mask_val), (X_test, y_test, mask_test), class_info = load_data(area_name)
    
    # Create dataloaders with class weighting
    train_loader, val_loader, class_counts = create_dataloaders(
        X_train, y_train, mask_train,
        X_val, y_val, mask_val,
        batch_size=config.BATCH_SIZE
    )
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  💻 Device: {device}")
    
    input_dim = X_train.shape[-1]
    num_classes = class_info['n_classes']
    
    # Modèle avec plus de capacités
    model = MCTNet(
        input_dim=input_dim,
        d_model=128,  # Augmenté de 64 à 128
        n_stages=4,   # Augmenté de 3 à 4
        nhead=8,      # Augmenté de 4 à 8
        kernel_size=5, # Augmenté de 3 à 5
        num_classes=num_classes,
        dropout=0.2    # Augmenté pour régularisation
    ).to(device)
    
    print(f"  📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  📚 Number of classes: {num_classes}")
    print(f"  ⚖️ Class distribution: {dict(sorted(class_counts.items())[:5])}...")
    
    # Loss with class weights for imbalance
    class_weights = torch.tensor(
        [1.0 / class_counts.get(i, 1) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer avec weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_val_balanced_acc = 0
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_balanced_acc': [],
        'val_loss': [], 'val_acc': [], 'val_balanced_acc': [],
        'lr': []
    }
    
    print("\n  🏋️ Starting training...")
    print(f"  Epochs: {config.EPOCHS} | Batch size: {config.BATCH_SIZE} | Patience: {config.PATIENCE}")
    print(f"  d_model: 128 | n_stages: 4 | nhead: 8 | kernel_size: 5")
    print("-" * 70)
    
    for epoch in range(config.EPOCHS):
        # Train
        train_loss, train_acc, train_balanced_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_balanced_acc = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_balanced_acc'].append(train_balanced_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_balanced_acc'].append(val_balanced_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        # Print progress
        print(f"    Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} (Bal: {train_balanced_acc:.4f}) | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} (Bal: {val_balanced_acc:.4f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping and checkpoint (utiliser balanced accuracy)
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_balanced_acc': val_balanced_acc,
                'class_info': class_info,
                'class_counts': dict(class_counts)
            }, f'{save_dir}/best_model.pth')
            print(f"      ✅ New best model! (Balanced Acc: {val_balanced_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"      ⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Save training history
    with open(f'{save_dir}/training_history.json', 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        history_serializable = {k: convert(v) for k, v in history.items()}
        json.dump(history_serializable, f, indent=2)
    
    plot_training_curves(history, area_name, save_dir)
    
    # Sauvegarder aussi les données de test pour l'évaluation
    np.save(f'{save_dir}/X_test.npy', X_test)
    np.save(f'{save_dir}/y_test.npy', y_test)
    np.save(f'{save_dir}/mask_test.npy', mask_test)
    
    print(f"\n  ✅ Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  ✅ Best balanced accuracy: {best_val_balanced_acc:.4f} ({best_val_balanced_acc*100:.2f}%)")
    print(f"  📁 Results saved to: {save_dir}/")
    
    return model, history, best_val_acc, best_val_balanced_acc

def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🎯 MODEL TRAINING - Part 1 (Version Améliorée)")
    print("   Architecture: MCTNet (CNN-Transformer)")
    print("   Optimisations: Class weighting, Balanced accuracy, Cosine annealing")
    print("="*70)
    
    results = {}
    
    for area_name in config.STUDY_AREAS.keys():
        try:
            model, history, best_acc, best_balanced_acc = train_area(area_name)
            results[area_name] = {
                'best_val_acc': best_acc,
                'best_val_balanced_acc': best_balanced_acc,
                'history': history
            }
        except Exception as e:
            print(f"\n❌ Error training {area_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    for area, res in results.items():
        print(f"  📍 {area}:")
        print(f"      Standard Accuracy: {res['best_val_acc']:.4f} ({res['best_val_acc']*100:.2f}%)")
        print(f"      Balanced Accuracy: {res['best_val_balanced_acc']:.4f} ({res['best_val_balanced_acc']*100:.2f}%)")
    
    print("\n✅ Training complete! Check 'results/train/{area_name}/' for outputs.")
    print("   - best_model.pth : Best model weights")
    print("   - training_curves.png : Loss and accuracy curves (4 subplots)")
    print("   - training_history.json : Detailed training history")
    print("   - X_test.npy, y_test.npy, mask_test.npy : Test data for evaluation")
    print("="*70)

if __name__ == "__main__":
    # Create results directories
    for area_name in config.STUDY_AREAS.keys():
        os.makedirs(f'results/train/{area_name}', exist_ok=True)
    
    main()