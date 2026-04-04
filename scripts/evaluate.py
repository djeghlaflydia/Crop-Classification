"""
evaluate.py - Final model evaluation and comparison with paper
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             confusion_matrix, classification_report,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.model import MCTNet

def load_test_data(area_name):
    """Load test data from train directory"""
    print(f"\n📂 Loading test data for {area_name}...")
    
    # Charger depuis results/train/{area}/
    train_dir = f'results/train/{area_name}'
    
    X_test = np.load(f'{train_dir}/X_test.npy')
    y_test = np.load(f'{train_dir}/y_test.npy')
    mask_test = np.load(f'{train_dir}/mask_test.npy')
    
    # Load class info from data directory
    with open(f'{config.DATA_DIR}/class_info_{area_name}.json', 'r') as f:
        class_info = json.load(f)
    
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(mask_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    print(f"    Test samples: {len(X_test)}")
    print(f"    Classes: {class_info['n_classes']}")
    
    return test_ds, class_info

def plot_confusion_matrix(y_true, y_pred, class_names, area_name, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title(f'Confusion Matrix (Counts) - {area_name}', fontsize=14, fontweight='bold')
    axes[0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(class_names, rotation=0, fontsize=8)
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title(f'Confusion Matrix (Normalized) - {area_name}', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(class_names, rotation=0, fontsize=8)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm.diagonal() / cm.sum(axis=1)

def plot_per_class_metrics(precision, recall, f1, class_names, area_name, save_dir, top_k=20):
    """Plot per-class metrics (top-k classes)"""
    n_classes = len(class_names)
    display_n = min(top_k, n_classes)
    
    metrics_df = pd.DataFrame({
        'Class': class_names[:display_n],
        'Precision': precision[:display_n],
        'Recall': recall[:display_n],
        'F1': f1[:display_n]
    })
    
    fig, ax = plt.subplots(figsize=(14, max(6, display_n * 0.3)))
    
    x = np.arange(display_n)
    width = 0.25
    
    bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', color='coral')
    bars3 = ax.bar(x + width, metrics_df['F1'], width, label='F1-Score', color='green')
    
    ax.set_xlabel('Crop Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Performance (Top {display_n}) - {area_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Class'], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Top-5 Classes by F1-Score:")
    top5 = metrics_df.nlargest(5, 'F1')[['Class', 'Precision', 'Recall', 'F1']]
    for _, row in top5.iterrows():
        print(f"    {row['Class']}: P={row['Precision']:.3f}, R={row['Recall']:.3f}, F1={row['F1']:.3f}")

def evaluate_model(area_name):
    """Evaluate trained model on test set"""
    
    # Définir les dossiers de sauvegarde
    train_dir = f'results/train/{area_name}'
    save_dir = f'results/evaluate/{area_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📊 EVALUATING MODEL - {area_name}")
    print(f"   Results saved to: {save_dir}")
    print(f"{'='*60}")
    
    # Load test data
    test_ds, class_info = load_test_data(area_name)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    num_classes = class_info['n_classes']
    class_names = [str(c) for c in class_info['classes']]
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = test_ds.tensors[0].shape[-1]
    
    model = MCTNet(
        input_dim=input_dim,
        d_model=128,
        n_stages=4,
        nhead=8,
        kernel_size=5,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)
    
    # Load best model
    model_path = f'{train_dir}/best_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ Loaded model from {model_path}")
        print(f"     Best validation accuracy: {checkpoint.get('val_acc', 0):.4f}")
        print(f"     Best balanced accuracy: {checkpoint.get('val_balanced_acc', 0):.4f}")
    else:
        print(f"  ❌ No model found at {model_path}")
        return None
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, m, y in test_loader:
            x, m = x.to(device), m.to(device)
            output = model(x, m)
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    results = {
        'area': area_name,
        'accuracy': accuracy,
        'kappa': kappa,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'per_class': {
            'class': class_names,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'n_samples': len(all_labels),
        'n_classes': num_classes
    }
    
    # Print results
    print(f"\n{'='*40}")
    print(f"📈 TEST RESULTS - {area_name}")
    print(f"{'='*40}")
    print(f"  Test samples: {len(all_labels)}")
    print(f"  Number of classes: {num_classes}")
    print(f"\n  Overall Metrics:")
    print(f"    Overall Accuracy (OA): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Cohen's Kappa: {kappa:.4f}")
    print(f"    F1-Score (Macro): {f1_macro:.4f}")
    print(f"    F1-Score (Weighted): {f1_weighted:.4f}")
    
    # Plot confusion matrix
    per_class_acc = plot_confusion_matrix(all_labels, all_preds, class_names, area_name, save_dir)
    print(f"  ✅ Saved: {save_dir}/confusion_matrix.png")
    
    # Plot per-class metrics
    plot_per_class_metrics(precision, recall, f1, class_names, area_name, save_dir, top_k=20)
    print(f"  ✅ Saved: {save_dir}/per_class_metrics.png")
    
    # Save results
    with open(f'{save_dir}/evaluation_results.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_serializable = {k: convert(v) for k, v in results.items()}
        json.dump(results_serializable, f, indent=2)
    
    print(f"  ✅ Saved: {save_dir}/evaluation_results.json")
    
    # Also save a summary text file
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(f"Evaluation Results - {area_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Test samples: {len(all_labels)}\n")
        f.write(f"Number of classes: {num_classes}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Overall Accuracy (OA): {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"  Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"  F1-Score (Macro): {f1_macro:.4f}\n")
        f.write(f"  F1-Score (Weighted): {f1_weighted:.4f}\n")
    
    print(f"  ✅ Saved: {save_dir}/summary.txt")
    
    return results

def compare_with_paper(results):
    """Compare results with those reported in the paper"""
    
    save_dir = 'results/evaluate'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📚 COMPARISON WITH PAPER RESULTS")
    print(f"{'='*60}")
    
    # Paper reported results (from the CNN-Transformer paper)
    paper_results = {
        'Arkansas': {'OA': 0.892, 'Kappa': 0.875, 'F1': 0.883},
        'California': {'OA': 0.876, 'Kappa': 0.858, 'F1': 0.865}
    }
    
    comparison = []
    
    for area, res in results.items():
        if area not in paper_results:
            continue
        
        our_oa = res['accuracy']
        paper_oa = paper_results[area]['OA']
        diff_oa = our_oa - paper_oa
        
        our_kappa = res['kappa']
        paper_kappa = paper_results[area]['Kappa']
        diff_kappa = our_kappa - paper_kappa
        
        comparison.append({
            'area': area,
            'our_OA': our_oa,
            'paper_OA': paper_oa,
            'diff_OA': diff_oa,
            'our_Kappa': our_kappa,
            'paper_Kappa': paper_kappa,
            'diff_Kappa': diff_kappa
        })
        
        print(f"\n  {area}:")
        print(f"    OA:     Our {our_oa:.4f} vs Paper {paper_oa:.4f} → Diff: {diff_oa:+.4f} ({diff_oa*100:+.2f}%)")
        print(f"    Kappa:  Our {our_kappa:.4f} vs Paper {paper_kappa:.4f} → Diff: {diff_kappa:+.4f}")
        
        if diff_oa > 0:
            print(f"    ✅ Our model outperforms the paper by {diff_oa*100:.2f}%")
        else:
            print(f"    ⚠️ Paper outperforms our model by {abs(diff_oa)*100:.2f}%")
    
    # Save comparison
    with open(f'{save_dir}/comparison_with_paper.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  ✅ Saved: {save_dir}/comparison_with_paper.json")
    
    # Save comparison as text
    with open(f'{save_dir}/comparison_summary.txt', 'w') as f:
        f.write("Comparison with Paper Results\n")
        f.write("="*50 + "\n\n")
        for comp in comparison:
            f.write(f"{comp['area']}:\n")
            f.write(f"  Our OA: {comp['our_OA']:.4f} ({comp['our_OA']*100:.2f}%)\n")
            f.write(f"  Paper OA: {comp['paper_OA']:.4f} ({comp['paper_OA']*100:.2f}%)\n")
            f.write(f"  Difference: {comp['diff_OA']:+.4f} ({comp['diff_OA']*100:+.2f}%)\n\n")
    
    print(f"  ✅ Saved: {save_dir}/comparison_summary.txt")
    
    return comparison

def main():
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("🔍 FINAL MODEL EVALUATION - Part 1")
    print("   Results will be saved to: results/evaluate/")
    print("="*70)
    
    results = {}
    
    for area_name in config.STUDY_AREAS.keys():
        try:
            results[area_name] = evaluate_model(area_name)
        except Exception as e:
            print(f"\n❌ Error evaluating {area_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare with paper
    compare_with_paper(results)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("📁 Results saved to:")
    print("   - results/evaluate/{area_name}/confusion_matrix.png")
    print("   - results/evaluate/{area_name}/per_class_metrics.png")
    print("   - results/evaluate/{area_name}/evaluation_results.json")
    print("   - results/evaluate/{area_name}/summary.txt")
    print("   - results/evaluate/comparison_with_paper.json")
    print("   - results/evaluate/comparison_summary.txt")
    print("="*70)

if __name__ == "__main__":
    main()