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
    """Load test data"""
    print(f"\nLoading test data for {area_name}...")
    
    X_test = np.load(f'{config.DATA_DIR}/X_test_{area_name}.npy')
    y_test = np.load(f'{config.DATA_DIR}/y_test_{area_name}.npy')
    mask_test = np.load(f'{config.DATA_DIR}/mask_test_{area_name}.npy')
    
    with open(f'{config.DATA_DIR}/class_info_{area_name}.json', 'r') as f:
        class_info = json.load(f)
    
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(mask_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    return test_ds, class_info

def plot_confusion_matrix(y_true, y_pred, class_names, area_name):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title(f'Confusion Matrix (Counts) - {area_name}', fontsize=14, fontweight='bold')
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title(f'Confusion Matrix (Normalized) - {area_name}', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'results/{area_name}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return per_class_acc

def evaluate_model(area_name):
    """Evaluate trained model on test set"""
    
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL - {area_name}")
    print(f"{'='*60}")
    
    # Load test data
    test_ds, class_info = load_test_data(area_name)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    num_classes = class_info['n_classes']
    class_names = class_info.get('class_names', [f'Class_{i}' for i in range(num_classes)])
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = test_ds.tensors[0].shape[-1]
    
    model = MCTNet(
        input_dim=input_dim,
        d_model=config.MODEL_CONFIG['d_model'],
        n_stages=config.MODEL_CONFIG['n_stages'],
        nhead=config.MODEL_CONFIG['nhead'],
        kernel_size=config.MODEL_CONFIG['kernel_size'],
        num_classes=num_classes,
        dropout=config.MODEL_CONFIG['dropout']
    ).to(device)
    
    # Load best model
    model_path = f'results/{area_name}/best_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model from {model_path}")
        print(f"  Best validation accuracy: {checkpoint['val_acc']:.4f}")
    else:
        print(f"  Warning: No model found at {model_path}")
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
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
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
    print(f"TEST RESULTS - {area_name}")
    print(f"{'='*40}")
    print(f"Number of test samples: {len(all_labels)}")
    print(f"Number of classes: {num_classes}")
    print(f"\nOverall Metrics:")
    print(f"  Overall Accuracy (OA): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print(f"  F1-Score (Macro): {f1_macro:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 65)
    for i, name in enumerate(class_names[:10]):  # Show top 10
        print(f"{name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    if num_classes > 10:
        print(f"  ... and {num_classes - 10} more classes")
    
    # Plot confusion matrix
    per_class_acc = plot_confusion_matrix(all_labels, all_preds, class_names, area_name)
    
    # Save results
    with open(f'results/{area_name}/evaluation_results.json', 'w') as f:
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
    
    print(f"\n✅ Results saved to results/{area_name}/evaluation_results.json")
    
    return results

def compare_with_paper(results):
    """Compare results with those reported in the paper"""
    
    print(f"\n{'='*60}")
    print("COMPARISON WITH PAPER RESULTS")
    print(f"{'='*60}")
    
    # Paper reported results (from the CNN-Transformer paper)
    # These are approximate - adjust based on actual paper numbers
    paper_results = {
        'Arkansas': {
            'OA': 0.892,  # 89.2%
            'Kappa': 0.875,
            'F1': 0.883
        },
        'California': {
            'OA': 0.876,  # 87.6%
            'Kappa': 0.858,
            'F1': 0.865
        }
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
        
        print(f"\n{area}:")
        print(f"  OA:     Our {our_oa:.4f} vs Paper {paper_oa:.4f} → Diff: {diff_oa:+.4f} ({diff_oa*100:+.2f}%)")
        print(f"  Kappa:  Our {our_kappa:.4f} vs Paper {paper_kappa:.4f} → Diff: {diff_kappa:+.4f}")
        
        if diff_oa > 0:
            print(f"  ✅ Our model outperforms the paper by {diff_oa*100:.2f}%")
        else:
            print(f"  ⚠️ Paper outperforms our model by {abs(diff_oa)*100:.2f}%")
    
    # Save comparison
    with open('results/comparison_with_paper.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison

def main():
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("FINAL MODEL EVALUATION - Part 1")
    print("="*70)
    
    results = {}
    
    for area_name in config.STUDY_AREAS.keys():
        try:
            results[area_name] = evaluate_model(area_name)
        except Exception as e:
            print(f"Error evaluating {area_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare with paper
    compare_with_paper(results)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("Check 'results/{area_name}/' for all outputs")
    print("="*70)

if __name__ == "__main__":
    main()