
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from typing import List, Dict
from openset_config import KNOWN_ORIG_IDS, ID2NAME

def plot_learning_curves(history: Dict[str, List], save_dir: str):
    """
    Plot learning curves from training history
    """
    epochs = range(1, len(history['loss_total']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss_total'], label='Total Loss')
    if 'loss_adv' in history:
        plt.plot(epochs, history['loss_adv'], label='Adv Loss')
    if 'loss_boundary' in history:
        plt.plot(epochs, history['loss_boundary'], label='Boundary Loss')
    if 'loss_pseudo' in history:
        plt.plot(epochs, history['loss_pseudo'], label='Pseudo Loss')
        
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # 2. Metrics Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['known_acc'], label='Known Acc')
    plt.plot(epochs, history['open_f1'], label='Open F1')
    plt.plot(epochs, history['open_recall'], label='Open Recall')
    plt.plot(epochs, history['open_precision'], label='Open Precision')
    
    plt.title('Evaluation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_curve.png'))
    plt.close()

@torch.no_grad()
def plot_tsne(model, loader, device, save_dir, title="t-SNE Visualization"):
    """
    Compute and plot t-SNE of features
    """
    model.eval()
    all_feats = []
    all_labels = []
    
    print("Computing t-SNE...")
    
    for x, y, _, _ in loader:
        x = x.to(device)
        # Get features (using pos branch as standard)
        _, _, _, f_pos = model(x, None)
        f_pos = F.normalize(f_pos, dim=1)
        
        all_feats.append(f_pos.cpu().numpy())
        all_labels.append(y.numpy())
        
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    feats_2d = tsne.fit_transform(all_feats)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        'x': feats_2d[:, 0],
        'y': feats_2d[:, 1],
        'label': all_labels
    })
    
    # Separate Known and Unknown for better coloring
    # Known: >=0, Unknown: -1 or -2 etc (usually -1 in dataset)
    # But in test_loader, we might have original labels or mapped labels.
    # We should distinguish them.
    # Assuming y < 0 is unknown.
    
    df['type'] = df['label'].apply(lambda x: 'Known' if x >= 0 else 'Unknown')
    
    # Plot Known classes
    sns.scatterplot(
        data=df[df['type']=='Known'], 
        x='x', y='y', 
        hue='label', 
        palette='viridis',
        style='type',
        s=60,
        alpha=0.8,
        legend='full'
    )
    
    # Plot Unknown classes (if any)
    if len(df[df['type']=='Unknown']) > 0:
        sns.scatterplot(
            data=df[df['type']=='Unknown'], 
            x='x', y='y', 
            color='red',
            marker='X',
            label='Unknown',
            s=80,
            alpha=0.6
        )
        
    plt.title(title)
    # 优化图例位置，避免遮挡
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_plot.png'))
    plt.close()
