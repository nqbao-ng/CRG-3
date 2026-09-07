from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def plot_tsne(features, labels, dataset, file_name, save_dir='results/tnse'):

    os.makedirs(save_dir, exist_ok=True)

    if dataset == 'IEMOCAP':
        label_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    if dataset == 'MELD':
        label_names = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    for i, label_name in enumerate(label_names):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label_name, s=10, alpha=0.7)

    plt.legend()
    plt.title(f'{dataset} Dataset - t-SNE Visualization', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{file_name}.pdf')
    plt.savefig(save_path, format="pdf")
    plt.close()


def plot_confusion_matrix(matrix, dataset, file_name, save_dir='results/confusion_matrix'):
    
    os.makedirs(save_dir, exist_ok=True)

    if dataset == 'IEMOCAP':
        label_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    if dataset == 'MELD':
        label_names = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']

    row_sums = np.sum(matrix, axis=1, keepdims=True)
    matrix_percent = (matrix / (row_sums + 1e-9)) * 100 

    plt.figure(figsize=(8, 7))
    
    ax = sns.heatmap(
        matrix_percent, 
        annot=True, 
        fmt='.2f',
        cmap='Blues',
        cbar=True,
        linewidths=0.5,
        linecolor='lightgrey',
        xticklabels=label_names,
        yticklabels=label_names,
        square=True
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label('Prediction Accuracy (%)', rotation=90, labelpad=15)

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'{dataset} Dataset - Confusion Matrix', fontsize=14, fontweight='bold', y=1.03)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{file_name}.pdf')
    plt.savefig(save_path, format='pdf')
    plt.close()




