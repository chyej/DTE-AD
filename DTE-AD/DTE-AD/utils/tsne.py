import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne(embeddings, gt, dataset, data_):
    #embeddings = embeddings.reshape(-1, num_feature)
    #embeddings = embeddings.reshape(-1, 1)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(embeddings)

    unique_labels = np.unique(gt)
    plt.figure(figsize=(10, 8))

    for label in unique_labels:
        plt.scatter(X_tsne[gt == label, 0], X_tsne[gt == label, 1], label=f'Class {label}')

    plt.legend()
    plt.savefig(f'utils/result_tsne/{data_}_{dataset}')
