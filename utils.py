import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch

def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    features, labels = [], []

    for img_batch, label_batch in tqdm(loader, desc="Extracting features"):
        batch_feat = img_batch.view(img_batch.size(0), -1)
        features.append(batch_feat)
        labels.append(label_batch)

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    print(features.shape)
    return features, labels

# def choose_n_components(X, threshold=0.95):
#     pca = PCA().fit(X)
#     cumulative_var = np.cumsum(pca.explained_variance_ratio_)
#     n_components = np.argmax(cumulative_var >= threshold) + 1

#     return n_components
