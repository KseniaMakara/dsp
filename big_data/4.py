import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import Isomap, TSNE
import umap

data = pd.read_csv('test.csv')

selector = VarianceThreshold()
selector.fit_transform(data)
low_variance_cols = data.columns[~selector.get_support()]
data = data.drop(columns=low_variance_cols)

correlation_matrix = data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Кореляційна матриця')
plt.show()

data = data.drop(columns=['sc_w', 'pc'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])

plt.scatter(pca_df['PCA1'], pca_df['PCA2'], alpha=0.5)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA')
plt.show()

isomap = Isomap(n_neighbors=5, n_components=2)
isomap_result = isomap.fit_transform(scaled_data)
plt.scatter(isomap_result[:, 0], isomap_result[:, 1], alpha=0.5)
plt.title('IsoMap')
plt.show()

# Візуалізація за допомогою t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
tsne_result = tsne.fit_transform(scaled_data)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
plt.title('t-SNE')
plt.show()

# Візуалізація за допомогою UMAP
umap_result = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(scaled_data)
plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.5)
plt.title('UMAP')
plt.show()





