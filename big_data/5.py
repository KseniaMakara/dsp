import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv('housing.csv')

if data.isnull().values.any():
    print("Увага! Деякі значення у даних відсутні. Рекомендується виконати їх обробку.")

data.drop('ocean_proximity', axis=1, inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

agg_clustering = AgglomerativeClustering(n_clusters=2)
agg_clusters = agg_clustering.fit_predict(scaled_data)

kmeans = KMeans(n_clusters=2)
kmeans_clusters = kmeans.fit_predict(scaled_data)

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_clusters = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans_clusters))

optimal_n_clusters = np.argmax(silhouette_scores) + 2 
print("Оптимальна кількість кластерів за коефіцієнтом силуета:", optimal_n_clusters)

agg_clusters_descr = pd.Series(agg_clusters).value_counts().sort_index()
kmeans_clusters_descr = pd.Series(kmeans_clusters).value_counts().sort_index()
print("Кількість об'єктів у кожному кластері (ієрархічний кластерний аналіз):")
print(agg_clusters_descr)
print("Кількість об'єктів у кожному кластері (k-Means кластерний аналіз):")
print(kmeans_clusters_descr)





