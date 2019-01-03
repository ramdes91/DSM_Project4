
# coding: utf-8

# In[1]:


##Import libraries##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


##Input data##

df = pd.read_csv('data_stocks.csv')
df1 = df.copy()
print(df.shape)
df.head()


# In[3]:


##Feature scaling##

from sklearn.preprocessing import StandardScaler
features = df.values
sc = StandardScaler()
X_scaled = sc.fit_transform(features)
print('Shape of Scaled features : ')
print(X_scaled.shape)


# In[4]:


#Determining optimal number of components for PCA looking at the explained variance 
#as a function of the components

sns.set()
sns.set_style('whitegrid')
pca = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# Here we see that we'd need about 100 components to retain 100% of the variance. Looking at this plot for a high-dimensional dataset can help us understand the level of redundancy present in multiple observations

# In[5]:


##Convert to 2Dimensions using PCA##

pca = PCA(n_components=2)
pca.fit(X_scaled)
print('explained variance :')
print(pca.explained_variance_)
print('PCA Components : ')
print(pca.components_)
X_transformed = pca.transform(X_scaled)
print('Transformed Feature values first five rows :')
print(X_transformed[:5,:])
print('Transformed Feature shape :')
print(X_transformed.shape)
print('Original Feature shape :')
print(X_scaled.shape)
print('Restransformed Feature shape :')
X_retransformed = pca.inverse_transform(X_transformed)
print(X_retransformed.shape)
print('Retransformed Feature values first five rows :')
print(X_retransformed[:5,:])


# Problem 1:
# There are various stocks for which we have collected a data set, which all stocks are apparently similar in performance

# In[6]:


##Number of clusters for KMeans##

wcss=[]
for i in range(1, 21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X_transformed)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Mean Squared Errors')
plt.show()


# In[7]:


#Optimum number of cluster from the elbow method is determined to be 5
#Applying K-Means Clustering to find stocks which are similar in performance

k_means = KMeans(n_clusters=5,random_state=0,init='k-means++')
k_means.fit(X_transformed)
y_kmeans = kmeans.fit_predict(X_transformed)
labels = k_means.labels_
print("labels generated :\n",labels)
print('label lengths: \n',len(labels))


# In[8]:


##Visualization##

plt.scatter(X_transformed[y_kmeans == 0, 0], X_transformed[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_transformed[y_kmeans == 1, 0], X_transformed[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_transformed[y_kmeans == 2, 0], X_transformed[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_transformed[y_kmeans == 3, 0], X_transformed[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_transformed[y_kmeans == 4, 0], X_transformed[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of stocks')
plt.xlabel('Principal Component (1)')
plt.ylabel('Principal Component (2)')
plt.legend()
plt.show()


# The above 5 clusters shows the stocks which are similar in stock performance

# Problem 2:
# How many Unique patterns that exist in the historical stock data set, based on fluctuations in price.

# In[9]:


df_comp = pd.DataFrame(pca.components_,columns=df1.columns)
df_comp.head()


# In[10]:


sns.set_style('whitegrid')
sns.heatmap(df_comp)


# Problem 3:
# Identify which all stocks are moving together and which all stocks are different from each other.

# In[11]:


df['labels'] = labels
df.head()


# In[12]:


df['labels'].unique().tolist()
for i in df['labels'].unique().tolist():
    count = df[df['labels'] == i].shape[0]
    print('\nFor lablel {} the number of similar stock performances is : {} '.format(i,count))


# In[13]:


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import SpectralClustering
hc = SpectralClustering(n_clusters = 5, affinity = 'nearest_neighbors')
hc.fit(X_transformed)


# In[14]:


hc.fit_predict(X_transformed)


# In[15]:


y_labels = hc.labels_
len(y_labels),np.unique(y_labels)


# In[16]:


# Visualising the clusters
X = X_transformed
plt.scatter(X[y_labels == 0, 0], X[y_labels == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_labels == 1, 0], X[y_labels == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_labels == 2, 0], X[y_labels == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_labels == 3, 0], X[y_labels == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_labels == 4, 0], X[y_labels == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[17]:


df1.columns
df2 = df1.copy()
df2['labels'] = y_labels
for i in df2['labels'].unique().tolist():
    count = df2[df2['labels'] == i].shape[0]
    print('\nFor lablel {} the number of similar stock performances is : {} '.format(i,count))


# For the given data set KMeans Clustering creates a better and distinct clustering compared to Spectral Clustering.
