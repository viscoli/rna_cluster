import csv
import gzip
import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import Birch, AffinityPropagation, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from itertools import cycle, islice


matrix_dir = '../data/filtered_feature_bc_matrix'
mat = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx"))
mat = np.array(mat.todense())
features_path = os.path.join(matrix_dir, "features.tsv")
annotation  = pd.read_csv(features_path,sep='\t',header=None)
annotation.columns = ['feature_ids','gene_names','feature_types']
barcodes_path = os.path.join(matrix_dir, "barcodes.tsv")
barcodes = [line.strip() for line in open(barcodes_path, 'r')]
print('Matrix dimensionality {}'.format(mat.shape))
mat = mat.T #becase we want (samples,features) matrix


#===== 1
low_expr_thr = 100
high_expr_thr  = 100000

per_cell_sum = mat.sum(axis=1)
per_gene_sum = mat.sum(axis=0)

#===== 2
mat = mat[:,(per_gene_sum>=low_expr_thr) & (per_gene_sum<=high_expr_thr)] #just remove extreme outliers

mean_exp = mat.mean(axis=0)
std_exp = np.sqrt(mat.std(axis=0))
CV = std_exp/mean_exp

#===== 3
mat = mat[:,CV>=10]


cells_expression = mat.sum(axis=1)

#===== 4
mat = mat[cells_expression>=100,:]
mat = np.log(mat+1)

pca = PCA(n_components=100)
pca.fit(mat)
mat_reduce = pca.transform(mat)
embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.5,
                      metric='euclidean').fit_transform(mat_reduce.astype(np.float32, order='A'))


params = {'quantile': .3,
          'eps': .3,
          'damping': .9,
          'preference': -200,
          'n_neighbors': 10,
          'n_clusters': 5}
bandwidth = estimate_bandwidth(embedding, quantile=params['quantile'])
connectivity = kneighbors_graph(
        embedding, n_neighbors=params['n_neighbors'], include_self=False)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ward = AgglomerativeClustering(
    n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
spectral = SpectralClustering(
    n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
dbscan = DBSCAN(eps=params['eps'])
affinity_propagation = AffinityPropagation(
    damping=params['damping'], preference=params['preference'])
average_linkage = AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
birch = Birch(n_clusters=params['n_clusters'])
gmm = GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
clustering_algorithms = (
    ('AffinityPropagation', affinity_propagation),
    ('MeanShift', ms),
    ('SpectralClustering', spectral),
    ('Ward', ward),
    ('AgglomerativeClustering', average_linkage),
    ('DBSCAN', dbscan),
    ('Birch', birch),
    ('GaussianMixture', gmm))
#now plot everything
f, ax = plt.subplots(2, 4, figsize=(20,15))
for idx, (name, algorithm) in enumerate(clustering_algorithms):
    algorithm.fit(embedding)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(embedding)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    ax[idx//4, idx%4].scatter(embedding[:, 0], embedding[:, 1], s=2, color=colors[y_pred])
    #ax[idx//4, idx%4].xlim(-2.5, 2.5)
    #ax[idx//4, idx%4].ylim(-2.5, 2.5)
    ax[idx//4, idx%4].set_xticks(())
    ax[idx//4, idx%4].set_yticks(())
    ax[idx//4, idx%4].set_title(name)

plt.tight_layout()