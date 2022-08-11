import pickle
from clustering import obtain_clusterings, plot_usage_distribution, make_usage_matrices_

# with open('data/gulordava/century/usages_len128.dict', 'rb') as f:
#     usages = pickle.load(f)
#
# with open('usages_nahar.pkl', 'rb') as f:
#     usages = pickle.load(f)

usages = make_usage_matrices_('usages_nahar.pkl', mode='concat', usages_out=None, ndims=768)

# clusterings = obtain_clusterings(
#     usages,
#     out_path='data/gulordava/century/usages_len128.clustering.2.dict',
#     method='kmeans',
#     criterion='silhouette'
# )

print()
clusterings = obtain_clusterings(
    usages,
    out_path='usages_nahar_clustering.pkl',
    method='kmeans',
    criterion='silhouette'
)

plot_usage_distribution(usages, clusterings, '', normalized=True)
