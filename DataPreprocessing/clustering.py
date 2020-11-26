import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import mixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,silhouette_samples
import numpy as np
from matplotlib import cm


# Silhouette function
def plotSilhouette(X, pred):
    cluster_labels = np.unique(pred)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, pred, metric='euclidean')
    pred_ax_lower, pred_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[pred == c]
        c_silhouette_vals.sort()
        pred_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i/n_clusters)

        plt.barh(range(pred_ax_lower, pred_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none',
                 color=color)
        yticks.append(pred_ax_lower + pred_ax_upper/2)
        pred_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels+1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()


# Clustering functions
def em(n_components, max_iter, df, inform):
    max_score = 0
    max_score_model = None
    max_score_df = None

    for d in df:
        for n in n_components:
            for m in max_iter:
                gmm = mixture.GaussianMixture(n_components=n, max_iter=m, random_state=0)
                gmm.fit(d)

                label = gmm.predict(d)

                # evaluation with silhouette
                s_score = silhouette_score(d, label, metric='euclidean')
                if s_score > max_score:
                    max_score = s_score
                    max_score_model = label
                    max_score_df = d
                if s_score > 0.8:
                    labels = gmm.predict(d)
                    plt.scatter(d[d.columns[0]], d[d.columns[1]], c=labels, s=40, cmap='viridis')
                    plt.title((n, 'components with', m, 'iteratons'))
                    plt.show()
    print(inform + " EM Clustering Max Silhouette Score: " + str(max_score))
    plotSilhouette(max_score_df, max_score_model)


def dbscan(dist_func, eps, min_samples, df, inform):
    algorithm_euc = ['ball_tree', 'kd_tree', 'brute']
    algorithm_ha = ['ball_tree', 'brute']
    algorithm = list()
    max_score = 0
    max_score_model = None
    max_score_df = None

    if dist_func == 'euclidean':
        algorithm = algorithm_euc
    else:
        algorithm = algorithm_ha

    for d in df:
        for e in eps:
            for m in min_samples:
                for a in algorithm:
                    db = DBSCAN(metric=dist_func, eps=e, min_samples=m, algorithm=a).fit(d)

                    if max(db.labels_) > 0:
                        # evaluation with silhouette
                        s_score = silhouette_score(d, db.labels_, metric='euclidean')
                        if s_score > max_score:
                            max_score = s_score
                            max_score_model = db
                            max_score_df = d
                        if s_score > 0.8:
                            labels = db.labels_
                            plt.scatter(d[d.columns[0]], d[d.columns[1]], c=labels, s=40, cmap='viridis')
                            if dist_func == 'euclidean':
                                plt.title((e, 'eps with', m, 'min_samples', a, 'Algorithm & Euclidean'))
                            else:
                                plt.title((e, 'eps with', m, 'min_samples', a, 'Algorithm & Haming'))
                            plt.show()
    print(inform + " DBSCAN " + dist_func + " Max Silhouette Score: " + str(max_score))
    if max_score != 0:
        plotSilhouette(max_score_df, max_score_model.labels_)


def kmeans(n_cluster, max_iter, df, inform):
    max_score = 0
    max_score_model = None
    max_score_df = None

    for d in region_pca_df:
        for n in n_cluster:
            for m in max_iter:
                kmeans = KMeans(n_clusters=n, max_iter=m, random_state=0)
                kmeans.fit(d)

                # evaluation with silhouette
                s_score = silhouette_score(d, kmeans.labels_, metric='euclidean')
                if s_score > max_score:
                    max_score = s_score
                    max_score_model = kmeans
                    max_score_df = d

                if s_score > 0.8:
                    labels = pd.DataFrame(kmeans.labels_)
                    plt.scatter(d[d.columns[0]], d[d.columns[1]], c=labels, s=40, cmap='viridis')
                    plt.title((n, 'N_cluster with', m, 'max_iter'))
                    plt.show()
    print(inform + " K-Means Clustering Max Silhouette Score: " + str(max_score))
    plotSilhouette(max_score_df, max_score_model.labels_)


# 불러오기
df_region_1960 = pd.read_csv('../Data/sample/1960r.csv')
df_region_2010 = pd.read_csv('../Data/sample/2010r.csv')
df_income_1960 = pd.read_csv('../Data/sample/1960i.csv')
df_income_2010 = pd.read_csv('../Data/sample/2010i.csv')
df_unit_1960 = pd.read_csv('../Data/sample/1960u.csv')
df_unit_2010 = pd.read_csv('../Data/sample/2010u.csv')

# ## Scaling & PCA
sd = StandardScaler()
new_cols = df_region_1960.columns

sd_region_1960 = sd.fit_transform(df_region_1960[new_cols[2:7]])
sd_region_2010 = sd.fit_transform(df_region_2010[new_cols[2:7]])
sd_income_1960 = sd.fit_transform(df_income_1960[new_cols[2:7]])
sd_income_2010 = sd.fit_transform(df_income_2010[new_cols[2:7]])
sd_unit_1960 = sd.fit_transform(df_unit_1960[new_cols[2:7]])
sd_unit_2010 = sd.fit_transform(df_unit_2010[new_cols[2:7]])

pca = PCA(n_components=2)
X_principal = pca.fit_transform(sd_region_1960)
X_principal_region_1960 = pd.DataFrame(X_principal)
X_principal_region_1960.columns = ['P1_1960r','P2_1960r']

X_principal = pca.fit_transform(sd_region_2010)
X_principal_region_2010 = pd.DataFrame(X_principal)
X_principal_region_2010.columns = ['P1_2010r','P2_2010r']

X_principal = pca.fit_transform(sd_income_1960)
X_principal_income_1960 = pd.DataFrame(X_principal)
X_principal_income_1960.columns = ['P1_1960i','P2_1960i']

X_principal = pca.fit_transform(sd_region_2010)
X_principal_income_2010 = pd.DataFrame(X_principal)
X_principal_income_2010.columns = ['P1_2010i','P2_2010i']

X_principal = pca.fit_transform(sd_unit_1960)
X_principal_unit_1960 = pd.DataFrame(X_principal)
X_principal_unit_1960.columns = ['P1_1960u','P2_1960u']

X_principal = pca.fit_transform(sd_unit_2010)
X_principal_unit_2010 = pd.DataFrame(X_principal)
X_principal_unit_2010.columns = ['P1_2010u','P2_2010u']

region_pca_df = [X_principal_region_1960, X_principal_region_2010]
income_pca_df = [X_principal_income_1960, X_principal_income_2010]
unit_pca_df = [X_principal_unit_1960, X_principal_unit_2010]


# ## DBSCAN
# hyper parameter setting
eps_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
min_samples_list = [5, 10, 20, 50, 100]
distance = ['euclidean', 'hamming']

# Euclidean
dbscan(distance[0], eps_list, min_samples_list, region_pca_df, 'Region')
dbscan(distance[0], eps_list, min_samples_list, income_pca_df, 'Income')
dbscan(distance[0], eps_list, min_samples_list, unit_pca_df, 'Unit')

# Haming
dbscan(distance[1], eps_list, min_samples_list, region_pca_df, 'Region')
dbscan(distance[1], eps_list, min_samples_list, income_pca_df, 'Income')
dbscan(distance[1], eps_list, min_samples_list, unit_pca_df, 'Unit')


# ## K-means
# hyper parameter setting
n_cluster_list = [2,3,4,5,6]
max_iter_list = [50,100,200,300]

kmeans(n_cluster_list, max_iter_list, region_pca_df, 'Region')
kmeans(n_cluster_list, max_iter_list, income_pca_df, 'Income')
kmeans(n_cluster_list, max_iter_list, unit_pca_df, 'Unit')

# ## EM
# hyper parameter setting
n_components_list = [2,3,4,5,6]
max_iter_list = [50,100,200,300]

em(n_components_list, max_iter_list, region_pca_df, 'Region')
em(n_components_list, max_iter_list, income_pca_df, 'Income')
em(n_components_list, max_iter_list, unit_pca_df, 'Unit')



