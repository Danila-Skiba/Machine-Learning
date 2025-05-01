from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, fowlkes_mallows_score
from sklearn.metrics.cluster import rand_score
import pandas as pd
import matplotlib.pyplot as plt
def plot_scatter(model, data, centroiids = True):
    plt.scatter(data[0], data[1], c = model.labels_, alpha=0.6);
    if centroiids:
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], s=100, c='red', alpha=0.6);
        for i, centroid in enumerate(model.cluster_centers_):
            plt.annotate(str(i),  
                         xy=(centroid[0], centroid[1]),  
                         xytext=(5, 5), 
                         textcoords="offset points",  
                         ha='center',  
                         va='bottom',
                         color='black', 
                         fontsize=10)   

def plot_scatter_gm(model, data, x = 0, y = 1):
    plt.scatter(data[x], data[y], c = model.predict(data), alpha=0.7)
    plt.scatter(model.means_[:, x], model.means_[:, y], s =100, c = 'red', alpha=0.6)
    for i, centroid in enumerate(model.means_):
            plt.annotate(str(i),  
                         xy=(centroid[0], centroid[1]),  
                         xytext=(5, 5), 
                         textcoords="offset points",  
                         ha='center',  
                         va='bottom',
                         color='black', 
                         fontsize=10)  
def print_metrics(model, data, data_Y = None, rs = False, labels = None ):
    if labels is None:
        labels = model.labels_
    ss = silhouette_score(data, labels)
    ch = calinski_harabasz_score(data, labels)
    print(pd.DataFrame([ss,ch], index = ['Silhouette', 'Calinski'], columns = ['Внутренние метрики']));
    if rs:
        r_score = rand_score(data_Y, labels)
        f_score = fowlkes_mallows_score(data_Y, labels)
        print(pd.DataFrame([r_score, f_score], index = ['R-score', 'Fowlkes-score'], columns=['Внешние метрики']));

def elbow_method(data):
    inertias = []

    for i in range(2, 10):
        m = KMeans(i, n_init='auto', init='k-means++').fit(data)
        inertias.append(m.inertia_)
    print("Метод Локтя")
    plt.plot(range(2,10), inertias);


def siluet_method(data):
    ss =[]
    for i in range(2, 10):
        ss.append(silhouette_score(data, KMeans(i, n_init='auto').fit(data).labels_))
    print("Метод силуэта")
    plt.plot(range(2,10), ss);
