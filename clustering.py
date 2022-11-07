from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.base import ClusterMixin
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import FastICA as ICA
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import FactorAnalysis
from scipy.stats import norm, kurtosis
from sklearn.random_projection import SparseRandomProjection
import pdb
import pandas as pd
import numpy as np
from NNAgent import NNAgent
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, KElbow
from sklearn.metrics.pairwise import pairwise_distances
from yellowbrick.features import PCA, pca_decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import decomposition
import matplotlib.pyplot as plt


# From https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_vs_pca.html
# For ICA visualization
def plot_samples(S, axis_list=None):
    plt.scatter(
        S[:, 0], S[:, 1], s=2, marker="o", zorder=10, color="steelblue", alpha=0.5
    )
    if axis_list is not None:
        for axis, color, label in axis_list:
            axis /= axis.std()
            x_axis, y_axis = axis
            plt.quiver(
                (0, 0),
                (0, 0),
                x_axis,
                y_axis,
                zorder=11,
                width=0.01,
                scale=6,
                color=color,
                label=label,
            )

    plt.hlines(0, -3, 3)
    plt.vlines(0, -3, 3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("x")
    plt.ylabel("y")


# Wrapper to get elbow visualization for EM, as per this post about the library.  This post makes no indication about being used for this class
# https://stackoverflow.com/questions/58648969/the-supplied-model-is-not-a-clustering-estimator-in-yellowbrick
class GaussianMixtureCluster(GaussianMixture, ClusterMixin):
    """Subclass of GaussianMixture to make it a ClusterMixin."""

    def fit(self, X):
        super().fit(X)
        self.labels_ = self.predict(X)
        return self

    def get_params(self, **kwargs):
        output = super().get_params(**kwargs)
        output["n_clusters"] = output.get("n_components", None)
        return output

    def set_params(self, **kwargs):
        kwargs["n_components"] = kwargs.pop("n_clusters", None)
        return super().set_params(**kwargs)

def elbow(X, dataname, expName):
    model = KMeans(random_state=0)
    vis_elbow = KElbowVisualizer(model, k=(2,20))

    vis_elbow.fit(X)
    # vis.show()
    vis_elbow.show('./graphs/elbow/{}_{}_{}.png'.format('Water', "Kmeans", expName))
    plt.figure()


    # Citation/ use of calinski_harabasz_score
    # https://stackoverflow.com/questions/58648969/the-supplied-model-is-not-a-clustering-estimator-in-yellowbrick
    model_g = GaussianMixtureCluster()
    vis_gm = KElbow(model_g, k=(2,20), force_model=True)
    vis_gm.fit(X)
    vis_gm.show('./graphs/elbow/{}_{}_{}.png'.format('Water', "EM", expName))
    plt.figure()


def silh(X, dataname, expName):

    model = KMeans(random_state=0)
    vis_silh = SilhouetteVisualizer(model, k=(2,20))

    vis_silh.fit(X)
    # vis_silh.show()
    vis_silh.show('./graphs/silh/{}_{}_{}.png'.format(dataname, "Kmeans", expName))
    plt.figure()


    model_g = GaussianMixtureCluster()
    vis_silh_g = SilhouetteVisualizer(model, k=(2,20), force_model=True)

    vis_silh_g.fit(X)
    # vis_silh.show()
    vis_silh_g.show('./graphs/silh/{}_{}_{}.png'.format(dataname, "EM", expName))
    plt.figure()


def adj_rand(X, Y, dataname, expName):

    clusters = np.arange(2,21)

    homo_scores = []
    ar_scores = []

    for cluster in clusters:
        model = KMeans(random_state=0, n_clusters=cluster)
        model.fit(X)
        ar = adjusted_rand_score(Y, model.predict(X))
        homo = homogeneity_score(Y, model.predict(X))
        homo_scores.append(homo)
        ar_scores.append(ar)


    # pdb.set_trace()
    plt.figure()
    plt.plot(clusters, homo_scores, color="red", label="Homogeneity")
    plt.plot(clusters, ar_scores, color="cyan", label="ARI")
    plt.title('{} {} ({}, {})'.format("Homogeneity Score vs Clusters", 'Kmeans', dataname,  expName))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Homogeneity Score")
    plt.legend()
    plt.savefig('./graphs/homo/{}_{}_{}.png'.format(dataname, "Kmeans", expName))


def pcaWater(X, Y):
    
    # pca = PCA(random_state=0)
    # pca.fit(X)

    # pdb.set_trace()
    classes = [0, 1]

    model = decomposition.PCA(n_components=9)
    model.fit(X)
    X_train = model.transform(X)

    visualizer = PCA(scale=True, proj_features=True, classes=classes)
    visualizer.fit_transform(X_train, Y)
    visualizer.show("./graphs/pca/Water_pre")

    visualizer = PCA(scale=True, proj_features=True, classes=classes)
    visualizer.fit_transform(X, Y)
    visualizer.show("./graphs/pca/Water_post")

    #pca_decomposition(X, Y, projection=3)

def pcaWine(X, Y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(Y)
    classes = [1.0,2.0,3.0]

    model = decomposition.PCA(n_components=11)
    model.fit(X)
    X_train = model.transform(X)
    # pdb.set_trace()

    visualizer = PCA(scale=True, proj_features=True, classes=classes)
    visualizer.fit_transform(X_train, y)
    visualizer.show("./graphs/pca/Wine_pre")

    visualizer = PCA(scale=True, proj_features=True, classes=classes)
    visualizer.fit_transform(X, y)
    visualizer.show("./graphs/pca/Wine_post")

def icaWater(X,Y):
    model = FastICA(random_state=0, n_components=9, max_iter=1000)
    model.fit_transform(X)
    est = model.transform(X)
    est /= est.std(axis=0)

    plt.figure()
    #plt.subplot(2, 2, 1)
    #plot_samples(X / X.std())
    #plt.title("True Independent Sources")

    # axis_list = [(model.mixing_, "red", "ICA"), (model.mixing_, "red", "ICA")]
    plt.subplot(2, 2, 2)
    plot_samples(X / np.std(X), axis_list=(model.mixing_, "red", "ICA"))
    legend = plt.legend(loc="lower right")
    legend.set_zorder(100)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
    plt.show()


def ica(X,Y, dataname, expName):
    comps = np.arange(2,X.shape[1]+1)

    scores = []
    for comp in comps:
        model = FastICA(random_state=0, n_components=comp, max_iter=1000)
        model.fit_transform(X)
        new_X = model.transform(X)
        scores.append(kurtosis(new_X)[0])

    plt.figure()
    plt.plot(comps, scores, color="red", label="Kurtosis Score")
    plt.title('{} ({}, {})'.format("ICA Kurtosis vs Components", dataname,  expName))
    plt.xlabel("Number of Components")
    plt.ylabel("Kurtosis")
    plt.legend()
    plt.savefig('./graphs/ica/{}_{}_{}.png'.format(dataname, "dimred", expName))


def rand_proj(X,Y, dataname, expName):
    comps = np.arange(2,X.shape[1]+1)

    scores = []
    for comp in comps:
        model = SparseRandomProjection(random_state=0, n_components=comp)
        new_X = model.fit_transform(X)
        d0 = pairwise_distances(new_X)
        d1 = pairwise_distances(X)
        score = np.corrcoef(d0.ravel(),d1.ravel())[0,1]
        scores.append(score)

    plt.figure()
    plt.plot(comps, scores, color="red", label="Mean Recon Error")
    plt.title('{} ({}, {})'.format("Rand Proj Mean Recons Error vs Components", dataname,  expName))
    plt.xlabel("Number of Components")
    plt.ylabel("Mean Reconstruction Error")
    plt.legend()
    plt.savefig('./graphs/rp/{}_{}_{}.png'.format(dataname, "dimred", expName))

def fa(X,Y, dataname, expName):
    comps = np.arange(2,X.shape[1]+1)

    scores = []
    for comp in comps:
        model = FactorAnalysis(random_state=0, n_components=comp, )
        model.fit_transform(X)
        new_X = model.transform(X)
        # pdb.set_trace()
        d0 = pairwise_distances(new_X)
        d1 = pairwise_distances(X)
        score = np.corrcoef(d0.ravel(),d1.ravel())[0,1]
        scores.append(score)

    plt.figure()
    plt.plot(comps, scores, color="red", label="Mean Recon Error")
    plt.title('{} ({}, {})'.format("Factor Analysis Recons Error vs Components", dataname,  expName))
    plt.xlabel("Number of Components")
    plt.ylabel("Mean Reconstruction Error")
    plt.legend()
    plt.savefig('./graphs/fa/{}_{}_{}.png'.format(dataname, "dimred", expName))

'''
    comps = np.arange(2,X.shape[1]+1)

    scores = []
    for comp in comps:
        model = FastICA(random_state=0, n_components=comp, max_iter=1000)
        cv_score = np.mean(cross_val_score(model, X))
        scores.append(np.mean(cv_score))
'''

def exp3Water(X,Y):

    model_pca = decomposition.PCA(n_components=6)
    model_pca.fit(X)
    X_pca = model_pca.transform(X)
    adj_rand(X,Y, "Water", "pca")
    elbow(X, "Water", "pca")
    silh(X, "Water", "pca")

    model_ica = FastICA(random_state=0, n_components=5, max_iter=1000)
    model_ica.fit_transform(X)
    X_ica = model_ica = model_ica.transform(X)
    adj_rand(X,Y, "Water", "ica")
    elbow(X, "Water", "ica")
    silh(X, "Water", "ica")

    model_rp = SparseRandomProjection(random_state=0, n_components=2)
    model_rp.fit_transform(X)
    X_rp = model_rp.fit_transform(X)
    adj_rand(X,Y, "Water", "rp")
    elbow(X, "Water", "rp")
    silh(X, "Water", "rp")

    model_fa = FactorAnalysis(random_state=0, n_components=9 )
    model_fa.fit_transform(X)
    X_fa = model_fa.fit_transform(X)
    adj_rand(X,Y, "Water", "fa")
    elbow(X, "Water", "fa")
    silh(X, "Water", "fa")

def exp3Wine(X,Y):

    model_pca = decomposition.PCA(n_components=9)
    model_pca.fit(X)
    X_pca = model_pca.transform(X)
    adj_rand(X,Y, "Wine", "pca")
    elbow(X, "Wine", "pca")
    silh(X, "Wine", "pca")

    model_ica = FastICA(random_state=0, n_components=11, max_iter=1000)
    model_ica.fit_transform(X)
    X_ica = model_ica = model_ica.transform(X)
    adj_rand(X,Y, "Wine", "ica")
    elbow(X, "Wine", "ica")
    silh(X, "Wine", "ica")

    model_rp = SparseRandomProjection(random_state=0, n_components=2)
    model_rp.fit_transform(X)
    X_rp = model_rp.fit_transform(X)
    adj_rand(X,Y, "Wine", "rp")
    elbow(X, "Wine", "rp")
    silh(X, "Wine", "rp")

    model_fa = FactorAnalysis(random_state=0, n_components=7 )
    model_fa.fit_transform(X)
    X_fa = model_fa.fit_transform(X)
    adj_rand(X,Y, "Wine", "fa")
    elbow(X, "Wine", "fa")
    silh(X, "Wine", "fa")


def exp4Water(X,Y):

    model_pca = decomposition.PCA(n_components=6)
    model_pca.fit(X)
    X_pca = model_pca.transform(X)

    print("Running NN Experiment with Dataset:{}".format("pca"))
    x_train, x_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2)
    nnAgent = NNAgent(dataset="pca", scorer='accuracy')
    nnAgent.initModel(x_train,y_train)

    nnAgent.plot_learning_timing_curve(x_train, y_train)
    nnAgent.plot_validation_curve(x_train, y_train)
    nnAgent.get_cv_results(x_train, y_train)
    nnAgent.get_final_acc(x_test, y_test)
    nnAgent.save_final_params()

    print("Running NN Experiment with Dataset:{}".format("ica"))
    model_ica = FastICA(random_state=0, n_components=5, max_iter=1000)
    model_ica.fit_transform(X)
    X_ica = model_ica = model_ica.transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_ica, Y, test_size=0.2)
    nnAgent = NNAgent(dataset="ica", scorer='accuracy')
    nnAgent.initModel(x_train,y_train)

    nnAgent.plot_learning_timing_curve(x_train, y_train)
    nnAgent.plot_validation_curve(x_train, y_train)
    nnAgent.get_cv_results(x_train, y_train)
    nnAgent.get_final_acc(x_test, y_test)
    nnAgent.save_final_params()


    print("Running NN Experiment with Dataset:{}".format("rp"))
    model_rp = SparseRandomProjection(random_state=0, n_components=2)
    model_rp.fit_transform(X)
    X_rp = model_rp.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_rp, Y, test_size=0.2)
    nnAgent = NNAgent(dataset="rp", scorer='accuracy')
    nnAgent.initModel(x_train,y_train)

    nnAgent.plot_learning_timing_curve(x_train, y_train)
    nnAgent.plot_validation_curve(x_train, y_train)
    nnAgent.get_cv_results(x_train, y_train)
    nnAgent.get_final_acc(x_test, y_test)
    nnAgent.save_final_params()


    print("Running NN Experiment with Dataset:{}".format("fa"))
    model_fa = FactorAnalysis(random_state=0, n_components=9 )
    model_fa.fit_transform(X)
    X_fa = model_fa.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X_fa, Y, test_size=0.2)
    nnAgent = NNAgent(dataset="fa", scorer='accuracy')
    nnAgent.initModel(x_train,y_train)

    nnAgent.plot_learning_timing_curve(x_train, y_train)
    nnAgent.plot_validation_curve(x_train, y_train)
    nnAgent.get_cv_results(x_train, y_train)
    nnAgent.get_final_acc(x_test, y_test)
    nnAgent.save_final_params()



def runExpWater(X, Y):

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # pdb.set_trace()

    #adj_rand(X,Y, "Water", "std")
    #elbow(X, "Water", "std")
    #silh(X, "Water", "std")
    #pcaWater(X,Y)
    #ica(X,Y, "Water", "std")
    #rand_proj(X,Y, "Water", "std")
    #fa(X,Y, "Water", "std")

    #exp3Water(X,Y)

    exp4Water(X,Y)


def runExpWine(X,Y):
    # adj_rand(X,Y, "Wine", "std")
    # elbow(X, "Wine", "std")
    # silh(X, "Wine", "std")
    pcaWine(X,Y)
    ica(X,Y, "Wine", "std")
    rand_proj(X,Y, "Wine", "std")
    fa(X,Y, "Wine", "std")
