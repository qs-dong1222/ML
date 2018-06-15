#!/usr/bin/python3

import numpy as xp
import matplotlib.pyplot as plt
import argparse
import random
import math

def gen_cluster_ctr(n_features, n_classes, cluster_ctr_dev):
    """
    generate centroids list for each cluster
    - n_features, number of features(dimensionality) of each centroid.
    - n_classes, number of cluster centroids
    - cluster_ctr_dev, deviation coefficient for centroid of each class.
                       The higher, the more separate for each cluster.
    :return: list of centroids
    """
    centroids = []
    for i_class in range(n_classes):
        c = []
        for i_feature in range(n_features):
            c.append(random.randint(round(-cluster_ctr_dev*100), round(cluster_ctr_dev*100)))
        centroids.append(c)
    return centroids




parser = argparse.ArgumentParser(description="*** Random Dataset Generation ***")
parser.add_argument('--n_classes', '-nc', nargs='?', type=int, default=2, help='number of classes to generate. '
                                                                               'This number is set 2 to circular dataset as constant.')
parser.add_argument('--n_samples', '-ns', nargs='?', type=int, default=200, help='number of samples to generate')
parser.add_argument('--n_features', '-nf', nargs='?', type=int, default=10, help='number of features to generate')
parser.add_argument('--simplicity', '-sp', nargs='?', type=float, default=5.0, help='simplicity of feature scattering. Larger values spread '
                                                                                    'out the clusters/classes and make the classification task easier.')
parser.add_argument('--cluster_std', '-cs', nargs='?', type=float, default=0.3, help='cluster standard deviation')
parser.add_argument('--feature_range', '-fr', nargs='?', type=float, default=0.5, help='deviation coefficient of feature of each class. '
                                                                                 'The higher, the wider range features can scatter at')
parser.add_argument('--dataset_type', '-dt', nargs='?', type=int, default=0, help='type of random dataset you desire to generate. '
                                                                                  '0: classification. '
                                                                                  '1: cluster. '
                                                                                  '2: circular. ')
parser.add_argument('--directory', '-dir', nargs='?', default="./", help='output directory path')
parser.add_argument('--verbose', '-v', action='store_true', help='verbose mode')  # false when not provided
args = parser.parse_args()


n_classes = args.n_classes
n_samples = args.n_samples
n_features = args.n_features
simplicity = args.simplicity
cluster_std = args.cluster_std
feature_range = args.feature_range
dataset_type = args.dataset_type
directory = args.directory
verbose = args.verbose

if(directory[-1] != '/'):
    directory += '/'

if(dataset_type==0):
    from sklearn.datasets.samples_generator import make_classification
    X, labels = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                    n_clusters_per_class=2, n_informative=math.ceil(math.log(2*n_classes, 2)),
                                    class_sep=simplicity, scale=feature_range)
elif(dataset_type==1):
    from sklearn.datasets.samples_generator import make_blobs
    centroids = gen_cluster_ctr(n_features, n_classes, feature_range)
    X, labels = make_blobs(n_samples=n_samples, centers=centroids, n_features=n_features,
                    cluster_std=cluster_std, random_state=0)
else:
    from sklearn.datasets.samples_generator import make_circles
    X, labels = make_circles(n_samples=n_samples, noise=0.2, factor=0.2, random_state=1)


if(verbose):
    print("---------------- configuration ----------------")
    print("n_classes =", n_classes)
    print("n_samples =", n_samples)
    print("n_features =", n_features)
    print("dataset_type =", dataset_type)

    # classification visualization
    if(dataset_type==0):
        print("classification simplicity =", simplicity)
        print("-----------------------------------------------")
        print('dataset shape', X.shape)
        print("labels", set(labels))
        rng = xp.random.RandomState(2)
        X += 2*rng.uniform(size=X.shape) # add noise
        unique_lables=set(labels)
        colors = plt.cm.Spectral(xp.linspace(0, 1, len(unique_lables)))
        for k,col in zip(unique_lables,colors):
            x_k = X[labels==k]
            plt.plot(x_k[:,0], x_k[:, random.randint(1, n_features-1)], 'o', markerfacecolor=col, markeredgecolor="k",
                     markersize=14)
        plt.title('data for classification(display the 1st and a random feature)')
        plt.show()

    # cluster visualization
    if(dataset_type==1):
        print("cluster_std =", cluster_std)
        print("feature_range =", feature_range)
        print("-----------------------------------------------")
        print('dataset shape', X.shape)
        print("labels", set(labels))
        unique_lables = set(labels)
        colors = plt.cm.Spectral(xp.linspace(0, 1, len(unique_lables)))
        for k,col in zip(unique_lables, colors):
            x_k = X[labels==k]
            plt.plot(x_k[:,0], x_k[:, random.randint(1, n_features-1)], 'o', markerfacecolor=col, markeredgecolor="k",
                     markersize=14)
        plt.title('data for cluster(display the 1st and a random feature)')
        plt.show()

    # circular dataset visualization
    if (dataset_type == 2):
        print("-----------------------------------------------")
        print('dataset shape', X.shape)
        print("labels", set(labels))
        unique_lables = set(labels)
        colors = plt.cm.Spectral(xp.linspace(0, 1, len(unique_lables)))
        for k,col in zip(unique_lables,colors):
            x_k = X[labels==k]
            plt.plot(x_k[:,0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="k",
                     markersize=14)
        plt.title('data for circular dataset(display the 1st and 2nd feature)')
        plt.show()


# save dataset
xp.save(directory+"features.npy", X)
xp.save(directory+"labels.npy", labels)
print("save: " + directory + "features.npy")
print("save: " + directory + "labels.npy")

