#!/usr/bin/env python
#coding:utf-8

# Clustering for images for cloud to map
# Thoughts.
# 1) Re-run kmeans for all pixels in the "Land" section to get some mountains/terrain
# 2) For any small clusters (up to 10 pixels) just make them into the surrounding cluster value
# 3) Add a map border
# 4) Add styles instead of just pixel values (how?)


import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy import signal as sig


soften_edges = False
run_dbscan = False

# Initial source for dbscan pulled from the following source
# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
# modifications made as necessary.
def dbscan(): # Currently in work
    # In case the code is run with or without a parameter, cover these two
    in_img = plt.imread(sys.argv[1])[:,:,:3]

    # Reshape the input image to allow it to run through kmeans
    img = np.reshape(in_img,(len(in_img)*len(in_img[0]),len(in_img[0][0])))
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(img)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(np.amax(labels))

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = img[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

        xy = img[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def kmeans():
    new_color = [[162/255,197/255,220/255],[140/255,197/255,220/255],[0.337,0.596,0.345]]
    #reverse color order to get inverse
    new_color = new_color[::-1]

    # Cluster image into three different colors
    k_vals = [3]

    # In case the code is run with or without a parameter, cover these two
    in_img = plt.imread(sys.argv[1])[:,:,:3]

    # Reshape the input image to allow it to run through kmeans
    img = np.reshape(in_img,(len(in_img)*len(in_img[0]),len(in_img[0][0])))

    # For each K value (if more than 1, this will create multiple images)
    for j,k_val in enumerate(k_vals):
        # Run KMeans on the image
        kmeans = KMeans(n_clusters=k_val).fit(img)

        # Initialize an image for the output data
        kmeans_img = np.zeros(img.shape)

        # Get the max pixel value to determine if read in as 0-1 or 0-255
        max_val = np.amax(img)
        print(max_val)
        # Find out the darkest and lightest colors in the clusters
        color_sums = list({(idx,sum(color)) for idx,color in enumerate(kmeans.cluster_centers_)})
        color_sums.sort(key=lambda x:x[1],reverse=True)

        # Re-color the image using the different color scheme
        for idx,color in enumerate(color_sums):
            kmeans.cluster_centers_[color[0]] = new_color[idx]

        # Draw out to the image using the cluster centers and the associated pixel labels
        for idx,pixel in enumerate(kmeans.labels_): 
            # Accoutn for if read in as 0-1 or 0-255
            if max_val > 1.0:
                print("using scale")
                kmeans_img[idx] = kmeans.cluster_centers_[pixel] / 255
            else:
                kmeans_img[idx] = kmeans.cluster_centers_[pixel]
        
        print(kmeans.labels_)

        if soften_edges == True:
            R = np.reshape(np.asarray(kmeans_img)[:,0],(len(in_img),len(in_img[0])))
            G = np.reshape(np.asarray(kmeans_img)[:,1],(len(in_img),len(in_img[0])))
            B = np.reshape(np.asarray(kmeans_img)[:,2],(len(in_img),len(in_img[0])))
            filt_size = 3
            filt_R = sig.medfilt2d(R,kernel_size=filt_size)
            filt_G = sig.medfilt2d(G,kernel_size=filt_size)
            filt_B = sig.medfilt2d(B,kernel_size=filt_size)
            filt_img = np.zeros((len(filt_R),len(filt_R[0]),3))
            for pix_x in range(len(filt_R)):
                for pix_y in range(len(filt_R[0])):
                    filt_img[pix_x][pix_y] = filt_R[pix_x][pix_y],filt_G[pix_x][pix_y],filt_B[pix_x][pix_y]
            out_img = np.reshape(filt_img,(len(in_img),len(in_img[0]),len(in_img[0][0])))
        else:
            out_img = np.reshape(kmeans_img,(len(in_img),len(in_img[0]),len(in_img[0][0])))
            np.reshape(out_img,(len(in_img),len(in_img[0]),len(in_img[0][0])))

        plt.imshow((out_img*255).astype(np.uint8))
        plt.draw()
    plt.show()

if __name__ == '__main__':
    if run_dbscan == True:
        print('dbscanning')
        dbscan()
    else:
        print('kmeaning')
        kmeans()