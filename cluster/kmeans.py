import torch
import json

import numpy as np

from tqdm import tqdm

from cluster.io import *
from cluster.json2binary import main as json2binary


def init_centroids(X, centroid_num):
    centroids = X[torch.randperm(X.shape[0])[:centroid_num]]
    return centroids


def assign_clusters(X, centroids, batch_size):
    total = X.shape[0]
    cluster_labels = torch.tensor([]).cuda()
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        X_batch = X[start: end]
        distances_batch = torch.cdist(X_batch, centroids)
        _, cluster_labels_batch = torch.min(distances_batch, dim=1)
        cluster_labels = torch.cat((cluster_labels, cluster_labels_batch), dim=0)
    return cluster_labels


def update_centroid(X, labels, centroid_num):
    new_centroids = torch.zeros(centroid_num, X.shape[1])
    for i in range(centroid_num):
        new_centroids[i] = X[labels == i].mean(dim=0)
    return new_centroids.cuda()


def calculate_loss(x, y, batch_size):
    total = x.shape[0]
    loss = 0
    y = y.reshape((1, -1))
    for s in range(0, total, batch_size):
        e = min(total, s + batch_size)
        x_batch = x[s: e]
        dist_batch = torch.cdist(x_batch, y)
        loss += torch.sum(dist_batch)
    return loss


def kmeans(X, max_iter, centroid_num, desc):
    centroids = init_centroids(X, centroid_num)
    
    for _ in tqdm(range(max_iter), desc=desc):
        labels = assign_clusters(X, centroids, batch_size)
        new_centroids = update_centroid(X, labels, centroid_num)
        if torch.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids


def main():
    X = read_fvecs(file_path)

    X = torch.tensor(X).cuda()
    
    statistics = {}
    statistics['meta'] = {
        'centroid_num_stage1': centroid_num_stage1,
        'centroid_num_stage2': centroid_num_stage2,
        'iter_1': iter_1,
        'iter_2': iter_2,
    }
    statistics['data'] = {}
    
    labels_stage1, centroids_stage1 = kmeans(X, iter_1, centroid_num_stage1, 'stage-1')
    
    for i in range(centroid_num_stage1):
        X_stage2 = X[labels_stage1 == i]
        labels_stage2, centroid_stage2 = kmeans(X_stage2, iter_2, centroid_num_stage2, f'stage-2-{i}')
        origin_indices = torch.nonzero(labels_stage1 == i).squeeze()
        
        for j in range(centroid_num_stage2):
            partition_id = centroid_num_stage2 * i + j
            x = X_stage2[labels_stage2 == j]
            
            cur_indices = torch.nonzero(labels_stage2 == j).squeeze().reshape((1, -1))
            ids = origin_indices[cur_indices].tolist()
            statistics['data'][partition_id] = ids

    with open(result_json_path, 'w') as fp:
        json.dump(statistics, fp=fp, indent=4)


if __name__ == '__main__':

    main()

    json2binary(result_json_path, result_path, file_path)
