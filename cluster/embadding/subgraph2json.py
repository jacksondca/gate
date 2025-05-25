import queue
import json
import random

from tqdm import tqdm
from cluster.io import read_nsg_index, read_kmeans_file, read_fvecs


MAX_HOP = 5

def sample_negb(x, negb):
    sorted_negb = sorted(negb, key=lambda n: distance(x, n))
    nearest = sorted_negb[:SAMPLE_SIZE // 2]
    farthest = sorted_negb[-(SAMPLE_SIZE-SAMPLE_SIZE//2):]
    return set(nearest + farthest)


def sample_graph(ep, g, fvecs) -> dict:
    visited = set()
    q = queue.Queue()
    q.put((ep, 0))
    
    while not q.empty():
        id, hop = q.get()
        if id in visited or hop > MAX_HOP:
            continue
        visited.add(id)
        
        for nx in sample_negb(id, g[id]):
            q.put((nx, hop+1))
    
    # 1. edge
    edges = []
    for point in visited:
        for nx in g[point]:
            if nx in visited:
                edges.append([int(point), int(nx)])
    # 2. feature
    features = {}
    for point in visited:
        vec = fvecs[point]
        features[str(point)] = vec.tolist()

    return {
        "edges": edges,
        "features": features,
    }


def main(nsg_index, kmeans_file, fvecs_files, subgraph_dir):
    fvecs = read_fvecs(fvecs_file)
    iter1_num, iter2_num, _, _, eps = read_kmeans_file(kmeans_file)
    _, _, nsg = read_nsg_index(nsg_index)

    for id in tqdm(range(iter1_num * iter2_num), desc='subgraphing'):
        row, col = id // iter2_num, id % iter2_num
        ep = eps[row][col]
        json_obj = sample_graph(ep, nsg, fvecs)

        with open(f"{subgraph_dir}/{id}.json", 'w') as fp:
            json.dump(json_obj, fp=fp, indent=4)


if __name__ == '__main__':
    db = ''
    nsg_index = f''
    fvecs_file = f''
    kmeans_file = f''
    subgraph_dir = f''

    SAMPLE_SIZE = 10

    import os
    os.makedirs(subgraph_dir, exist_ok=True)

    main(nsg_index, kmeans_file, fvecs_file, subgraph_dir)
