import torch
import json
import pandas as pd
import numpy as np
import torch.nn.functional as F

from pathlib import Path

from cluster.io import read_fvecs, read_ivecs, write_fvecs
from cluster.model.train.utils import get_model_name
from cluster.model.train.params import device, deep10m_query1b_dataset_params, sift_query1k_dataset_params, deep10m_query1k_dataset_params, \
                                        gist_query1k_dataset_params, \
                                        tiny5m_query1k_kmeans16_32_dataset_params, \
                                        text2image10m_query1k_kmeans16_32_dataset_params, \
                                        laion3m_query1k_kmeans16_16_dataset_params, \
                                        sift_query1k_rq4_dataset_params


params = sift_query1k_rq4_dataset_params


def infer(params, model):
    # vecs
    EMBADDING_FILE_PATH = params['EMBADDING_FILE_PATH']
    QUERY_FILE_PATH = params['QUERY_FILE_PATH']
    EP_FILE_PATH = params['EP_FILE_PATH']

    # sample
    POS_SAMPLE_PATH = params['POS_SAMPLE_PATH']
    NEG_SAMPLE_PATH = params['NEG_SAMPLE_PATH']

    # model
    EP_SAVE_PATH = params['EP_SAVE_PATH']
    MODEL_PATH = params['MODEL_PATH']

    kmeans_size = params['KMEANS_SIZE']
    hop_dir = Path(params['HOP_DIR'])

    # load hop count
    hop_counts = []
    for i in range(kmeans_size):
        file = hop_dir / 'data' / f'hop_count_ep{i}.ivecs'
        hop_counts.append(read_ivecs(file).reshape(-1))

    hop_counts = np.array(hop_counts).T

    # load gt ep
    gt_ep = []
    with open(POS_SAMPLE_PATH, 'r') as f:
        for line in f.readlines():
            gt_ep.append(list(map(int, line.split())))

    # load graph feature
    x_data = pd.read_csv(EMBADDING_FILE_PATH, index_col=0)
    graph_features = [x_data.loc[i].tolist() for i in range(len(x_data))]

    # load node feature
    node_features = read_fvecs(EP_FILE_PATH)

    # load query vecs
    query_vecs = read_fvecs(QUERY_FILE_PATH)
    
    query_vecs = torch.Tensor(query_vecs).to(device)
    node_features = torch.Tensor(node_features).to(device)
    graph_features = torch.Tensor(graph_features).to(device)

    embaddings = model.forward(node_features, graph_features)

    eps = []
    ds = []
    g, e, o = [], [], []
    total_delta = 0
    opt = 0
    for qid, (query_vec, hop_count, gt) in enumerate(zip(query_vecs, hop_counts, gt_ep)):
        # dot_products = torch.matmul(embaddings, query_vec)
        # ep = torch.argmax(dot_products)

        # dist = torch.cdist(embaddings, query_vec.unsqueeze(0)).squeeze()
        # ep = int(torch.argmin(dist))

        cosine_similarities = F.cosine_similarity(embaddings, query_vec.unsqueeze(0), dim=1)
        ep = int(torch.argmax(cosine_similarities))

        structured_dist = torch.cdist(node_features, query_vec.unsqueeze(0)).squeeze()
        origin_ep = int(torch.argmin(structured_dist))

        total_delta = total_delta + hop_count[origin_ep] - hop_count[ep]

        # alpha = 1000
        # dist = dist + alpha * cosine_similarities
        # dist = dist * cosine_similarities
        # dist[cosine_similarities < 0] = 1e9
        # ep = int(torch.argmin(dist))

        delta = hop_count[ep] - np.min(hop_count)
        ds.append(delta)

        # cos = F.cosine_similarity(torch.zeros(128).to(device), query_vec, dim=0)
        if delta > 3:
            e.append(qid)
        elif delta in [0, 1]:
            g.append(qid)
        else:
            o.append(qid)
        
        opt += max(0, hop_count[origin_ep] - hop_count[ep])

        # dists = torch.cdist(embaddings, query_vec.unsqueeze(0)).reshape(-1)
        # ep = int(torch.argmax(dists))
        # breakpoint()
        eps.append(ep)

    opt_eps = np.array(eps)
    opt_eps = opt_eps[np.array(g)]
    opt_query_vecs = np.array(query_vecs.cpu())[np.array(g)]

    # import collections
    # print('hop diff with opt:')
    # print(collections.Counter(ds))
    # print(dict(collections.Counter(ds)))
    print(f"{opt=}")

    count = 0
    for ep, gt_eps in zip(eps, gt_ep):
        if ep in gt_eps:
            count += 1
    # print(f"{count / len(eps)}")
    return len(g) / len(eps), opt_eps, opt_query_vecs, embaddings, opt


def main(params, model):
    acc, eps, opt_vecs, embaddings, _ = infer(params, model)

    embaddings = embaddings.cpu().detach().numpy()
    print(f'{acc=:.4f}')
    
    eps = np.concatenate((eps, eps, eps, eps, eps, eps, eps, eps, eps, eps))
    opt_vecs = np.concatenate((opt_vecs, opt_vecs, opt_vecs, opt_vecs, opt_vecs, opt_vecs, opt_vecs, opt_vecs, opt_vecs, opt_vecs))

    eps = eps[: 4000]
    opt_vecs = opt_vecs[: 4000]

    base_path = ''
    write_fvecs(f'', embaddings)


if __name__ == '__main__':
    with open(params['JSON_PATH'], 'r') as f:
        model_params = json.load(f)

    # load model
    MODEL_PATH = params['MODEL_PATH']
    model_path = f"{MODEL_PATH}/{get_model_name(model_params)}.pth"
    model = torch.load(model_path).to(device)

    main(params, model)
