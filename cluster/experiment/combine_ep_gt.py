import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from cluster.io import read_ivecs, write_fvecs, read_fvecs, read_gt, read_kmeans_file
from cluster.recall import recall


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NEG_SIZE = 64
iter1, iter2 = 32, 32
step = iter1 * iter2
dataset = 'sift'
k = 1

def write_smaple_file(file_path, samples):
    with open(file_path, 'w') as f:
        for eps in samples:
            f.write(f"{' '.join(map(str, eps))}\n")


def get_hop_count(hop_path, result_path, gt):
    hop_data = np.squeeze(read_ivecs(hop_path))

    result = np.squeeze(read_ivecs(result_path))

    for idx, (result, g) in enumerate(zip(result, gt)):
        if result not in g:
            hop_data[idx] = 999

    return hop_data


def sample_neg(array):
    array = array.copy()
    for idx, eps in enumerate(array):
        if eps == 999:
            array[idx] = -1

    # # bar = np.max(array) - 5
    # bar = np.min(array) + 5
    # # bar = (np.max(array) + np.min(array)) // 2
    # # bar = np.min(array) + 3
    # neg_eps = np.argwhere(array >= bar).reshape(-1).tolist()
    # return neg_eps

    hop_bar = 0
    while True:
        # bar = max(np.min(array) + hop_bar, np.max(array) - hop_bar)
        # bar = min(np.max(array), bar)
        bar = np.max(array) - hop_bar
        neg_eps = np.argwhere(array >= bar).reshape(-1).tolist()

        # if len(neg_eps) > NEG_SIZE or np.min(array) + hop_bar > np.max(array) - hop_bar:
        if len(neg_eps) > NEG_SIZE or bar == np.min(array) + 1:
            return neg_eps
        
        hop_bar = hop_bar + 1


def check(database_vecs, query_vecs, pos_sample_ids, neg_sample_ids):

    def _cal_vecs_dist(query, vecs):
        # distances = np.linalg.norm(vecs - query, axis=1)
        # average_distance = np.mean(distances)
        # return average_distance

        dot_products = vecs.dot(query)
        norm_vecs = np.linalg.norm(vecs, axis=1)
        norm_query = np.linalg.norm(query)
        cosine_similarities = dot_products / (norm_vecs * norm_query)
        average_cosine_similarity = np.mean(cosine_similarities)
        return average_cosine_similarity


    dists = []
    for query_vec, pos_ids, neg_ids in zip(query_vecs, pos_sample_ids, neg_sample_ids):
        pos_vecs = database_vecs[pos_ids]
        neg_vecs = database_vecs[neg_ids]

        pos_dist, neg_dist = _cal_vecs_dist(query_vec, pos_vecs), _cal_vecs_dist(query_vec, neg_vecs)
        dists.append((pos_dist, neg_dist))


def gen_train_data(dir):
    gt = np.array(read_gt(f''))
    base_file = dir / 'data' / 'hop_count_structured.ivecs'
    result_dir = Path(f'')

    structured = get_hop_count(base_file, result_dir / 'result_structured.ivecs', gt)


    arrays = []
    for i in tqdm(range(step)):
        hop_file = dir / 'data' / f'hop_count_ep{i}.ivecs'
        result_file = result_dir / f'result{i}.ivecs'

        array = get_hop_count(hop_file, result_file, gt)
        arrays.append(array)

    arrays = np.array(arrays).T

    breakpoint()

    opt_ids = []
    for qid, (struct, g) in enumerate(zip(structured, arrays)):
        if struct > np.min(g):
            opt_ids.append(qid)

    print(f"opt: {len(opt_ids)} / {len(structured)} = {len(opt_ids) / len(structured)}")

    query_vecs = read_fvecs(f"")
    opt_query_vecs = query_vecs
    arrays = arrays

    neg_sample_ids = []
    for array in arrays:
        neg_eps = sample_neg(array)
        neg_sample_ids.append(neg_eps)

    pos_sample_ids = []
    for array in arrays:
        # pos_eps = np.argwhere(array <= np.min(array) + HOP_THRESHOLD).reshape(-1).tolist()
        pos_eps = np.argwhere(array <= np.min(array)).reshape(-1).tolist()
        pos_sample_ids.append(pos_eps)


    d = Path(f'')
    d.mkdir(exist_ok=True, parents=True)
    write_smaple_file(d / 'pos_sample.txt', pos_sample_ids)
    write_smaple_file(d / 'neg_sample.txt', neg_sample_ids)
    write_fvecs(d / 'ground_truth.fvecs', [[ids[0]] for ids in pos_sample_ids])
    write_fvecs(d / 'opt_query1k.fvecs', opt_query_vecs)


    import collections
    print('hop opt statistics: ')
    print(collections.Counter([int(np.max(array) - np.min(array)) for array in arrays]))
    print(dict(collections.Counter([int(np.max(array) - np.min(array)) for array in arrays])))
    print(set([int(np.max(array) - np.min(array)) for array in arrays]))

    for qid, array in enumerate(arrays):
        if np.min(array) == np.max(array):
            print(f"{qid=}, {np.min(array)=}")
            pass
    #         breakpoint()
        if np.max(array) == 999:
            # print(qid)
            pass


if __name__ == '__main__':
    hop_dir = f''
    gen_train_data(Path(hop_dir))
