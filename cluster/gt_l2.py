import numpy as np
import torch
import tqdm
import time

from cluster.io import read_bin_file, read_fvecs


def main():
    query_batch_size = 5000  # 查询批次大小
    base_batch_size = 5000  # 库批次大小
    
    q_vecs = read_fvecs(query_file)
    # q_vecs = read_bin_file(f"{project_file}/{dataset_name}/query_{dataset_name}.bin", dim)
    # q_vecs = np.array(q_vecs, dtype='float32')[0:10000]

    # b_vecs = read_bin_file(f"{project_file}/{dataset_name}/database_{dataset_name}.bin", dim)
    # b_vecs = np.array(b_vecs, dtype='float32')
    b_vecs = read_fvecs(base_file)

    print(f"Query vectors: {q_vecs.shape[0]}")
    print(f"Base vectors: {b_vecs.shape[0]}")
    print(f"Query Batch size: {query_batch_size}")
    print(f"Base Batch size: {base_batch_size}")

    results = []

    for query_batch_start in tqdm.tqdm(range(0, q_vecs.shape[0], query_batch_size)):
        query_batch_end = min(query_batch_start + query_batch_size, q_vecs.shape[0])
        query_batch = q_vecs[query_batch_start: query_batch_end]
        query_vectors_gpu = torch.tensor(query_batch).cuda()

        batch_distances = []
        batch_indices = []

        for base_batch_start in range(0, b_vecs.shape[0], base_batch_size):
            base_batch_end = min(base_batch_start + base_batch_size, b_vecs.shape[0])
            base_batch = b_vecs[base_batch_start: base_batch_end]
            base_vectors_gpu = torch.tensor(base_batch).cuda()

            distances = torch.cdist(query_vectors_gpu, base_vectors_gpu, p=2.0)
            top_k_distances, top_k_indices = torch.topk(distances, k, largest=False)

            batch_distances.append(top_k_distances.cpu().numpy())
            batch_indices.append((top_k_indices + base_batch_start).cpu().numpy())

            del base_vectors_gpu
            torch.cuda.empty_cache()

        top_k_distances = np.hstack(batch_distances)
        top_k_indices = np.hstack(batch_indices)
            
        final_top_k_distances = np.zeros((query_batch.shape[0], k))
        final_top_k_indices = np.zeros((query_batch.shape[0], k), dtype=int)
        for i in range(query_batch.shape[0]):
            sorted_indices = np.argsort(top_k_distances[i])[:k]
            final_top_k_distances[i] = top_k_distances[i, sorted_indices]
            final_top_k_indices[i] = top_k_indices[i, sorted_indices]

        results.append((final_top_k_distances, final_top_k_indices))

        del query_vectors_gpu
        torch.cuda.empty_cache()

    s = time.time()
    with open(result_file, "w") as f:
        for batch in results:
            top_k_indices_cpu = batch[1]
            for indices in top_k_indices_cpu:
                f.write(" ".join(map(str, indices)) + "\n")

    t = time.time() - s
    print(f"Time: {t}")
    print(f"Results written to {result_file}")



if __name__ == '__main__':
    k = 100
    database = ''
    base_file = f''
    query_file = f''
    result_file = f""
    
    main()
