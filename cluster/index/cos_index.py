import time

import hnswlib
import numpy as np


from cluster.io import read_fvecs, write_fvecs



def main(query_path, embadding_path, output_path):
    embadding = read_fvecs(embadding_path)
    queries = read_fvecs(query_path)
    dim = embadding.shape[1]
    num_elements = embadding.shape[0]

    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements = num_elements, ef_construction=efc, M=M)

    ids = np.arange(num_elements)
    index.add_items(embadding, ids)

    index.set_ef(efs)

    count = 0
    for query in queries:
        labels, distances = index.knn_query(query, k=topk)
        labels = labels[0]
    
    s = time.time()
    labels, _ = index.knn_query(queries, k=topk)
    e = time.time()
    print(f'Time cost: {e - s}')
    # print(f"acc: {count / len(queries)}")
    
    write_fvecs(output_path, labels)



if __name__ == '__main__':
    
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-g', '--gt', dest='gt_path')
    parser.add_option('-q', '--queries', dest='queries')
    parser.add_option('-e', '--embaddings', dest='embaddings')
    parser.add_option('-o', '--output', dest='output')
    parser.add_option('--efs', type = 'int', dest='efs')
    parser.add_option('--efc', type = 'int', dest='efc')
    parser.add_option('-m', type = 'int', dest='m')
    parser.add_option('--topk', type = 'int', dest='topk')

    options, args = parser.parse_args()

    gt_path = options.gt_path
    query_path = options.queries
    embadding_path = options.embaddings
    output_path = options.output
    efs = options.efs
    efc = options.efc
    M = options.m
    topk = options.topk

    main(query_path, embadding_path, output_path)
