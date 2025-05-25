import json
import torch

import numpy as np


def read_ivecs(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()
 
def read_fvecs(fname):
    return read_ivecs(fname).view('float32')


def write_iter2_eps(f, centroid_num_stage1, centroid_num_stage2, data, fvecs):
    check_set = set()
    for iter1_id in range(centroid_num_stage1):
        eps = []
        for iter2_id in range(centroid_num_stage2 * iter1_id, centroid_num_stage2 * (iter1_id + 1)):
            # check data
            for id in data[str(iter2_id)][0]:
                if id in check_set:
                    raise Exception(f"{id} point in two cluster!!!")
                check_set.add(id)
                
            ids = data[str(iter2_id)][0]
            ids_cuda = torch.tensor(ids).cuda()
            vecs = fvecs[ids_cuda]
            centroid = vecs.mean(dim=0).reshape((1, -1))
            dist = torch.cdist(vecs, centroid)
            ep_index = int(torch.argmin(dist))
            ep = ids[ep_index]
            eps.append(int(ep))
            
        f.write(f"{' '.join(map(str, eps))}\n")


def main(json_path, result_path, fvecs_path):
    fvecs = read_fvecs(fvecs_path)
    fvecs = torch.tensor(fvecs).cuda()
    
    with open(json_path, 'r') as fp:
        json_obj = json.load(fp)
        data = json_obj['data']
        meta = json_obj['meta']
        
        with open(result_path, 'w') as f:
            # 1. write centroid_num_stage1 centroid_num_stage2
            centroid_num_stage1 = meta['centroid_num_stage1']
            centroid_num_stage2 = meta['centroid_num_stage2']
            f.write(f"{centroid_num_stage1} {centroid_num_stage2} {fvecs.shape[1]}\n")
            
            # # 2. write partition
            # for id in range(centroid_num_stage1 * centroid_num_stage2):
            #     sid = str(id)
            #     if sid not in data:
            #         raise Exception(f"{sid} is not exist, json file foramt error!")
                
            #     partition_list = data[sid][0]
            #     f.write(f"{' '.join(map(str, [len(partition_list), *partition_list]))}\n")
                
            # 3. cal and write iter1 centroid
            for id in range(centroid_num_stage1):
                partition_ids = []
                
                for item in range(centroid_num_stage2 * id, centroid_num_stage2 * (id + 1)):
                    l = data[str(item)][0]
                    partition_ids.extend(l)
                    
                partition_ids = torch.tensor(partition_ids).cuda()
                vecs = fvecs[partition_ids]
                centroid = vecs.mean(dim=0)
                f.write(f"{' '.join(map(str, centroid.tolist()))}\n")
                
            # 4. cal and write iter2 centroid ep
            write_iter2_eps(f, centroid_num_stage1, centroid_num_stage2, data, fvecs)

        


if __name__ == '__main__':
    database = ''
    kmeans = ''
    json_path = f''
    result_path = f''
    fvecs_path = f''
    
    main(json_path, result_path, fvecs_path)
    