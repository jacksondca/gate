import h5py

import numpy as np

from tqdm import tqdm


def read_ivecs(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    ret = a.reshape(-1, d + 1)[:, 1:].copy()
    return ret
 

def read_fvecs(fname):
    return read_ivecs(fname).view('float32')


def read_gt(fname):
    gt = []
    with open(fname) as fp:
        for line in fp.readlines():
            gt.append(list(map(int, line.split())))
    return gt


def read_bin_file(file_path, dim):
    data = np.fromfile(file_path, dtype='float32')
    return data.reshape(-1, dim).copy()


def read_nsg_index(filename):
    data = np.fromfile(filename, dtype='int32')

    def get_next():
        for d in tqdm(data, desc='reading nsg'):
            yield d

    gen = get_next()
    width, ep = next(gen), next(gen)

    graph = []
    for k in gen:
        graph.append([next(gen) for _ in range(k)])

    return width, ep, graph


def read_kmeans_file(filename):
    with open(filename, 'r') as fp:
        def get_next():
            for line in fp.readlines():
                yield line

        gen = get_next()
        
        iter1_num, iter2_num, dim = list(map(int, next(gen).split()))
        iter1_centroids = [list(map(float, next(gen).split())) for _ in range(iter1_num)]

        iter2_eps = [list(map(int, next(gen).split())) for _ in range(iter1_num)]

        return iter1_num, iter2_num, dim, iter1_centroids, iter2_eps


def write_fvecs(fname, data):
    data = np.array(data)
    dim = data.shape[1]
    int_data = np.full((data.shape[0], 1), dim, dtype='int32')
    fvecs = data.astype('float32')

    with open(fname, 'wb') as f:
        for i in range(fvecs.shape[0]):
            f.write(int_data[i].tobytes())
            f.write(fvecs[i].tobytes())


def write_spann_bin_file(fname, data):
    data = np.array(data)
    num, dim = data.shape[0], data.shape[1]

    num_data = np.full((data.shape[0], 1), num, dtype='int32')
    dim_data = np.full((data.shape[0], 1), dim, dtype='int32')

    with open(fname, 'wb') as f:
        f.write(num_data[0].tobytes())
        f.write(dim_data[0].tobytes())

        fvecs = data.astype('float32')
        for i in range(fvecs.shape[0]):
            f.write(fvecs[i].tobytes())


def write_spann_gt_file(fname, data):
    data = np.array(data)
    num, dim = data.shape[0], data.shape[1]
    num_data = np.full((data.shape[0], 1), num, dtype='int32')
    dim_data = np.full((data.shape[0], 1), dim, dtype='int32')
    ivecs = data.astype('int32')

    with open(fname, 'wb') as f:
        f.write(num_data[0].tobytes())
        f.write(dim_data[0].tobytes())

        for i in range(ivecs.shape[0]):
            f.write(ivecs[i].tobytes())

# WARNING: unchecked
def write_ivecs(fname, data):
    data = np.array(data)
    dim = data.shape[1]
    int_data = np.full((data.shape[0], 1), dim, dtype='int32')
    ivecs = data.astype('int32')

    with open(fname, 'wb') as f:
        for i in range(ivecs.shape[0]):
            f.write(int_data[i].tobytes())
            f.write(ivecs[i].tobytes())


def read_hdf5(file_name):
    """
    hdf5_file["train"] is base data
    hdf5_file['test'] is test data
    """
    hdf5_file = h5py.File(file_name, "r")
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    base = np.array(hdf5_file['train'])
    query = np.array(hdf5_file['test'])

    return base, query
